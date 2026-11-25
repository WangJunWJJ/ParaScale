# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Production-facing training runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from parascale.core import (
    ClusterTopology,
    CollectiveBackend,
    CpuDeviceBackend,
    DeviceBackend,
    MockCollectiveBackend,
)
from parascale.parallel.capabilities import validate_legacy_parallelism
from parascale.runtime.backends import TrainingBackend, create_runtime_training_backend
from parascale.runtime.lifecycle import execution_world_size
from parascale.strategy import StrategyPlan, build_strategy_plan

from .accumulation import AccumulationController
from .checkpointing import CheckpointController
from .coordination import coordinated_call
from .data_state import DataResumeController
from .fit_loop import FitLoopRunner
from .memory import RuntimeMemoryTracker
from .metrics import (
    RuntimeMetrics,
    add_pipeline_profile_metrics,
    merge_accumulated_batches,
    merge_pipeline_profiles,
    metric_value,
    sum_metric_value,
)
from .precision import PrecisionController
from .prefetch import maybe_cuda_prefetch_iterator
from .step import StepRunner


@dataclass
class TrainState:
    initialized: bool = False
    global_step: int = 0
    last_metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_history: list[Dict[str, Any]] = field(default_factory=list)
    metrics_history_dropped: int = 0
    consumed_micro_batches: int = 0
    consumed_samples: int = 0
    data_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainEngine:
    config: Any
    model_profile: Any = field(default_factory=dict)
    hardware_profile: Any = field(default_factory=dict)
    topology: Optional[ClusterTopology] = None
    device_backend: DeviceBackend = field(default_factory=CpuDeviceBackend)
    collective: CollectiveBackend = field(default_factory=MockCollectiveBackend)
    strategy_plan: Optional[StrategyPlan] = None
    training_backend: Optional[TrainingBackend] = None
    state: TrainState = field(default_factory=TrainState)
    amp_scaler: Any = None
    memory: RuntimeMemoryTracker = field(default_factory=RuntimeMemoryTracker)
    precision: Any = None
    step_runner: Any = None
    fit_loop: Any = None
    data_resume: Any = None
    objective_metadata: Dict[str, Any] = field(default_factory=dict)
    scheduler: Any = None
    scheduler_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.precision is None:
            self.precision = PrecisionController(self)
        if self.step_runner is None:
            self.step_runner = StepRunner(self)
        if self.fit_loop is None:
            self.fit_loop = FitLoopRunner(self)
        if self.data_resume is None:
            self.data_resume = DataResumeController(self)

    def plan(self) -> StrategyPlan:
        validate_legacy_parallelism(self.config, self.strategy_plan)
        world_size = execution_world_size() or 1
        if self.strategy_plan is None:
            self.strategy_plan = build_strategy_plan(
                self.model_profile, self.hardware_profile, self.config,
                world_size=world_size,
            )
        if self.strategy_plan.dp_size != world_size:
            raise ValueError("Cached strategy plan conflicts with execution world_size")
        return self.strategy_plan

    def _validated_plan(self) -> StrategyPlan:
        return coordinated_call(
            self.plan, phase="runtime plan",
            identity=lambda value: value.to_dict(),
        )

    def initialize(self) -> "TrainEngine":
        plan = self._validated_plan()
        world_size = max(1, plan.dp_size * plan.tp_size * plan.pp_size)
        self.collective.init_process_group(
            world_size=world_size,
            rank=self._distributed_rank(),
        )
        if self.training_backend is None:
            self.training_backend = create_runtime_training_backend(
                config=self.config, local_rank=self._local_rank()
            )
        self.state.initialized = True
        return self

    def setup(self) -> "TrainEngine":
        return self.initialize()

    def fit(
        self,
        dataloader: Any = None,
        max_steps: Optional[int] = None,
        *,
        model: Any = None,
        optimizer: Any = None,
        optimizer_builder: Any = None,
        scheduler: Any = None,
        scheduler_builder: Any = None,
        loss_fn: Any = None,
        step_fn: Any = None,
        checkpoint_manager: Any = None,
        checkpoint_interval: Optional[int] = None,
    ) -> TrainState:
        validate_legacy_parallelism(self.config, self.strategy_plan)
        if dataloader is None:
            raise ValueError("TrainEngine.fit requires a dataloader.")
        if step_fn is None and (model is None or optimizer is None or loss_fn is None):
            raise RuntimeError(
                "TrainEngine.fit requires either step_fn(batch, engine) or model, optimizer and loss_fn."
            )
        self._validate_quantized_training(model, step_fn)
        if not self.state.initialized:
            self.initialize()
        if model is not None or optimizer is not None:
            if (
                self.training_backend is not None
                and getattr(self.training_backend, "model", None) is not None
            ):
                model = self.training_backend.model
                optimizer = self.training_backend.optimizer
            else:
                backend_name = getattr(self.config, "training_backend", "native")
                if optimizer_builder is not None and backend_name == "deepspeed":
                    optimizer = optimizer_builder(model)
                self.training_backend = create_runtime_training_backend(
                    model=model,
                    optimizer=optimizer,
                    config=self.config,
                    local_rank=self._local_rank(),
                )
                model, optimizer = self.training_backend.setup()
                if optimizer_builder is not None and backend_name != "deepspeed":
                    optimizer = optimizer_builder(model)
                    optimizer = self.training_backend.setup_optimizer(optimizer)
                    self.training_backend.optimizer = optimizer

        self._bind_loss_data_parallel(loss_fn)
        scheduler = self.prepare_scheduler(optimizer, scheduler, scheduler_builder)
        return self.fit_loop.run(
            dataloader,
            max_steps=max_steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            step_fn=step_fn,
            checkpoint_manager=checkpoint_manager,
            checkpoint_interval=checkpoint_interval,
        )

    def _bind_loss_data_parallel(self, loss_fn: Any) -> None:
        """Bind opt-in objectives to this runtime's validated pure-DP group."""
        bind = getattr(loss_fn, "bind_data_parallel", None)
        if not callable(bind):
            return
        validate_legacy_parallelism(self.config, self.strategy_plan)
        group = None
        if getattr(self.training_backend, "name", "") in {"native_ddp", "fsdp", "deepspeed"}:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed objective requires an initialized process group")
            group = dist.group.WORLD
        bind(group)

    def prepare_scheduler(
        self, optimizer: Any, scheduler: Any = None, scheduler_builder: Any = None
    ) -> Any:
        """Collectively bind and retain the scheduler for the final optimizer."""
        prepared = coordinated_call(
            lambda: self._prepare_scheduler(optimizer, scheduler, scheduler_builder),
            phase="scheduler",
            identity=lambda value: {
                "configuration": self.scheduler_metadata,
                "type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                    if value is not None else None
                ),
                "state": value.state_dict() if value is not None else None,
            },
        )
        self.scheduler = prepared
        return prepared

    def _prepare_scheduler(
        self, optimizer: Any, scheduler: Any, scheduler_builder: Any
    ) -> Any:
        if scheduler_builder is not None:
            if scheduler is not None or self.scheduler is not None:
                raise ValueError("scheduler_builder cannot replace an existing scheduler")
            scheduler = scheduler_builder(optimizer)
        elif scheduler is None:
            scheduler = self.scheduler
        if scheduler is not None:
            pending, visited = [scheduler], set()
            while pending:
                current = pending.pop()
                if id(current) in visited:
                    continue
                visited.add(id(current))
                if hasattr(current, "lr_lambdas") and not self.scheduler_metadata:
                    raise ValueError(
                        "Function-based scheduler requires explicit scheduler_metadata"
                    )
                pending.extend(getattr(current, "_schedulers", ()))
            if getattr(scheduler, "optimizer", optimizer) is not optimizer:
                raise ValueError("scheduler must be bound to the final backend optimizer")
            if self.scheduler is not None and scheduler is not self.scheduler:
                raise ValueError("cannot replace the active scheduler during training")
        return scheduler

    def _run_accumulated_step(
        self,
        first_batch: Any,
        iterator: Any,
        *,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        loss_fn: Any = None,
        step_fn: Any = None,
        dataloader_wait_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        return AccumulationController(self).run(
            first_batch,
            iterator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            step_fn=step_fn,
            dataloader_wait_seconds=dataloader_wait_seconds,
        )

    def _run_step(
        self,
        batch: Any,
        *,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        loss_fn: Any = None,
        step_fn: Any = None,
    ) -> Dict[str, Any]:
        return self.step_runner.run_step(
            batch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            step_fn=step_fn,
        )

    def _call_step_fn(self, step_fn: Any, batch: Any) -> Any:
        return self.step_runner.call_step_fn(step_fn, batch)

    def _with_throughput_metrics(
        self, metrics: Dict[str, Any], batch: Any, elapsed_seconds: float
    ) -> Dict[str, Any]:
        return RuntimeMetrics(world_size=self._world_size()).with_throughput_metrics(
            metrics,
            batch,
            elapsed_seconds,
        )

    def _add_end_to_end_metrics(
        self, metrics: Dict[str, Any], batch: Any, dataloader_wait_seconds: float
    ) -> None:
        RuntimeMetrics(world_size=self._world_size()).add_end_to_end_metrics(
            metrics,
            batch,
            dataloader_wait_seconds,
        )

    def _add_pipeline_profile_metrics(
        self, metrics: Dict[str, Any], batch: Dict[str, Any]
    ) -> None:
        add_pipeline_profile_metrics(metrics, batch)

    def _maybe_cuda_prefetch_iterator(self, iterator: Iterator[Any]) -> Iterator[Any]:
        return maybe_cuda_prefetch_iterator(
            iterator,
            config=self.config,
            local_rank=self._local_rank(),
        )

    @staticmethod
    def _metric_value(value: Any) -> Any:
        return metric_value(value)

    @staticmethod
    def _sum_metric_value(value: Any) -> Any:
        return sum_metric_value(value)

    def _merge_accumulated_batches(self, batches: list[Any]) -> Any:
        return merge_accumulated_batches(batches)

    @staticmethod
    def _merge_pipeline_profiles(profiles: list[Dict[str, Any]]) -> Dict[str, float]:
        return merge_pipeline_profiles(profiles)

    def train_step(self, batch: Any) -> TrainState:
        size = len(batch) if hasattr(batch, "__len__") else 1
        return self.record_step({"batch_size": size})

    def backward(self, loss: Any) -> None:
        if self.precision.backward(loss):
            return
        backend = self.training_backend or create_runtime_training_backend(
            config=self.config, local_rank=self._local_rank()
        )
        backend.backward(loss)

    def step(self, optimizer: Any = None) -> bool:
        if self.precision.step(optimizer):
            return self.precision.optimizer_updated
        backend = self.training_backend or create_runtime_training_backend(
            config=self.config, local_rank=self._local_rank()
        )
        clip_norm = getattr(self.config, "grad_clip_norm", None)
        if clip_norm is not None and backend.name != "deepspeed":
            backend.clip_gradients(float(clip_norm), optimizer)
        return backend.step(optimizer) is not False

    def _validate_quantized_training(self, model: Any, step_fn: Any) -> None:
        config = getattr(self.config, "quantization", None)
        if config is None or not config.enabled:
            return
        if config.mode != "qat":
            raise ValueError("PTQ cannot be used for training; use the quantize command")
        if (
            self.config.training_backend != "native"
            or self.config.precision != "fp32"
            or self.config.adapter_only_checkpoint
            or self._world_size() != 1
        ):
            raise ValueError("QAT requires single-process native FP32 and full checkpoints")
        from parascale.quantization.fake_quantize import (
            FakeQuantizedConv2d,
            FakeQuantizedLinear,
        )

        quantizers = [m.weight_fake_quant for m in model.modules()
                      if isinstance(m, (FakeQuantizedLinear, FakeQuantizedConv2d))] if model is not None else []
        if step_fn is not None or not quantizers:
            raise ValueError("QAT requires a prepared model from QuantizationAwareTraining.prepare()")
        if any(m.config.to_dict() != config.to_dict() for m in quantizers):
            raise ValueError("prepared QAT model configuration does not match the runtime")

    def step_scheduler(self, scheduler: Any = None) -> None:
        if scheduler is not None and hasattr(scheduler, "step"):
            scheduler.step()

    def evaluate(self, dataloader: Any = None) -> Dict[str, Any]:
        samples = (
            len(dataloader)
            if dataloader is not None and hasattr(dataloader, "__len__")
            else 0
        )
        return {"samples": samples, "global_step": self.state.global_step}

    def save_checkpoint(
        self,
        checkpoint_manager: Any = None,
        step: Optional[int] = None,
        *,
        scheduler: Any = None,
    ) -> Any:
        return CheckpointController(self).save(
            checkpoint_manager,
            step,
            scheduler=scheduler,
        )

    def _distributed_rank(self) -> int:
        dist = self._initialized_torch_distributed()
        if dist is not None:
            return int(dist.get_rank())
        if "RANK" in os.environ:
            return int(os.environ["RANK"])
        return int(getattr(self.collective, "rank", 0) or 0)

    def _distributed_world_size(self) -> int:
        dist = self._initialized_torch_distributed()
        if dist is not None:
            return max(1, int(dist.get_world_size()))
        return max(1, int(getattr(self.collective, "world_size", 1) or 1))

    def _distributed_barrier(self) -> None:
        dist = self._initialized_torch_distributed()
        if dist is not None:
            dist.barrier()
            return
        if self._distributed_world_size() > 1:
            self.collective.barrier()

    @staticmethod
    def _initialized_torch_distributed() -> Any | None:
        try:
            import torch.distributed as dist
        except Exception:
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        return dist

    def load_checkpoint(
        self,
        checkpoint_manager: Any,
        step: int,
        *,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        dataloader: Any = None,
        optimizer_builder: Any = None,
        scheduler_builder: Any = None,
    ) -> Any:
        self._validated_plan()
        return CheckpointController(self).load(
            checkpoint_manager,
            step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            optimizer_builder=optimizer_builder,
            scheduler_builder=scheduler_builder,
        )

    def record_step(self, metrics: Optional[Dict[str, Any]] = None) -> TrainState:
        self.state.global_step += 1
        self.state.last_metrics = dict(metrics or {})
        self.state.metrics_history.append(dict(metrics or {}))
        limit = int(getattr(self.config, "metrics_history_limit", 1000))
        excess = max(0, len(self.state.metrics_history) - limit)
        if excess:
            del self.state.metrics_history[:excess]
            self.state.metrics_history_dropped += excess
        return self.state

    def shutdown(self) -> None:
        self.collective.shutdown()
        self.state.initialized = False

    @staticmethod
    def _local_rank() -> int:
        return int(os.environ.get("LOCAL_RANK", "0") or 0)

    @staticmethod
    def _world_size() -> int:
        return execution_world_size() or 1

    def _reset_peak_memory_stats(self) -> None:
        self.memory.reset_peak_memory_stats()

    def _add_peak_memory_metrics(self, metrics: Dict[str, Any]) -> None:
        self.memory.add_peak_memory_metrics(metrics)

    def _synchronize_device(self) -> None:
        self.memory.synchronize_device()

    def _elapsed_since(self, start: float, *, synchronized: bool = False) -> float:
        return self.memory.elapsed_since(start, synchronized=synchronized)

    def _backend_name(self) -> str:
        return str(
            getattr(
                self.training_backend,
                "name",
                getattr(self.config, "training_backend", "native"),
            )
        )

    def _gradient_accumulation_steps(self) -> int:
        return max(1, int(getattr(self.config, "gradient_accumulation_steps", 1) or 1))

    def _setup_amp_scaler(self) -> None:
        self.precision.setup_scaler()

    def _autocast_context(self):
        return self.precision.autocast_context()
