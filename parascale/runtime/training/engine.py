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
from parascale.runtime.backends import TrainingBackend, create_runtime_training_backend
from parascale.strategy import StrategyPlan, build_strategy_plan

from .accumulation import AccumulationController
from .checkpointing import CheckpointController
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

    def __post_init__(self) -> None:
        if self.precision is None:
            self.precision = PrecisionController(self)
        if self.step_runner is None:
            self.step_runner = StepRunner(self)
        if self.fit_loop is None:
            self.fit_loop = FitLoopRunner(self)

    def plan(self) -> StrategyPlan:
        if self.strategy_plan is None:
            self.strategy_plan = build_strategy_plan(
                self.model_profile, self.hardware_profile, self.config
            )
        return self.strategy_plan

    def initialize(self) -> "TrainEngine":
        plan = self.plan()
        world_size = max(1, plan.dp_size * plan.tp_size * plan.pp_size)
        self.collective.init_process_group(world_size=world_size, rank=0)
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
        loss_fn: Any = None,
        step_fn: Any = None,
        checkpoint_manager: Any = None,
        checkpoint_interval: Optional[int] = None,
    ) -> TrainState:
        if dataloader is None:
            raise ValueError("TrainEngine.fit requires a dataloader.")
        if step_fn is None and (model is None or optimizer is None or loss_fn is None):
            raise RuntimeError(
                "TrainEngine.fit requires either step_fn(batch, engine) or model, optimizer and loss_fn."
            )
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

    def step(self, optimizer: Any = None) -> None:
        if self.precision.step(optimizer):
            return
        backend = self.training_backend or create_runtime_training_backend(
            config=self.config, local_rank=self._local_rank()
        )
        backend.step(optimizer)

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

    def load_checkpoint(
        self,
        checkpoint_manager: Any,
        step: int,
        *,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
    ) -> Any:
        return CheckpointController(self).load(
            checkpoint_manager,
            step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def record_step(self, metrics: Optional[Dict[str, Any]] = None) -> TrainState:
        self.state.global_step += 1
        self.state.last_metrics = dict(metrics or {})
        self.state.metrics_history.append(dict(metrics or {}))
        return self.state

    def shutdown(self) -> None:
        self.collective.shutdown()
        self.state.initialized = False

    @staticmethod
    def _local_rank() -> int:
        return int(os.environ.get("LOCAL_RANK", "0") or 0)

    @staticmethod
    def _world_size() -> int:
        return max(1, int(os.environ.get("WORLD_SIZE", "1") or 1))

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
