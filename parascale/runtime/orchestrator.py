# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:04
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Runtime orchestration for train, serve, and benchmark execution."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

from parascale.checkpoint import CheckpointManager
from parascale.config import ParaScaleConfig
from parascale.configuration import (
    config_artifact_overrides,
    write_config_artifacts,
)
from parascale.reporting.aggregation import aggregate_stable_metrics
from parascale.reporting.benchmark import benchmark_result_from_train_payload
from parascale.runtime.backends.devices import (
    npu_is_available,
    set_current_device,
)
from parascale.runtime.inference.engine import InferenceEngine
from parascale.runtime.training import TrainEngine
from parascale.strategy import build_strategy_plan
from parascale.workloads import (
    build_optimizer_for_model,
    build_serving_model_from_checkpoint,
    build_training_components,
)
from parascale.workloads.capability import capability_level_for_scope, describe_workload


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def run_train_from_config(
    config_data: Dict[str, Any], resume_step: int | None = None
) -> Dict[str, Any]:
    parascale_config = ParaScaleConfig.from_dict(_section(config_data, "parascale"))
    training = _section(config_data, "training")
    workload = str(training.get("workload", "synthetic_regression"))
    workload_capability = describe_workload(config_data)
    data_type = workload_capability.data_type
    requested_backend = str(parascale_config.training_backend)
    if requested_backend not in {
        "native",
        "native_ddp",
        "auto",
        "fsdp",
        "deepspeed",
        "ascend_native",
    }:
        raise ValueError(f"unsupported CLI training backend: {requested_backend}")
    if requested_backend == "auto":
        if _is_distributed_launch():
            strategy_plan = build_strategy_plan(
                _section(config_data, "model_profile"),
                _section(config_data, "hardware_profile"),
                parascale_config,
            )
            _apply_strategy_plan_to_config(parascale_config, strategy_plan)
        else:
            parascale_config.training_backend = "native"
    distributed_requested = parascale_config.training_backend in {
        "native_ddp",
        "fsdp",
        "deepspeed",
    }
    if distributed_requested and not _is_distributed_launch():
        raise ValueError(
            f"CLI backend '{parascale_config.training_backend}' requires a "
            "distributed launcher. Use torchrun/deepspeed launcher, or set "
            "parascale.training_backend=native for local smoke."
        )
    if distributed_requested:
        _initialize_distributed_for_cli(parascale_config.training_backend)

    config_artifacts = _write_runtime_config_artifacts(
        config_data,
        parascale_config,
        strategy_selected=requested_backend == "auto",
    )

    max_steps = int(training.get("max_steps", training.get("num_batches", 2)) or 2)
    checkpoint_dir = str(
        training.get("checkpoint_dir", parascale_config.checkpoint_save_path)
    )
    checkpoint_interval = int(
        training.get("checkpoint_interval", parascale_config.checkpoint_save_interval)
        or max_steps
    )
    flags = workload_capability.payload_flags()
    model, optimizer, dataloader, loss_fn = build_training_components(config_data)
    runtime_status = "synthetic" if flags["synthetic"] else "real_local"
    engine = TrainEngine(
        config=parascale_config,
        model_profile=_section(config_data, "model_profile"),
        hardware_profile=_section(config_data, "hardware_profile"),
    )
    manager = CheckpointManager(checkpoint_dir)
    if resume_step is None and training.get("resume_step") is not None:
        resume_step = int(training["resume_step"])
    resumed_from = None
    if resume_step is not None:
        manifest = engine.load_checkpoint(
            manager, int(resume_step), model=model, optimizer=optimizer
        )
        resumed_from = manifest.to_dict()

    start = time.perf_counter()
    optimizer_builder = (
        (lambda wrapped_model: build_optimizer_for_model(wrapped_model, config_data))
        if distributed_requested
        else None
    )
    state = engine.fit(
        dataloader,
        max_steps=max_steps,
        model=model,
        optimizer=optimizer,
        optimizer_builder=optimizer_builder,
        loss_fn=loss_fn,
        checkpoint_manager=manager,
        checkpoint_interval=checkpoint_interval,
    )
    elapsed = max(time.perf_counter() - start, 1e-9)
    if bool(training.get("skip_final_checkpoint", False)):
        final_manifest_path = None
        checkpoint_validation = {
            "ok": True,
            "skipped": True,
            "reason": "skip_final_checkpoint enabled for benchmark run",
        }
    else:
        final_manifest_path, checkpoint_validation = _validate_final_checkpoint_result(
            manager,
            engine.save_checkpoint(manager),
        )
    return {
        "mode": "train",
        "dry_run": False,
        "runtime_status": runtime_status,
        "capability_level": workload_capability.capability_level,
        **flags,
        "mock": False,
        "workload": workload,
        "data_type": data_type,
        "backend": parascale_config.training_backend,
        "global_step": state.global_step,
        "last_metrics": dict(state.last_metrics),
        "metrics_history": list(state.metrics_history),
        "train_device": _model_device(model),
        "elapsed_seconds": elapsed,
        "steps_per_second": max_steps / elapsed,
        "checkpoint": str(final_manifest_path) if final_manifest_path else None,
        "checkpoint_validation": checkpoint_validation,
        "resumed_from": resumed_from,
        "strategy_plan": engine.plan().to_dict(),
        "config_artifacts": config_artifacts,
    }


def _validate_final_checkpoint_result(
    manager: CheckpointManager,
    checkpoint_result: Any,
) -> tuple[Path | str | None, Dict[str, Any]]:
    if isinstance(checkpoint_result, (str, Path)):
        validation = manager.validate_manifest(
            manager.read_manifest_path(checkpoint_result)
        ).to_dict()
        return checkpoint_result, validation
    if isinstance(checkpoint_result, dict) and checkpoint_result.get("skipped"):
        return None, {
            "ok": True,
            "skipped": True,
            "rank": int(checkpoint_result.get("rank", 0) or 0),
            "reason": str(checkpoint_result.get("reason", "checkpoint save skipped")),
        }
    raise TypeError(
        "checkpoint save must return a manifest path or a rank-skipped result"
    )


def _write_runtime_config_artifacts(
    config_data: Dict[str, Any],
    runtime_config: ParaScaleConfig,
    *,
    strategy_selected: bool,
) -> Dict[str, str | None]:
    runtime = _section(config_data, "runtime")
    run_dir = runtime.get("run_dir")
    if not run_dir:
        return {
            "run_dir": None,
            "resolved_config": None,
            "deepspeed_final_config": None,
        }
    strategy_updates = None
    if strategy_selected:
        strategy_updates = {
            "backend.training_backend": runtime_config.training_backend,
            "backend.zero_stage": runtime_config.zero_stage,
            "backend.zero_offload": runtime_config.zero_offload,
            "backend.precision": runtime_config.precision,
        }
    overrides = config_artifact_overrides(config_data)
    return write_config_artifacts(
        config_data,
        run_dir,
        cli_overrides=overrides.get("cli_overrides"),
        strategy_updates=strategy_updates or overrides.get("strategy_updates"),
        emergency_overrides=overrides.get("emergency_overrides"),
    )


def _apply_strategy_plan_to_config(config: ParaScaleConfig, plan: Any) -> None:
    config.training_backend = plan.backend
    config.precision = plan.precision
    config.enable_activation_checkpointing = plan.activation_checkpointing
    config.batching_strategy = plan.batch_policy
    if plan.max_tokens_per_batch is not None:
        config.max_tokens_per_batch = plan.max_tokens_per_batch
    if plan.backend == "fsdp":
        config.fsdp_state_dict_type = plan.fsdp_state_dict_type
    if plan.backend == "deepspeed":
        config.zero_stage = plan.zero_stage
        config.zero_offload = plan.zero_offload
    if plan.backend == "native_ddp":
        config.ddp_comm_hook = plan.ddp_comm_hook
        config.ddp_gradient_as_bucket_view = plan.ddp_gradient_as_bucket_view
        config.ddp_static_graph = plan.ddp_static_graph


def _is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1") or 1) > 1


def _capability_level_for_scope(base_level: str, config_data: Dict[str, Any]) -> str:
    return capability_level_for_scope(base_level, config_data)


def _initialize_distributed_for_cli(backend: str) -> None:
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        raise ImportError(
            f"CLI backend '{backend}' requires torch and torch.distributed."
        ) from exc

    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    set_current_device(torch, local_rank=local_rank)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if not dist.is_initialized():
        dist_backend = _distributed_backend_for_torch(torch)
        dist.init_process_group(backend=dist_backend)


def _destroy_distributed_for_cli() -> None:
    try:
        import torch.distributed as dist
    except Exception:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_backend_for_torch(torch: Any) -> str:
    if torch.cuda.is_available():
        return "nccl"
    if npu_is_available(torch):
        return "hccl"
    return "gloo"


def _distributed_rank() -> int:
    return int(os.environ.get("RANK", "0") or 0)


def _model_device(model: Any) -> str:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "unknown"
    try:
        return str(next(parameters()).device)
    except StopIteration:
        return "unknown"
    except Exception:
        return "unknown"


def run_serve_from_config(
    config_data: Dict[str, Any], checkpoint: str | None = None
) -> Dict[str, Any]:
    serving = _section(config_data, "serving")
    checkpoint = checkpoint or serving.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "parascale serve requires --checkpoint or serving.checkpoint for real execution."
        )
    manager = _checkpoint_manager_for_path(checkpoint)
    manifest = manager.read_manifest_path(checkpoint)
    checkpoint_validation = manager.validate_manifest(manifest)
    if not checkpoint_validation.ok:
        raise RuntimeError(
            f"checkpoint validation failed: {checkpoint_validation.to_dict()}"
        )
    mock = bool(serving.get("mock", False))
    if mock:
        engine = (
            InferenceEngine(config=config_data)
            .initialize(world_size=1)
            .load_model(checkpoint=manifest, mock=True)
        )
        runtime_status = "mock"
        capability_level = "manifest_load_validation"
    else:
        model = build_serving_model_from_checkpoint(config_data, manifest, manager)
        engine = (
            InferenceEngine(config=config_data)
            .initialize(world_size=1)
            .load_model(model=model)
        )
        runtime_status = "real_local"
        capability_level = "local_tiny_torch_checkpoint"
    requests = serving.get("requests", ["hello"])
    result = engine.generate(requests)
    return {
        "mode": "serve",
        "dry_run": False,
        "runtime_status": runtime_status,
        "capability_level": capability_level,
        "mock": mock,
        "checkpoint": str(checkpoint),
        "checkpoint_validation": checkpoint_validation.to_dict(),
        "manifest": manifest.to_dict(),
        "result": result,
    }


def _checkpoint_manager_for_path(checkpoint: str | Path) -> CheckpointManager:
    path = Path(checkpoint)
    if path.is_file() and path.name == "manifest.json":
        return CheckpointManager(str(path.parent.parent))
    if path.is_dir() and path.name.startswith("step-"):
        return CheckpointManager(str(path.parent))
    return CheckpointManager(str(path))


def run_benchmark_from_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    training = dict(_section(config_data, "training"))
    training.setdefault("max_steps", int(training.get("benchmark_steps", 3) or 3))
    benchmark_config = dict(config_data)
    benchmark_config["training"] = training
    start = time.perf_counter()
    train_payload = run_train_from_config(benchmark_config, resume_step=None)
    elapsed = max(time.perf_counter() - start, 1e-9)
    validation = _build_benchmark_validation_payload(
        benchmark_config, train_payload, training
    )
    last_metrics = dict(train_payload.get("last_metrics", {}))
    warmup_steps = int(training.get("warmup_steps", 0) or 0)
    stable_metrics = aggregate_stable_metrics(
        train_payload.get("metrics_history", []), warmup_steps=warmup_steps
    )
    benchmark_result = benchmark_result_from_train_payload(train_payload)
    task_type = _section(config_data, "parascale").get("task_type", "generic")
    workload = str(training.get("workload", "synthetic_regression"))
    return {
        "mode": "benchmark",
        "dry_run": False,
        "runtime_status": train_payload.get("runtime_status", "real_local"),
        "capability_level": train_payload.get("capability_level", "local_benchmark"),
        "synthetic": bool(train_payload.get("synthetic", False)),
        "benchmark_type": f"{workload}_train",
        "elapsed_seconds": elapsed,
        "metrics": {
            "steps_per_second": train_payload["steps_per_second"],
            "step_time_ms": 1000.0 / max(train_payload["steps_per_second"], 1e-9),
            "warmup_steps": warmup_steps,
            "measured_steps": int(stable_metrics.get("measured_steps", 0)),
            "samples_per_second": benchmark_result.metric("samples_per_second"),
            "tokens_per_second": float(
                last_metrics.get("tokens_per_second", 0.0) or 0.0
            ),
            "end_to_end_tokens_per_second": float(
                last_metrics.get("end_to_end_tokens_per_second", 0.0) or 0.0
            ),
            "images_per_second": float(
                last_metrics.get("images_per_second", 0.0) or 0.0
            ),
            "end_to_end_images_per_second": float(
                last_metrics.get("end_to_end_images_per_second", 0.0) or 0.0
            ),
            "patch_tokens_per_second": float(
                last_metrics.get("patch_tokens_per_second", 0.0) or 0.0
            ),
            "end_to_end_patch_tokens_per_second": float(
                last_metrics.get("end_to_end_patch_tokens_per_second", 0.0) or 0.0
            ),
            "image_text_pairs_per_second": float(
                last_metrics.get("image_text_pairs_per_second", 0.0) or 0.0
            ),
            "end_to_end_image_text_pairs_per_second": float(
                last_metrics.get("end_to_end_image_text_pairs_per_second", 0.0) or 0.0
            ),
            "padding_ratio": float(last_metrics.get("padding_ratio", 0.0) or 0.0),
            "peak_memory_bytes": float(
                last_metrics.get("peak_memory_bytes", 0.0) or 0.0
            ),
            "allocated_memory_bytes": float(
                last_metrics.get("allocated_memory_bytes", 0.0) or 0.0
            ),
            "dataloader_wait_ms": float(
                last_metrics.get("dataloader_wait_ms", 0.0) or 0.0
            ),
            "adapter_params": int(last_metrics.get("adapter_params", 0) or 0),
            "trainable_params": int(last_metrics.get("trainable_params", 0) or 0),
            "total_params": int(last_metrics.get("total_params", 0) or 0),
            "trainable_ratio": float(last_metrics.get("trainable_ratio", 0.0) or 0.0),
            "lora_rank": int(last_metrics.get("lora_rank", 0) or 0),
            **stable_metrics,
        },
        "validation": validation,
        "benchmark_result": benchmark_result.to_dict(),
        "comparison_contract": {
            "primary_metrics": [
                "samples_per_second",
                "images_per_second",
                "patch_tokens_per_second",
                "image_text_pairs_per_second",
                "step_time_ms",
            ],
            "baseline_backends": ["fsdp", "deepspeed"],
            "requires_same_workload": True,
            "requires_same_hardware": True,
            "task_type": task_type,
            "workload": workload,
        },
        "config_artifacts": train_payload.get("config_artifacts", {}),
        "train": train_payload,
    }


def _build_benchmark_validation_payload(
    config_data: Dict[str, Any],
    train_payload: Dict[str, Any],
    training: Dict[str, Any],
) -> Dict[str, Any]:
    checkpoint_validation = train_payload.get("checkpoint_validation", {})
    validation: Dict[str, Any] = {
        "checkpoint": {
            "ok": bool(checkpoint_validation.get("ok", False)),
            "skipped": bool(checkpoint_validation.get("skipped", False)),
            "details": checkpoint_validation,
        },
        "resume": {
            "ok": None,
            "skipped": True,
            "reason": "set training.validate_resume=true to run resume validation",
        },
    }
    if not bool(training.get("validate_resume", False)):
        return validation

    checkpoint = train_payload.get("checkpoint")
    global_step = int(train_payload.get("global_step", 0) or 0)
    if not checkpoint or checkpoint_validation.get("skipped"):
        validation["resume"] = {
            "ok": False,
            "skipped": False,
            "reason": "resume validation requires a saved checkpoint",
        }
        return validation

    resume_steps = int(training.get("resume_validation_steps", 1) or 1)
    resume_config = dict(config_data)
    resume_training = dict(training)
    resume_training["max_steps"] = max(1, resume_steps)
    resume_training["benchmark_steps"] = max(1, resume_steps)
    resume_training["skip_final_checkpoint"] = bool(
        training.get("skip_resume_final_checkpoint", True)
    )
    resume_config["training"] = resume_training

    resume_start = time.perf_counter()
    try:
        resumed = run_train_from_config(resume_config, resume_step=global_step)
        validation["resume"] = {
            "ok": int(resumed.get("global_step", 0) or 0) >= global_step + resume_steps,
            "skipped": False,
            "elapsed_seconds": time.perf_counter() - resume_start,
            "resumed_from_step": global_step,
            "global_step": resumed.get("global_step"),
            "backend_state_loaded": bool(
                resumed.get("resumed_from", {})
                .get("metadata", {})
                .get("backend_state_loaded", False)
            ),
        }
    except Exception as exc:
        validation["resume"] = {
            "ok": False,
            "skipped": False,
            "elapsed_seconds": time.perf_counter() - resume_start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return validation
