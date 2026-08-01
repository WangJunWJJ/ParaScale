# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:04
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Training execution runner."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from parascale.checkpoint import CheckpointManager
from parascale.config import ParaScaleConfig
from parascale.optimizers.spec import OptimizerSpec
from parascale.runtime.evidence import attach_runtime_evidence
from parascale.runtime.lifecycle import (
    distributed_rank,
    initialize_distributed_for_backend,
    is_distributed_launch,
    model_device,
    validate_distributed_topology,
)
from parascale.runtime.runner_common import (
    _apply_strategy_plan_to_config,
    _section,
    _write_runtime_config_artifacts,
)
from parascale.runtime.training import TrainEngine
from parascale.strategy import build_strategy_plan
from parascale.workloads import build_optimizer_for_model, build_training_components
from parascale.workloads.capability import capability_level_for_scope, describe_workload


def _resume_component_config(
    config_data: Dict[str, Any], manifest: Any, *, max_steps: int
) -> Dict[str, Any]:
    """Expand only the workload build window for replay-based data resume."""
    data_state = dict(getattr(manifest, "data_state", {}) or {})
    if str(data_state.get("resume_mode", "")) != "replay_skip":
        return config_data
    consumed = max(0, int(data_state.get("consumed_micro_batches", 0) or 0))
    if consumed == 0:
        return config_data
    component_config = deepcopy(config_data)
    component_training = component_config.setdefault("training", {})
    component_training["max_steps"] = consumed + max(0, int(max_steps))
    return component_config


def _rank_component_config(
    config_data: Dict[str, Any], *, rank: int, distributed: bool
) -> Dict[str, Any]:
    """Derive a deterministic rank-specific data seed for distributed input."""
    if not distributed:
        return config_data
    component_config = deepcopy(config_data)
    training = component_config.setdefault("training", {})
    training["seed"] = int(training.get("seed", 42) or 42) + int(rank)
    return component_config

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
        if is_distributed_launch():
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
    if distributed_requested and not is_distributed_launch():
        raise ValueError(
            f"CLI backend '{parascale_config.training_backend}' requires a "
            "distributed launcher. Use torchrun/deepspeed launcher, or set "
            "parascale.training_backend=native for local smoke."
        )
    if distributed_requested:
        validate_distributed_topology(config_data)
        initialize_distributed_for_backend(parascale_config.training_backend)

    optimizer_spec = OptimizerSpec.from_config(config_data)
    optimizer_spec.validate_backend(
        parascale_config.training_backend,
        zero_stage=int(parascale_config.zero_stage or 0),
    )

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
    manager = CheckpointManager(checkpoint_dir)
    if resume_step is None and training.get("resume_step") is not None:
        resume_step = int(training["resume_step"])
    resume_manifest = (
        manager.read_manifest(int(resume_step)) if resume_step is not None else None
    )
    component_config = (
        _resume_component_config(config_data, resume_manifest, max_steps=max_steps)
        if resume_manifest is not None
        else config_data
    )
    component_config = _rank_component_config(
        component_config,
        rank=distributed_rank(),
        distributed=distributed_requested,
    )
    flags = workload_capability.payload_flags()
    model, optimizer, dataloader, loss_fn = build_training_components(component_config)
    optimizer = build_optimizer_for_model(model, config_data)
    runtime_status = "synthetic" if flags["synthetic"] else "real_local"
    engine = TrainEngine(
        config=parascale_config,
        model_profile=_section(config_data, "model_profile"),
        hardware_profile=_section(config_data, "hardware_profile"),
    )
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
    return attach_runtime_evidence({
        "mode": "train",
        "dry_run": False,
        "runtime_status": runtime_status,
        "capability_level": workload_capability.capability_level,
        **flags,
        "mock": False,
        "workload": workload,
        "data_type": data_type,
        "backend": parascale_config.training_backend,
        "optimizer": dict(
            getattr(optimizer, "_parascale_optimizer_metadata", {}) or {}
        ),
        "global_step": state.global_step,
        "last_metrics": dict(state.last_metrics),
        "metrics_history": list(state.metrics_history),
        "train_device": model_device(model),
        "elapsed_seconds": elapsed,
        "steps_per_second": max_steps / elapsed,
        "checkpoint": str(final_manifest_path) if final_manifest_path else None,
        "checkpoint_validation": checkpoint_validation,
        "resumed_from": resumed_from,
        "strategy_plan": engine.plan().to_dict(),
        "config_artifacts": config_artifacts,
    })

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

def _capability_level_for_scope(base_level: str, config_data: Dict[str, Any]) -> str:
    return capability_level_for_scope(base_level, config_data)
