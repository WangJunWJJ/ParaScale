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
from parascale.configuration import normalize_execution_config
from parascale.optimizers.spec import OptimizerSpec
from parascale.runtime.distillation import (
    build_detection_distillation_components,
    build_distillation_objective,
    validate_distillation_request,
)
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
    _, runtime_config = normalize_execution_config(config_data)
    accumulation = runtime_config.gradient_accumulation_steps
    if consumed % accumulation:
        raise ValueError(
            "resume consumed_micro_batches must end at an accumulation boundary"
        )
    component_training["max_steps"] = consumed // accumulation + max(0, int(max_steps))
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
    config_data, parascale_config = normalize_execution_config(config_data)
    training = _section(config_data, "training")
    workload = str(training.get("workload", "synthetic_regression"))
    workload_capability = describe_workload(config_data)
    data_type = workload_capability.data_type
    requested_backend = str(parascale_config.training_backend)
    distillation, teacher_spec = validate_distillation_request(
        config_data, parascale_config
    )
    if parascale_config.quantization.enabled:
        if parascale_config.quantization.mode != "qat":
            raise ValueError(
                "Use the quantize command for PTQ; train requires mode=qat."
            )
        if requested_backend != "native" or is_distributed_launch():
            raise ValueError("QAT currently requires a single-process native backend.")
        if parascale_config.precision != "fp32":
            raise ValueError("QAT currently requires precision=fp32.")
        if parascale_config.adapter_only_checkpoint:
            raise ValueError(
                "QAT requires full checkpoints, not adapter_only_checkpoint."
            )
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

    if requested_backend == "auto":
        from parascale.config import flat_config_fields_by_section

        _, initial_config = normalize_execution_config(config_data)
        updates = {
            ("backend.precision" if key == "precision" else f"{section}.{key}"): value
            for section, keys in flat_config_fields_by_section().items()
            for key in keys
            if (value := getattr(parascale_config, key)) != getattr(initial_config, key)
        }
        config_data, parascale_config = normalize_execution_config(
            config_data, strategy_updates=updates
        )
        training = _section(config_data, "training")
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
    student_output = None
    if distillation.enabled and distillation.task == "detection":
        student_output = Path(
            training.get("student_output", Path(checkpoint_dir) / "student.pt")
        )
        resolved_output = student_output.resolve()
        checkpoint_root = Path(checkpoint_dir).resolve()
        if resolved_output.is_relative_to(checkpoint_root):
            relative = resolved_output.relative_to(checkpoint_root)
            if relative.parts and relative.parts[0].startswith("step-"):
                raise ValueError(
                    "student_output must not be inside checkpoint step directories"
                )
        protected = [teacher_spec["checkpoint"]]
        for source_config in (config_data, teacher_spec["config"]):
            source_model = _section(source_config, "model")
            source_path = source_model.get("path", source_model.get("model_path"))
            if source_path:
                protected.append(source_path)
        if any(student_output.resolve() == Path(path).resolve() for path in protected):
            raise ValueError(
                "student_output must not overwrite a source model or teacher checkpoint"
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
    quantizer = None
    if parascale_config.quantization.enabled:
        from parascale.quantization import QuantizationAwareTraining

        quantizer = QuantizationAwareTraining(model, parascale_config.quantization)
        model = quantizer.prepare()
    objective_metadata = {}
    if distillation.enabled:
        if distillation.task == "detection":
            model, loss_fn, objective_metadata = (
                build_detection_distillation_components(
                    model, distillation, teacher_spec, loss_fn
                )
            )
        else:
            loss_fn, objective_metadata = build_distillation_objective(
                model, distillation, teacher_spec
            )
    optimizer = build_optimizer_for_model(model, config_data)
    runtime_status = "synthetic" if flags["synthetic"] else "real_local"
    engine = TrainEngine(
        config=parascale_config,
        model_profile=_section(config_data, "model_profile"),
        hardware_profile=_section(config_data, "hardware_profile"),
        objective_metadata=objective_metadata,
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
    quantization_result = None
    student_checkpoint = None
    if student_output is not None:
        import torch

        output = student_output
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in model.student.wrapped_model.state_dict().items()
            },
            output,
        )
        student_checkpoint = str(output)
    if quantizer is not None:
        output = str(
            training.get("quantized_output", Path(checkpoint_dir) / "quantized.pt")
        )
        quantizer.freeze_observer()
        quantizer.export(output)
        quantization_result = {
            "mode": "qat",
            "bits": parascale_config.quantization.bits,
            "checkpoint": output,
            "execution": "dequantize_float_reference",
        }
    return attach_runtime_evidence(
        {
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
            "quantization": quantization_result,
            "distillation": {
                "objective": objective_metadata,
                "last_components": dict(loss_fn.last_metrics),
                "student_checkpoint": student_checkpoint,
            }
            if distillation.enabled
            else None,
            "strategy_plan": engine.plan().to_dict(),
            "config_artifacts": config_artifacts,
        }
    )


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
