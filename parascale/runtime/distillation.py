# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Connect online distillation to existing workload builders and checkpoints."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from parascale.config import ParaScaleConfig
from parascale.distillation import DistillationConfig
from parascale.runtime.lifecycle import is_distributed_launch


def validate_distillation_request(
    config_data: dict[str, Any],
    runtime: ParaScaleConfig,
) -> tuple[DistillationConfig, dict[str, Any]]:
    """Validate the CLI execution boundary before constructing any model.

    Returns the objective configuration and teacher source specification.
    Raises ValueError for unsupported execution combinations or missing sources.
    """
    section = config_data.get("distillation", {})
    if not isinstance(section, dict):
        raise ValueError("distillation must be a mapping")
    options = dict(section)
    teacher = options.pop("teacher", {})
    config = DistillationConfig.from_dict(options)
    if not config.enabled:
        return config, {}
    if runtime.training_backend != "native" or is_distributed_launch():
        raise ValueError(
            "distillation currently requires a single-process native backend"
        )
    if any(
        getattr(runtime, field) != 1
        for field in (
            "data_parallel_size",
            "model_parallel_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
        )
    ):
        raise ValueError(
            "distillation currently requires single-process parallel sizes of one"
        )
    if runtime.precision != "fp32":
        raise ValueError("distillation currently requires precision=fp32")
    if runtime.adapter_only_checkpoint:
        raise ValueError("distillation requires full checkpoints")
    if runtime.quantization.enabled:
        raise ValueError("combined distillation and quantization is not yet validated")
    training = config_data.get("training", {})
    if (
        runtime.gradient_accumulation_steps != 1
        or training.get("gradient_accumulation_steps", 1) != 1
    ):
        raise ValueError("distillation gradient accumulation is not yet validated")
    if (
        not isinstance(teacher, dict)
        or set(teacher) != {"config", "checkpoint"}
        or not isinstance(teacher["config"], dict)
        or not isinstance(teacher["checkpoint"], str)
        or not teacher["checkpoint"]
    ):
        raise ValueError(
            "distillation.teacher requires config mapping and checkpoint path"
        )
    if not Path(teacher["checkpoint"]).is_file():
        raise ValueError("distillation teacher checkpoint does not exist")
    return config, deepcopy(teacher)


def build_distillation_objective(
    student: Any,
    config: DistillationConfig,
    teacher_spec: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Load a frozen teacher and return a loss callable plus a resume contract.

    Accepts a tensor state dictionary or a native backend_state.pt checkpoint.
    Teacher state and optimizer are never embedded in student checkpoints.
    Invalid or incomplete teacher state raises ValueError or RuntimeError.
    """
    import torch

    from parascale.distillation import OnlineDistillationLoss
    from parascale.workloads import build_training_components

    source = Path(teacher_spec["checkpoint"])
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "backend_state" in state:
        backend = state["backend_state"]
        state = backend.get("model_state_dict") if isinstance(backend, dict) else None
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(item, torch.Tensor) for item in state.values())
    ):
        raise ValueError(
            "teacher checkpoint must contain a full tensor model_state_dict"
        )
    teacher_config = teacher_spec["config"]
    config_digest = hashlib.sha256(
        json.dumps(
            teacher_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    # Teacher initialization must not alter the student's dropout/RNG trajectory.
    with torch.random.fork_rng():
        teacher, teacher_optimizer, teacher_data, _ = build_training_components(
            teacher_config
        )
    del teacher_optimizer, teacher_data
    teacher.load_state_dict(state, strict=True)
    for model in (student, teacher):
        if any(
            p.is_floating_point() and p.dtype != torch.float32
            for p in model.parameters()
        ):
            raise ValueError("native distillation models must have fp32 parameters")
    objective = OnlineDistillationLoss(student, teacher, config)
    metadata = {
        "kind": "online_logits_distillation",
        "schema_version": 1,
        "config": config.to_dict(),
        "teacher_checkpoint_sha256": digest.hexdigest(),
        "teacher_config_sha256": config_digest,
    }
    return objective, metadata
