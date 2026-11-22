# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free, bounded configuration for native stage training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from parascale.config import ParaScaleConfig
from parascale.configuration.resolver import normalize_execution_config, resolve_config
from parascale.optimizers.schedule import SchedulerSpec
from parascale.optimizers.spec import OptimizerSpec
from parascale.parallel.distributed import DistributedParallelConfig
from parascale.workloads.specs.causal_lm import CausalLmSpec

_FIXED_BATCH_DEFAULTS = {
    "batching_strategy": "sample",
    "max_tokens_per_batch": None,
    "max_patch_tokens_per_batch": None,
    "resolution_buckets": [],
    "dataloader_drop_last": False,
    "drop_last": False,
}


@dataclass(frozen=True)
class StageTrainingPlan:
    config_data: dict
    config: ParaScaleConfig
    topology: DistributedParallelConfig
    workload: CausalLmSpec
    max_steps: int
    checkpoint_dir: Path
    checkpoint_interval: int
    keep_last: int
    metrics_history_limit: int
    timeout_seconds: int
    optimizer_spec: OptimizerSpec
    scheduler_spec: SchedulerSpec | None

    def to_dict(self) -> dict[str, Any]:
        """Describe exact topology and unique-sample accounting, without assets."""
        payload = {
            "training_backend": "native_parallel",
            "config_data": self.config_data,
            "config": self.config.to_dict(),
            "topology": asdict(self.topology),
            "workload": asdict(self.workload),
            "world_size": self.topology.world_size,
            "global_batch_size": self.topology.global_batch_size(
                self.workload.batch_size
            ),
            "max_steps": self.max_steps,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_interval": self.checkpoint_interval,
            "keep_last": self.keep_last,
            "metrics_history_limit": self.metrics_history_limit,
            "timeout_seconds": self.timeout_seconds,
            "optimizer_spec": asdict(self.optimizer_spec),
            "scheduler_spec": (
                self.scheduler_spec.to_metadata() if self.scheduler_spec else None
            ),
        }
        return json.loads(json.dumps(payload, allow_nan=False))


def is_native_parallel(config_data: dict) -> bool:
    return resolve_config(config_data).backend["training_backend"] == "native_parallel"


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _local_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError(f"{name} must be a nonempty local path string")
    return value


def build_stage_training_plan(config_data: dict) -> StageTrainingPlan:
    resolved = resolve_config(config_data)
    if resolved.backend["training_backend"] != "native_parallel":
        raise ValueError("stage training requires training_backend=native_parallel")
    # Materialized defaults are not requests. Inspect history so aliases and
    # normalized configs retain the same unsupported-feature boundary.
    forbidden_axes = {
        "parallel.model_parallel_size",
        "parallel.pipeline_parallel_chunks",
    }
    enabled_only = {
        "training.enable_activation_checkpointing",
        "training.grad_clip_norm",
        "training.allow_world_size_change_on_resume",
        "training.adapter_only_checkpoint",
        "data.device_prefetch",
        "data.cuda_prefetch",
        "data.prefetch_device",
        "data.cuda_prefetch_device",
    }
    for path, field in resolved.fields.items():
        requests = [
            item for item in field.history if item["source"] != "built-in defaults"
        ]
        unsupported_backend = path.startswith("backend.") and path not in {
            "backend.training_backend",
            "backend.precision",
        }
        if requests and (path in forbidden_axes or unsupported_backend):
            raise ValueError(f"native_parallel does not support explicit {path}")
        if path in enabled_only and any(item["value"] for item in requests):
            raise ValueError(f"native_parallel does not support {path}")
        if path == "backend.precision" and any(
            item["value"] == "fp16" for item in requests
        ):
            raise ValueError("native_parallel does not support fp16")
        key = path.split(".", 1)[1]
        if (
            path.startswith("data.")
            and key in _FIXED_BATCH_DEFAULTS
            and any(item["value"] != _FIXED_BATCH_DEFAULTS[key] for item in requests)
        ):
            raise ValueError(f"native_parallel uses fixed token blocks; remove {path}")
    if resolved.backend["deepspeed_config"]:
        raise ValueError("native_parallel does not support deepspeed_config")

    data, config = normalize_execution_config(config_data)
    training = data.get("training", {})
    if "optimizer" in training:
        raise ValueError(
            "native_parallel does not support training.optimizer; "
            "move these settings to root-level optimizer"
        )
    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a mapping")
    if runtime.get("device", "auto") not in ("cuda", "auto"):
        raise ValueError("native_parallel requires runtime.device=cuda or auto")
    if runtime.get("accelerator", "auto") not in ("cuda", "auto"):
        raise ValueError("native_parallel requires runtime.accelerator=cuda or auto")
    for alias, guidance in {
        "checkpoint_dir": "use training.checkpoint_save_path",
        "checkpoint_interval": "use training.checkpoint_save_interval",
        "skip_final_checkpoint": "remove it; the final checkpoint is mandatory",
    }.items():
        if alias in training:
            raise ValueError(
                f"native_parallel does not support training.{alias}; {guidance}"
            )
    # These workload-level aliases are retained, but not interpreted, by the
    # resolver. Canonical ParaScale data fields are checked via history above.
    for section in ("data", "training"):
        options = data.get(section, {})
        if isinstance(options, dict):
            for key, default in _FIXED_BATCH_DEFAULTS.items():
                if key in options and options[key] != default:
                    raise ValueError(
                        f"native_parallel uses fixed token blocks; remove {section}.{key}"
                    )
    if training.get("workload") != "causal_lm":
        raise ValueError("native_parallel requires training.workload=causal_lm")
    if config.quantization.enabled:
        raise ValueError("native_parallel does not support quantization/QAT")
    distillation = data.get("distillation", {})
    if not isinstance(distillation, dict) or distillation.get("enabled", False):
        raise ValueError("native_parallel does not support distillation")
    for section, key in (("model", "pretrained_model_name_or_path"), ("data", "path")):
        value = data.get(section, {})
        _local_path(
            value.get(key) if isinstance(value, dict) else None, f"{section}.{key}"
        )
    workload = CausalLmSpec.from_config(data)
    if workload.activation_checkpointing:
        raise ValueError("native_parallel does not support activation checkpointing")
    _integer(workload.shuffle_window_blocks, "data.shuffle_window_blocks")
    topology = DistributedParallelConfig(
        dp_size=config.data_parallel_size,
        tp_size=config.tensor_parallel_size,
        pp_size=config.pipeline_parallel_size,
        microbatches=config.gradient_accumulation_steps,
        pipeline_schedule=config.pipeline_schedule,
    )
    topology.global_batch_size(workload.batch_size)
    for section in ("launch", "distributed"):
        declared = data.get(section, {})
        if not isinstance(declared, dict):
            raise ValueError(f"{section} must be a mapping")
        for key, expected in {
            "nnodes": 1,
            "nproc_per_node": topology.world_size,
            "world_size": topology.world_size,
            "node_rank": 0,
        }.items():
            if key in declared:
                actual = _integer(
                    declared[key], f"{section}.{key}", 0 if key == "node_rank" else 1
                )
                if actual != expected:
                    raise ValueError(
                        f"{section}.{key} must be {expected} for the single-node stage topology"
                    )
    optimizer = OptimizerSpec.from_config(data)
    if optimizer.is_four_bit:
        raise ValueError(
            "native_parallel requires optimizer.type=adamw; four-bit state is not qualified"
        )
    scheduler = SchedulerSpec.from_config(data)
    return StageTrainingPlan(
        config_data=data,
        config=config,
        topology=topology,
        workload=workload,
        max_steps=_integer(training.get("max_steps"), "training.max_steps"),
        checkpoint_dir=Path(
            _local_path(config.checkpoint_save_path, "checkpoint_save_path")
        ),
        checkpoint_interval=_integer(
            config.checkpoint_save_interval, "checkpoint_save_interval"
        ),
        keep_last=_integer(config.checkpoint_keep_last, "checkpoint_keep_last", 2),
        metrics_history_limit=_integer(
            config.metrics_history_limit, "metrics_history_limit", 0
        ),
        timeout_seconds=_integer(
            runtime.get("distributed_timeout_seconds", 180),
            "runtime.distributed_timeout_seconds",
        ),
        optimizer_spec=optimizer,
        scheduler_spec=scheduler,
    )
