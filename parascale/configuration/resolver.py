# -*- coding: utf-8 -*-
# @Time : 2026/6/26 下午12:07
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Resolve user, workload, backend, and CLI config into one snapshot."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

from parascale.config import ParaScaleConfig, flat_config_fields_by_section

from .resolved import ConfigIssue, ResolvedConfig, ResolvedField


@dataclass(frozen=True)
class ConfigPatch:
    path: str
    value: Any
    source: str
    priority: int
    reason: str | None = None


def resolve_config(
    config_data: Dict[str, Any],
    *,
    cli_overrides: Dict[str, Any] | None = None,
    strategy_updates: Dict[str, Any] | None = None,
    emergency_overrides: Dict[str, Any] | None = None,
    dry_run: bool = False,
) -> ResolvedConfig:
    config_data = deepcopy(config_data)
    parsed = ParaScaleConfig.from_dict(_section(config_data, "parascale"))
    parsed_values = parsed.to_dict()
    config_data["parascale"] = {
        key: parsed_values[key] for key in parsed._explicit_fields
    }
    nested_ds = parsed.deepspeed_config
    top_ds = config_data.get("deepspeed_config")
    if nested_ds is not None and top_ds is not None and nested_ds != top_ds:
        raise ValueError("conflicting parascale.deepspeed_config and deepspeed_config")
    config_data["deepspeed_config"] = deepcopy(top_ds or nested_ds or {})
    metadata = _section(config_data, "_resolution")
    cli_overrides = {**metadata.get("cli_overrides", {}), **(cli_overrides or {})}
    strategy_updates = {
        **metadata.get("strategy_updates", {}),
        **(strategy_updates or {}),
    }
    emergency_overrides = {
        **metadata.get("emergency_overrides", {}),
        **(emergency_overrides or {}),
    }
    patches = list(_base_patches())
    materialized = metadata.get("materialized_fields", {})
    restored = set()
    # Copies made for workload consumers must not become new user overrides.
    for patch in _user_config_patches(config_data):
        prior = materialized.get(patch.path, {})
        if prior and prior.get("value") == patch.value:
            if patch.path not in restored:
                patches.extend(
                    ConfigPatch(**item)
                    for item in prior["history"]
                    if item["source"] == "user config"
                )
                restored.add(patch.path)
        else:
            patches.append(patch)
    patches.extend(_deepspeed_native_patches(config_data))
    patches.extend(
        _mapping_patches(strategy_updates or {}, source="strategy/tuner", priority=80)
    )
    patches.extend(_mapping_patches(cli_overrides or {}, source="cli", priority=90))
    patches.extend(
        _mapping_patches(
            emergency_overrides or {}, source="emergency override", priority=100
        )
    )

    fields = _resolve_fields(patches)
    hardware = _section(config_data, "hardware_profile")
    runtime = {
        "dry_run": bool(dry_run),
        "world_size": _world_size(hardware),
    }
    backend = {
        "training_backend": _value(fields, "backend.training_backend"),
        "zero_stage": _value(fields, "backend.zero_stage"),
        "zero_offload": _value(fields, "backend.zero_offload"),
        "precision": _value(fields, "backend.precision"),
        "deepspeed_config": dict(_section(config_data, "deepspeed_config")),
    }
    workload = {
        "name": str(_section(config_data, "training").get("workload", "unknown")),
        "task_type": str(_section(config_data, "task").get("type", "generic")),
    }
    data = dict(_section(config_data, "data"))
    training = {
        "batch_size": _value(fields, "training.batch_size"),
        "gradient_accumulation_steps": _value(
            fields, "training.gradient_accumulation_steps"
        ),
        "learning_rate": _value(fields, "training.learning_rate"),
    }
    optimizer = dict(_section(config_data, "optimizer"))
    warnings = _validate_deepspeed(
        fields, config_data, world_size=runtime["world_size"]
    )

    return ResolvedConfig(
        runtime=runtime,
        backend=backend,
        workload=workload,
        data=data,
        training=training,
        optimizer=optimizer,
        hardware=hardware,
        fields=fields,
        warnings=warnings,
        errors=[],
    )


def normalize_execution_config(
    config_data: Dict[str, Any],
    *,
    cli_overrides: Dict[str, Any] | None = None,
    strategy_updates: Dict[str, Any] | None = None,
    emergency_overrides: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], ParaScaleConfig]:
    """Return isolated workload input and runtime config using resolver priority.

    Existing top-level training aliases win over ParaScale settings; optimizer.lr
    wins over learning-rate aliases. Explicit ParaScale fields win over native
    DeepSpeed settings, followed by strategy, CLI and emergency overrides.
    """
    data = deepcopy(config_data)
    resolved = resolve_config(
        data,
        cli_overrides=cli_overrides,
        strategy_updates=strategy_updates,
        emergency_overrides=emergency_overrides,
    )
    values = {}
    for path, field in resolved.fields.items():
        values[path.split(".", 1)[1]] = deepcopy(field.value)
    original = ParaScaleConfig.from_dict(_section(data, "parascale"))
    values["quantization"] = original.quantization
    values["deepspeed_config"] = resolved.backend["deepspeed_config"] or None
    config = ParaScaleConfig.from_dict(values)
    data["parascale"] = deepcopy(config.to_dict())
    training = data.setdefault("training", {})
    if "backend" in training:
        training["backend"] = config.training_backend
    for key in flat_config_fields_by_section()["training"]:
        training[key] = deepcopy(getattr(config, key))
    training["lr"] = config.learning_rate
    if "batch_size" in _section(data, "data"):
        data["data"]["batch_size"] = config.batch_size
    data.setdefault("optimizer", {})["lr"] = config.learning_rate
    if config.deepspeed_config:
        data["deepspeed_config"] = deepcopy(config.deepspeed_config)
    metadata = data.setdefault("_resolution", {})
    metadata["materialized_fields"] = {
        path: field.to_dict() for path, field in resolved.fields.items()
    }
    for name, overrides in (
        ("cli_overrides", cli_overrides),
        ("strategy_updates", strategy_updates),
        ("emergency_overrides", emergency_overrides),
    ):
        if overrides:
            metadata[name] = {**metadata.get(name, {}), **overrides}
    return data, config


def build_deepspeed_final_config(resolved: ResolvedConfig) -> Dict[str, Any]:
    zero_stage = int(resolved.field("backend.zero_stage").value)
    precision = str(resolved.field("backend.precision").value)
    ds_config = dict(resolved.backend.get("deepspeed_config", {}) or {})
    zero_config = dict(ds_config.get("zero_optimization", {}) or {})
    zero_config.update(
        {
            "stage": zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "ignore_unused_parameters": True,
        }
    )
    if zero_stage >= 3:
        zero_config.update(
            {
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1_000_000_000,
                "stage3_max_reuse_distance": 1_000_000_000,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        )
    if bool(resolved.field("backend.zero_offload").value):
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if zero_stage >= 3:
            zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}

    ds_config.update(
        {
            "train_micro_batch_size_per_gpu": int(
                resolved.field("training.batch_size").value
            ),
            "gradient_accumulation_steps": int(
                resolved.field("training.gradient_accumulation_steps").value
            ),
            "zero_optimization": zero_config,
        }
    )
    ds_config.setdefault("steps_per_print", 1_000_000)
    ds_config.setdefault("wall_clock_breakdown", False)
    if precision == "fp16":
        ds_config["fp16"] = {"enabled": True}
        ds_config["bf16"] = {"enabled": False}
    elif precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
        ds_config["fp16"] = {"enabled": False}
    else:
        ds_config["fp16"] = {"enabled": False}
        ds_config["bf16"] = {"enabled": False}
    ds_config["_parascale"] = {
        **dict(ds_config.get("_parascale", {})),
        "resolved_config": True,
        "warnings": [issue.to_dict() for issue in resolved.warnings],
    }
    grad_clip = resolved.fields.get("training.grad_clip_norm")
    if grad_clip is not None and grad_clip.value is not None:
        ds_config["gradient_clipping"] = grad_clip.value
    if resolved.backend.get("deepspeed_config"):
        ds_config["_parascale"].update(
            {
                "merged_user_config": True,
                "enforced_keys": [
                    "train_micro_batch_size_per_gpu",
                    "gradient_accumulation_steps",
                    "precision",
                    "zero_optimization.stage",
                ],
            }
        )
    return ds_config


def _base_patches() -> Iterable[ConfigPatch]:
    config = ParaScaleConfig()
    for section, keys in flat_config_fields_by_section().items():
        for key in keys:
            if key not in {
                "training_backend",
                "zero_stage",
                "zero_offload",
                "precision",
                "batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
            }:
                yield ConfigPatch(
                    f"{section}.{key}", getattr(config, key), "built-in defaults", 0
                )
    yield ConfigPatch(
        "backend.training_backend", config.training_backend, "built-in defaults", 0
    )
    yield ConfigPatch("backend.zero_stage", config.zero_stage, "built-in defaults", 0)
    yield ConfigPatch(
        "backend.zero_offload", config.zero_offload, "built-in defaults", 0
    )
    yield ConfigPatch("backend.precision", config.precision, "built-in defaults", 0)
    yield ConfigPatch("training.batch_size", config.batch_size, "built-in defaults", 0)
    yield ConfigPatch(
        "training.gradient_accumulation_steps",
        config.gradient_accumulation_steps,
        "built-in defaults",
        0,
    )
    yield ConfigPatch(
        "training.learning_rate", config.learning_rate, "built-in defaults", 0
    )


def _user_config_patches(config_data: Dict[str, Any]) -> Iterable[ConfigPatch]:
    parascale = _section(config_data, "parascale")
    training = _section(config_data, "training")
    optimizer = _section(config_data, "optimizer")
    path_map = {
        "training_backend": "backend.training_backend",
        "zero_stage": "backend.zero_stage",
        "zero_offload": "backend.zero_offload",
        "precision": "backend.precision",
        "batch_size": "training.batch_size",
        "gradient_accumulation_steps": "training.gradient_accumulation_steps",
        "learning_rate": "training.learning_rate",
    }
    for key, path in path_map.items():
        if key in parascale:
            yield ConfigPatch(path, parascale[key], "user config", 40)
    if "backend" in training:
        if "training_backend" in parascale and training["backend"] != parascale["training_backend"]:
            raise ValueError("training.backend conflicts with parascale.training_backend")
        yield ConfigPatch("backend.training_backend", training["backend"], "user config", 40)
    for section, keys in flat_config_fields_by_section().items():
        for key in keys:
            if key not in path_map and key in parascale:
                yield ConfigPatch(f"{section}.{key}", parascale[key], "user config", 40)
    for key in flat_config_fields_by_section()["training"]:
        if key in training and key not in {
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
        }:
            path = "backend.precision" if key == "precision" else f"training.{key}"
            yield ConfigPatch(path, training[key], "user config", 40)
    if "batch_size" in training:
        yield ConfigPatch(
            "training.batch_size", training["batch_size"], "user config", 40
        )
    if "gradient_accumulation_steps" in training:
        yield ConfigPatch(
            "training.gradient_accumulation_steps",
            training["gradient_accumulation_steps"],
            "user config",
            40,
        )
    if "learning_rate" in training:
        yield ConfigPatch(
            "training.learning_rate", training["learning_rate"], "user config", 40
        )
    if "lr" in training:
        yield ConfigPatch("training.learning_rate", training["lr"], "user config", 40)
    if "lr" in optimizer:
        yield ConfigPatch("training.learning_rate", optimizer["lr"], "user config", 40)
    data = _section(config_data, "data")
    if "batch_size" in data:
        yield ConfigPatch("training.batch_size", data["batch_size"], "user config", 40)


def _deepspeed_native_patches(config_data: Dict[str, Any]) -> Iterable[ConfigPatch]:
    ds_config = _section(config_data, "deepspeed_config")
    for native, path in {
        "train_micro_batch_size_per_gpu": "training.batch_size",
        "gradient_accumulation_steps": "training.gradient_accumulation_steps",
    }.items():
        if native in ds_config:
            yield ConfigPatch(path, ds_config[native], "deepspeed_config.json", 35)
    zero_config = ds_config.get("zero_optimization")
    if isinstance(zero_config, Mapping) and "stage" in zero_config:
        yield ConfigPatch(
            "backend.zero_stage",
            zero_config["stage"],
            "deepspeed_config.json",
            35,
            reason="DeepSpeed zero_optimization.stage",
        )


def _mapping_patches(
    values: Dict[str, Any], *, source: str, priority: int
) -> Iterable[ConfigPatch]:
    valid_paths = {patch.path for patch in _base_patches()}
    for path, value in values.items():
        if path == "training.precision":
            path = "backend.precision"
        if path not in valid_paths:
            raise ValueError(f"unknown or incorrectly owned configuration override: {path}")
        yield ConfigPatch(path, value, source, priority)


def _resolve_fields(patches: Iterable[ConfigPatch]) -> Dict[str, ResolvedField]:
    grouped: Dict[str, list[ConfigPatch]] = {}
    for patch in patches:
        grouped.setdefault(patch.path, []).append(patch)
    fields: Dict[str, ResolvedField] = {}
    for path, path_patches in grouped.items():
        ordered = sorted(path_patches, key=lambda item: item.priority)
        winner = ordered[-1]
        overridden_by = [winner.source] if len(ordered) > 1 else []
        history = [
            {
                "path": patch.path,
                "value": patch.value,
                "source": patch.source,
                "priority": patch.priority,
                "reason": patch.reason,
            }
            for patch in ordered
        ]
        fields[path] = ResolvedField(
            path=path,
            value=winner.value,
            source=winner.source,
            overridden_by=overridden_by,
            history=history,
            reason=winner.reason,
        )
    return fields


def _validate_deepspeed(
    fields: Dict[str, ResolvedField],
    config_data: Dict[str, Any],
    *,
    world_size: int,
) -> list[ConfigIssue]:
    if _value(fields, "backend.training_backend") != "deepspeed":
        return []
    issues: list[ConfigIssue] = []
    ds_config = _section(config_data, "deepspeed_config")
    precision = str(_value(fields, "backend.precision"))
    if precision == "bf16" and _enabled(ds_config.get("fp16")):
        issues.append(
            ConfigIssue(
                code="deepspeed_precision_conflict",
                path="deepspeed_config.fp16.enabled",
                message="ParaScale precision=bf16 conflicts with DeepSpeed fp16.enabled=true.",
            )
        )
    if precision == "fp16" and _enabled(ds_config.get("bf16")):
        issues.append(
            ConfigIssue(
                code="deepspeed_precision_conflict",
                path="deepspeed_config.bf16.enabled",
                message="ParaScale precision=fp16 conflicts with DeepSpeed bf16.enabled=true.",
            )
        )
    expected_global_batch = (
        int(_value(fields, "training.batch_size"))
        * int(_value(fields, "training.gradient_accumulation_steps"))
        * int(world_size)
    )
    if (
        "train_batch_size" in ds_config
        and int(ds_config["train_batch_size"]) != expected_global_batch
    ):
        issues.append(
            ConfigIssue(
                code="deepspeed_train_batch_mismatch",
                path="deepspeed_config.train_batch_size",
                message=(
                    "DeepSpeed train_batch_size does not match "
                    "micro_batch * gradient_accumulation_steps * world_size "
                    f"({expected_global_batch})."
                ),
            )
        )
    if "optimizer" in ds_config:
        issues.append(
            ConfigIssue(
                code="deepspeed_optimizer_conflict",
                path="deepspeed_config.optimizer",
                message=(
                    "DeepSpeed JSON optimizer is present while ParaScale may pass "
                    "a Python optimizer object; choose one optimizer owner."
                ),
            )
        )
    return issues


def _value(fields: Dict[str, ResolvedField], path: str) -> Any:
    return fields[path].value


def _enabled(section: Any) -> bool:
    return isinstance(section, Mapping) and bool(section.get("enabled", False))


def _world_size(hardware: Dict[str, Any]) -> int:
    for key in ("world_size", "num_gpus", "gpus_per_node"):
        value = hardware.get(key)
        if value:
            return int(value)
    return 1


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}
