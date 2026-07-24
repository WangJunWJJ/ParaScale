# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:04
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Shared helpers for runtime execution runners."""

from __future__ import annotations

from typing import Any, Dict

from parascale.config import ParaScaleConfig
from parascale.configuration import config_artifact_overrides, write_config_artifacts


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}

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
        config.ddp_bucket_cap_mb = plan.ddp_bucket_cap_mb
        config.ddp_gradient_as_bucket_view = plan.ddp_gradient_as_bucket_view
        config.ddp_static_graph = plan.ddp_static_graph
