# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Shared helpers for workload spec parsing."""

from __future__ import annotations

from typing import Any, Dict


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def training_micro_batches(config_data: Dict[str, Any]) -> int:
    """Translate the optimizer-update budget into a workload batch window."""
    from parascale.config import ParaScaleConfig

    training = _section(config_data, "training")
    runtime = ParaScaleConfig.from_dict(_section(config_data, "parascale"))
    steps = int(training.get("max_steps", training.get("num_batches", 2)) or 2)
    accumulation = int(training.get("gradient_accumulation_steps", runtime.gradient_accumulation_steps))
    return steps * max(1, accumulation)


__all__ = ["_section"]
