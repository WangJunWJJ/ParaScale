# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:21
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free capability checks for the legacy training runtime."""

from __future__ import annotations

from typing import Any


def validate_legacy_parallelism(config: Any, strategy_plan: Any = None) -> None:
    """Reject unsupported execution dimensions without modifying either input.

    Args:
        config: Configuration object or dictionary, optionally None.
        strategy_plan: Optional supplied strategy object or dictionary.

    Raises:
        ValueError: A requested model-parallel dimension exceeds one.
    """
    unsupported = []
    for source, names in (
        (
            config,
            ("tensor_parallel_size", "pipeline_parallel_size", "model_parallel_size"),
        ),
        (strategy_plan, ("tp_size", "pp_size")),
    ):
        for name in names:
            value = (
                source.get(name, 1)
                if isinstance(source, dict)
                else getattr(source, name, 1)
            )
            if int(value or 1) > 1:
                unsupported.append(f"{name}={value}")
    if unsupported:
        raise ValueError(
            "The legacy training runtime supports only data parallelism and sharding; "
            f"unsupported request: {', '.join(unsupported)}. "
            "For a model-specific experimental path only, use the explicit Python APIs "
            "in parascale.parallel.distributed and parascale.parallel.execution; "
            "they are not wired through the legacy training runtime. "
            "build_parallel_plan remains available for declarative introspection."
        )
