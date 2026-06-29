# -*- coding: utf-8 -*-
# @Time : 2026/6/26 上午11:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Optimizer helpers shared by workload adapters."""

from __future__ import annotations

from typing import Any


def build_adamw_optimizer_for_model(optim: Any, model: Any, *, lr: float):
    """Build AdamW over trainable parameters and attach parameter telemetry."""
    params, stats = trainable_parameter_stats(model)
    optimizer = optim.AdamW(params, lr=float(lr))
    setattr(optimizer, "_parascale_parameter_stats", stats)
    return optimizer


def trainable_parameter_stats(model: Any):
    """Return trainable parameters and parameter-count telemetry."""
    parameters = list(model.parameters())
    total_params = 0
    trainable_params = 0
    selected = []
    for parameter in parameters:
        count = _parameter_numel(parameter)
        total_params += count
        if bool(getattr(parameter, "requires_grad", True)):
            selected.append(parameter)
            trainable_params += count
    if not selected:
        raise RuntimeError(
            "no trainable parameters found; optimizer construction requires at "
            "least one parameter with requires_grad=True"
        )
    ratio = float(trainable_params / total_params) if total_params else 0.0
    return selected, {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": ratio,
    }


def _parameter_numel(parameter: Any) -> int:
    numel = getattr(parameter, "numel", None)
    if callable(numel):
        return int(numel())
    return 0
