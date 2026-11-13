# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Native FP32 distillation; importing configuration does not import torch."""

from importlib import import_module

from .config import DistillationConfig

__all__ = [
    "DistillationConfig",
    "DistillationLossOutput",
    "distillation_loss",
    "OnlineDistillationLoss",
]


def __getattr__(name):
    modules = {
        "DistillationLossOutput": "loss",
        "distillation_loss": "loss",
        "OnlineDistillationLoss": "online",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value
