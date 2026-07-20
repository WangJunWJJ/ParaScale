# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Optimizer compatibility and experimental quantized optimizer exports."""

from importlib import import_module

from .spec import OptimizerSpec

__all__ = [
    "ZeroOptimizer",
    "ZeroStage",
    "ZeroPlan",
    "ExperimentalZeroOptimizer",
    "build_zero_plan",
    "create_native_zero_redundancy_optimizer",
    "wrap_zero_optimizer",
    "AdamW",
    "FourBitAdamW",
    "FourBitSGD",
    "QuantizedState",
    "OptimizerSpec",
    "build_optimizer",
]


_LEGACY_OPTIMIZERS = {
    "AdamW",
    "FourBitAdamW",
    "FourBitSGD",
    "QuantizedState",
    "ZeroOptimizer",
}
_ZERO_EXPORTS = {
    "ExperimentalZeroOptimizer",
    "ZeroPlan",
    "ZeroStage",
    "build_zero_plan",
    "create_native_zero_redundancy_optimizer",
    "wrap_zero_optimizer",
}


def __getattr__(name):
    if name == "build_optimizer":
        return getattr(import_module(".factory", __name__), name)
    if name in _LEGACY_OPTIMIZERS:
        return getattr(import_module(".optimizers", __name__), name)
    if name in _ZERO_EXPORTS:
        return getattr(import_module(".zero", __name__), name)
    raise AttributeError(name)
