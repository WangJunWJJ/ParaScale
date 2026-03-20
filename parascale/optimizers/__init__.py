# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 优化器模块

本模块提供了 ParaScale 框架的优化器实现，
包括 ZeRO 优化器包装器和 AdamW 优化器。

该模块作为开源及自研优化器的统一扩展模块，
支持未来添加更多优化器实现。

Available Optimizers:
    - ZeroOptimizer: ZeRO (Zero Redundancy Optimizer) 优化器
    - AdamW: Adam with Decoupled Weight Decay 优化器
"""

from .optimizers import ZeroOptimizer, AdamW, FourBitAdamW, FourBitSGD, QuantizedState

__all__ = [
    "ZeroOptimizer",
    "AdamW",
    "FourBitAdamW",
    "FourBitSGD",
    "QuantizedState",
]
