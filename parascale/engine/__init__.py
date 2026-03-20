# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 引擎模块

本模块实现了 ParaScale 框架的核心引擎，
支持两种引擎模式：

1. Engine: 传统手动指定优化策略的引擎
   - 支持手动配置数据并行、模型并行、张量并行、流水线并行
   - 适合需要精细控制并行策略的场景

2. ParaEngine: 自动优化调度核心引擎
   - 根据模型规模和硬件状态自动选择最优并行策略
   - 适合快速部署和自动化训练场景

引擎架构:
    parascale/engine/
    ├── __init__.py          # 模块导出
    ├── engine.py            # Engine (手动策略引擎)
    └── para_engine.py       # ParaEngine (自动调度引擎)
"""

from .engine import Engine
from .para_engine import (
    HardwareMonitor,
    HardwareProfile,
    ModelAnalyzer,
    ModelProfile,
    ParallelStrategy,
    ParaEngine,
    StrategyDecider,
)

__all__ = [
    # 手动策略引擎
    "Engine",
    # 自动调度引擎
    "ParaEngine",
    "ModelAnalyzer",
    "HardwareMonitor",
    "StrategyDecider",
    "ModelProfile",
    "HardwareProfile",
    "ParallelStrategy",
]
