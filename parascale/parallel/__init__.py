# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 并行策略模块

本模块提供了 ParaScale 框架支持的所有并行策略实现，
包括数据并行、模型并行、张量并行、流水线并行和3D混合并行。
"""

from .base import BaseParallel
from .data_parallel import DataParallel
from .model_parallel import ModelParallel
from .tensor_parallel import TensorParallel
from .pipeline_parallel import PipelineParallel
from .hybrid_parallel import HybridParallel

__all__ = [
    "BaseParallel",
    "DataParallel",
    "ModelParallel",
    "TensorParallel",
    "PipelineParallel",
    "HybridParallel"
]
