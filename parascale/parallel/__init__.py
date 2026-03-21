# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 并行策略模块

本模块提供了 ParaScale 框架支持的所有并行策略实现。

架构说明:
    本模块采用统一架构设计，参考Megatron-LM和DeepSpeed的最佳实践：
    
    1. 单一实现 - 每个并行策略只有一个实现类
    2. 策略模式 - 通过配置选择不同的并行行为
    3. 参考业界最佳实践 - Megatron-LM和DeepSpeed

可用的并行策略:
    - DataParallel: 数据并行
    - ModelParallel: 模型并行（按层分割）
    - TensorParallel: 张量并行（按张量分割）
    - PipelineParallel: 流水线并行
    - HybridParallel: 3D混合并行（DP+TP+PP）

使用示例:
    >>> from parascale.parallel import TensorParallel, HybridParallel
    >>> 
    >>> # 张量并行
    >>> tp = TensorParallel(model, rank=0, world_size=4, tp_size=2)
    >>> 
    >>> # 3D混合并行
    >>> hp = HybridParallel(model, rank=0, world_size=8, dp_size=2, tp_size=2, pp_size=2)
"""

# 基础类
from .base import BaseParallel, ParallelConfigError, ParallelInitError

# 数据并行和模型并行
from .data_parallel import DataParallel
from .model_parallel import ModelParallel
from .pipeline_parallel import PipelineParallel

# 张量并行
from .tensor_parallel import (
    TensorParallel,
    TensorParallelConfig,
    ParallelStrategy,
    TensorParallelConverter,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    ParallelSelfAttention,
    ParallelMLP,
)

# 3D混合并行
from .hybrid_parallel import (
    HybridParallel,
    HybridParallelConfig,
    PipelineSchedule,
    PipelineScheduler,
    PipelineCommunicator,
)

# 通信优化模块
from .communication import (
    GradientCompressor,
    TopKCompressor,
    OneBitAdamCompressor,
    CommunicationOverlap,
    RingAllReduce,
    OptimizedCommunicator,
)

__all__ = [
    # 基础类
    "BaseParallel",
    "ParallelConfigError",
    "ParallelInitError",
    
    # 并行策略
    "DataParallel",
    "ModelParallel",
    "TensorParallel",
    "PipelineParallel",
    "HybridParallel",
    
    # 配置类
    "TensorParallelConfig",
    "HybridParallelConfig",
    "ParallelStrategy",
    "PipelineSchedule",
    
    # 并行层组件
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "ParallelSelfAttention",
    "ParallelMLP",
    
    # 工具类
    "TensorParallelConverter",
    "PipelineScheduler",
    "PipelineCommunicator",
    
    # 通信优化
    "GradientCompressor",
    "TopKCompressor",
    "OneBitAdamCompressor",
    "CommunicationOverlap",
    "RingAllReduce",
    "OptimizedCommunicator",
]
