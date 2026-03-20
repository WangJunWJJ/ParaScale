# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale - A PyTorch Distributed Training Framework

ParaScale 是一个基于 PyTorch 的深度学习训练框架，支持多种并行策略
（数据并行、模型并行、张量并行、流水线并行）、量化感知训练和多节点分布式训练。
"""

from .parascale.config import ParaScaleConfig, QuantizationConfig
from .parascale.engine import Engine
from .parascale.optimizers import AdamW, ZeroOptimizer
from .parascale.parallel import (
    BaseParallel,
    DataParallel,
    ModelParallel,
    PipelineParallel,
    TensorParallel,
)
from .parascale.quantization import (
    FakeQuantize,
    MinMaxObserver,
    MovingAverageObserver,
    QuantizationAwareTraining,
    convert_qat_model,
    prepare_qat_model,
)
from .parascale.utils import (
    cleanup_distributed,
    get_distributed_info,
    get_local_rank,
    get_node_rank,
    get_num_nodes,
    get_rank,
    get_world_size,
    initialize_distributed,
    is_main_process,
    print_distributed_info,
    print_rank_0,
    setup_logging,
)

__version__ = "0.1.0"
__all__ = [
    # 核心组件
    "Engine",
    "ParaScaleConfig",
    # 配置
    "QuantizationConfig",
    # 并行策略
    "BaseParallel",
    "DataParallel",
    "ModelParallel",
    "TensorParallel",
    "PipelineParallel",
    # 优化器
    "ZeroOptimizer",
    "AdamW",
    # 量化
    "QuantizationAwareTraining",
    "prepare_qat_model",
    "convert_qat_model",
    "FakeQuantize",
    "MinMaxObserver",
    "MovingAverageObserver",
    # 分布式
    "initialize_distributed",
    "cleanup_distributed",
    "get_distributed_info",
    "print_distributed_info",
    # 工具函数
    "print_rank_0",
    "setup_logging",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "get_node_rank",
    "get_num_nodes",
    "is_main_process",
]
