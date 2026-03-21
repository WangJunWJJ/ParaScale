# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 核心模块

本模块包含 ParaScale 框架的所有核心功能，
包括并行策略、量化训练、优化器和工具函数。
"""

from .config import ParaScaleConfig, QuantizationConfig
from .engine import Engine, ParaEngine
from .optimizers import AdamW, ZeroOptimizer
from .parallel import (
    BaseParallel,
    DataParallel,
    HybridParallel,
    ModelParallel,
    PipelineParallel,
    TensorParallel,
)
from .quantization import (
    FakeQuantize,
    MinMaxObserver,
    MovingAverageObserver,
    PostTrainingQuantization,
    QuantizationAwareTraining,
    convert_qat_model,
    prepare_qat_model,
    ptq_quantize,
)
from .utils import (
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

__version__ = "0.2.0"
__all__ = [
    # 核心引擎
    "Engine",
    "ParaEngine",
    "ParaScaleConfig",
    # 配置
    "QuantizationConfig",
    # 并行策略
    "BaseParallel",
    "DataParallel",
    "ModelParallel",
    "TensorParallel",
    "PipelineParallel",
    "HybridParallel",
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
    "PostTrainingQuantization",
    "ptq_quantize",
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
