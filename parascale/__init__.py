# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 核心模块

本模块包含 ParaScale 框架的所有核心功能，
包括并行策略、量化训练、优化器、工具函数和 Ascend 平台适配。

支持的硬件平台：
- NVIDIA GPU (CUDA)
- 华为昇腾 NPU (Ascend/CANN)
- CPU (Gloo)
"""

from .config import ParaScaleConfig, QuantizationConfig
from .engine import Engine, ParaEngine
from .optimizers import AdamW, ZeroOptimizer, FourBitAdamW, FourBitSGD
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

# Ascend 平台支持
try:
    from .ascend import (
        is_ascend_available,
        get_device_type,
        get_device_count,
        get_device_memory,
        initialize_ascend_environment,
        create_ascend_optimizer,
        convert_model_to_ascend,
        ASCEND_AVAILABLE,
        HCCL_AVAILABLE,
        # CANN 算子
        CANNLinear,
        CANNLayerNorm,
        CANNAttention,
        CANNMLP,
        CANNTransformerLayer,
        # HCCL 通信
        HCCLCommunicator,
        HCCLGradientCompressor,
        HCCLCommunicationOverlap,
        # 内存管理
        AscendMemoryManager,
        AscendOffloadManager,
        OffloadStrategy,
        # 并行策略
        AdaptiveParallelAnalyzer,
        AscendParallelStrategyOptimizer,
        create_optimal_strategy,
        ParallelStrategy,
    )
    ASCEND_SUPPORT = True
except ImportError:
    ASCEND_SUPPORT = False
    ASCEND_AVAILABLE = False
    HCCL_AVAILABLE = False

__version__ = "0.3.0"
__all__ = [
    # 版本信息
    "__version__",
    "ASCEND_SUPPORT",
    "ASCEND_AVAILABLE",
    "HCCL_AVAILABLE",
    
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
    "FourBitAdamW",
    "FourBitSGD",
    
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

# Ascend 平台特定导出
if ASCEND_SUPPORT:
    __all__.extend([
        # Ascend 环境检测
        "is_ascend_available",
        "get_device_type",
        "get_device_count",
        "get_device_memory",
        "initialize_ascend_environment",
        "create_ascend_optimizer",
        "convert_model_to_ascend",
        
        # CANN 算子
        "CANNLinear",
        "CANNLayerNorm",
        "CANNAttention",
        "CANNMLP",
        "CANNTransformerLayer",
        
        # HCCL 通信
        "HCCLCommunicator",
        "HCCLGradientCompressor",
        "HCCLCommunicationOverlap",
        
        # 内存管理
        "AscendMemoryManager",
        "AscendOffloadManager",
        "OffloadStrategy",
        
        # 并行策略
        "AdaptiveParallelAnalyzer",
        "AscendParallelStrategyOptimizer",
        "create_optimal_strategy",
        "ParallelStrategy",
    ])
