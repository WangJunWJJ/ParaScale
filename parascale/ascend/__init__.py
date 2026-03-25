# -*- coding: utf-8 -*-
# @Time    : 2026/3/23
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale Ascend 平台适配模块

本模块提供了 ParaScale 框架对华为昇腾 (Ascend) 平台的完整适配，
包括 CANN 算子映射、HCCL 通信优化、内存超分策略和自适应并行策略。

模块组成：
- cann_ops: CANN 算子适配，将 PyTorch 算子映射到昇腾算子
- hccl_comm: HCCL 通信优化，实现高效的分布式通信
- memory_offload: 内存超分策略，突破物理显存限制
- adaptive_parallel: 自适应并行策略，自动选择最优并行方案

核心特性：
1. 无缝切换：与 GPU 版本 API 保持一致，一行代码切换平台
2. 性能优化：针对昇腾硬件特性深度优化
3. 内存效率：支持 4bit 量化和内存超分
4. 自动调优：根据硬件环境自动选择最优配置

使用示例：
    >>> from parascale.ascend import AscendEngine
    >>> engine = AscendEngine(model, optimizer)
    >>> engine.train(dataloader)

    >>> # 或使用自动检测
    >>> from parascale import ParaEngine
    >>> engine = ParaEngine(model, optimizer)  # 自动检测平台
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, List, Dict

from .cann_ops import (
    ASCEND_AVAILABLE,
    TORCH_NPU_AVAILABLE,
    AscendDeviceManager,
    CANNOperatorRegistry,
    cann_linear,
    cann_linear_quantized,
    cann_conv2d,
    cann_layer_norm,
    cann_softmax,
    cann_gelu,
    cann_silu,
    cann_quantize_per_tensor,
    cann_dequantize,
    cann_fused_add_layernorm,
    cann_fused_bias_gelu,
    cann_fused_softmax_dropout,
    CANNLinear,
    CANNLayerNorm,
    CANNAttention,
    CANNMLP,
    CANNTransformerLayer,
)

from .hccl_comm import (
    HCCL_AVAILABLE,
    CommunicationBackend,
    TopologyType,
    TopologyInfo,
    HCCLTopologyDetector,
    HCCLProcessGroup,
    HCCLGradientCompressor,
    HCCLCommunicator,
    HCCLCommunicationOverlap,
    HCCLBucketCommunicator,
    initialize_hccl,
)

from .memory_offload import (
    MemoryTier,
    OffloadStrategy,
    MemoryStats,
    ParameterInfo,
    AscendMemoryMonitor,
    AscendMemoryPool,
    AscendOffloadManager,
    AscendMemoryManager,
    ActivationCheckpointManager,
)

from .adaptive_parallel import (
    ParallelStrategyType,
    HardwareType,
    ModelProfile,
    ClusterInfo,
    ParallelStrategy,
    ModelProfiler,
    ClusterDetector,
    AdaptiveParallelAnalyzer,
    AscendParallelStrategyOptimizer,
    DynamicStrategyAdjuster,
    create_optimal_strategy,
)

__all__ = [
    # 环境检测
    "ASCEND_AVAILABLE",
    "TORCH_NPU_AVAILABLE",
    "HCCL_AVAILABLE",
    
    # CANN 算子
    "AscendDeviceManager",
    "CANNOperatorRegistry",
    "cann_linear",
    "cann_linear_quantized",
    "cann_conv2d",
    "cann_layer_norm",
    "cann_softmax",
    "cann_gelu",
    "cann_silu",
    "cann_quantize_per_tensor",
    "cann_dequantize",
    "cann_fused_add_layernorm",
    "cann_fused_bias_gelu",
    "cann_fused_softmax_dropout",
    
    # CANN 模块
    "CANNLinear",
    "CANNLayerNorm",
    "CANNAttention",
    "CANNMLP",
    "CANNTransformerLayer",
    
    # HCCL 通信
    "CommunicationBackend",
    "TopologyType",
    "TopologyInfo",
    "HCCLTopologyDetector",
    "HCCLProcessGroup",
    "HCCLGradientCompressor",
    "HCCLCommunicator",
    "HCCLCommunicationOverlap",
    "HCCLBucketCommunicator",
    "initialize_hccl",
    
    # 内存管理
    "MemoryTier",
    "OffloadStrategy",
    "MemoryStats",
    "ParameterInfo",
    "AscendMemoryMonitor",
    "AscendMemoryPool",
    "AscendOffloadManager",
    "AscendMemoryManager",
    "ActivationCheckpointManager",
    
    # 并行策略
    "ParallelStrategyType",
    "HardwareType",
    "ModelProfile",
    "ClusterInfo",
    "ParallelStrategy",
    "ModelProfiler",
    "ClusterDetector",
    "AdaptiveParallelAnalyzer",
    "AscendParallelStrategyOptimizer",
    "DynamicStrategyAdjuster",
    "create_optimal_strategy",
]


def is_ascend_available() -> bool:
    """
    检查 Ascend 平台是否可用
    
    Returns:
        Ascend 是否可用
    """
    return ASCEND_AVAILABLE


def get_device_type() -> str:
    """
    获取当前设备类型
    
    Returns:
        设备类型字符串 ('npu', 'cuda', 'cpu')
    """
    if ASCEND_AVAILABLE:
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_device_count() -> int:
    """
    获取设备数量
    
    Returns:
        设备数量
    """
    if ASCEND_AVAILABLE:
        return torch.npu.device_count()
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def get_device_memory(device_id: int = 0) -> Tuple[int, int]:
    """
    获取设备内存信息
    
    Args:
        device_id: 设备 ID
    
    Returns:
        (已用内存, 总内存) 元组（字节）
    """
    if ASCEND_AVAILABLE:
        try:
            total = torch.npu.get_device_properties(device_id).total_memory
            used = torch.npu.memory_allocated(device_id)
            return used, total
        except Exception:
            pass
    
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(device_id).total_memory
        used = torch.cuda.memory_allocated(device_id)
        return used, total
    
    return 0, 0


def initialize_ascend_environment(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    master_addr: str = "localhost",
    master_port: str = "29500",
    backend: str = "hccl"
) -> bool:
    """
    初始化 Ascend 分布式环境
    
    Args:
        rank: 进程 rank（可选，从环境变量读取）
        world_size: 世界大小（可选，从环境变量读取）
        master_addr: 主节点地址
        master_port: 主节点端口
        backend: 通信后端
    
    Returns:
        初始化是否成功
    
    Example:
        >>> initialize_ascend_environment(rank=0, world_size=2)
    """
    import os
    import torch.distributed as dist
    
    if dist.is_initialized():
        return True
    
    # 设置环境变量
    if rank is not None:
        os.environ["RANK"] = str(rank)
    if world_size is not None:
        os.environ["WORLD_SIZE"] = str(world_size)
    
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    # 选择后端
    if ASCEND_AVAILABLE:
        backend = "hccl"
    elif torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    
    try:
        dist.init_process_group(backend=backend)
        return True
    except Exception as e:
        print(f"初始化分布式环境失败: {e}")
        return False


def create_ascend_optimizer(
    model: nn.Module,
    optimizer_class: type = torch.optim.AdamW,
    lr: float = 1e-3,
    offload: bool = True,
    **kwargs
) -> torch.optim.Optimizer:
    """
    创建针对 Ascend 优化的优化器
    
    Args:
        model: 模型
        optimizer_class: 优化器类
        lr: 学习率
        offload: 是否启用卸载
        **kwargs: 其他优化器参数
    
    Returns:
        优化器实例
    
    Example:
        >>> optimizer = create_ascend_optimizer(model, lr=1e-4)
    """
    optimizer = optimizer_class(model.parameters(), lr=lr, **kwargs)
    
    if offload and ASCEND_AVAILABLE:
        # 将优化器状态卸载到 CPU
        for param in model.parameters():
            if param in optimizer.state:
                for key, value in optimizer.state[param].items():
                    if isinstance(value, torch.Tensor):
                        optimizer.state[param][key] = value.cpu()
    
    return optimizer


def convert_model_to_ascend(model: nn.Module) -> nn.Module:
    """
    将模型转换为 Ascend 优化版本
    
    自动将模型中的标准层替换为 CANN 优化层。
    
    Args:
        model: 原始模型
    
    Returns:
        转换后的模型
    
    Example:
        >>> model = BertModel.from_pretrained('bert-base-uncased')
        >>> model = convert_model_to_ascend(model)
    """
    if not ASCEND_AVAILABLE:
        return model
    
    for name, module in model.named_children():
        # 替换线性层
        if isinstance(module, nn.Linear) and not isinstance(module, CANNLinear):
            new_module = CANNLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        
        # 替换层归一化
        elif isinstance(module, nn.LayerNorm) and not isinstance(module, CANNLayerNorm):
            new_module = CANNLayerNorm(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine
            )
            if module.elementwise_affine:
                new_module.weight.data.copy_(module.weight.data)
                new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        
        # 递归处理子模块
        else:
            convert_model_to_ascend(module)
    
    return model
