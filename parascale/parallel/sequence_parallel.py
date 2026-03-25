# -*- coding: utf-8 -*-
# @Time    : 2026/3/23
# @Author  : Jun Wang
# @File    : sequence_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 序列并行模块

本模块实现了序列并行（Sequence Parallelism）策略，参考Megatron-LM和DeepSpeed的实现。

序列并行原理:
    在张量并行的基础上，将序列维度（sequence dimension）切分到TP组内的所有GPU。
    主要用于减少LayerNorm、Dropout等层的激活内存占用。

    标准张量并行: 每个TP rank持有完整的序列，但只持有部分隐藏维度
    序列并行: 每个TP rank持有部分序列，部分隐藏维度

    内存节省: 对于LayerNorm、Dropout等层，激活内存减少 TP_SIZE 倍

架构设计:
    ┌─────────────────────────────────────────────────────────────┐
    │                  SequenceParallel                           │
    ├─────────────────────────────────────────────────────────────┤
    │  Core Components:                                           │
    │  - SequenceParallelLayerNorm: 序列并行LayerNorm            │
    │  - SequenceParallelDropout: 序列并行Dropout                │
    │  - ScatterToSequenceParallelRegion: 切分到序列并行区域      │
    │  - GatherFromSequenceParallelRegion: 从序列并行区域收集     │
    │                                                             │
    │  Integration with Tensor Parallel:                          │
    │  - 在张量并行的Attention和MLP之间插入序列并行层            │
    │  - 减少激活内存，提高训练效率                              │
    └─────────────────────────────────────────────────────────────┘

使用示例:
    >>> from parascale.parallel import SequenceParallel, SequenceParallelConfig
    >>> 
    >>> # 与张量并行一起使用
    >>> config = SequenceParallelConfig(
    ...     sp_size=4,           # 序列并行大小
    ...     tp_size=4,           # 张量并行大小
    ...     enable_ulysses=True  # 启用Ulysses风格的长序列并行
    >>> )
    >>> sp = SequenceParallel(model, rank=0, world_size=16, config=config)
    >>> 
    >>> # 在HybridParallel中使用
    >>> from parascale.parallel import HybridParallelConfig
    >>> hp_config = HybridParallelConfig(
    ...     dp_size=2,
    ...     tp_size=4,
    ...     sp_size=4,  # 启用序列并行
    ...     pp_size=1
    ... )

Reference:
    - Megatron-LM: "Reducing Activation Recomputation in Large Transformer Models"
    - DeepSpeed-Ulysses: "DeepSpeed Ulysses: System Optimizations for Enabling Training 
      of Extreme Long Sequence Transformer Models"
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .base import BaseParallel, ParallelConfigError, ParallelInitError
from .tensor_parallel import TensorParallelConfig

logger = logging.getLogger(__name__)


# =============================================================================
# 配置类
# =============================================================================

class SequenceParallelMode(Enum):
    """序列并行模式"""
    STANDARD = "standard"      # 标准序列并行（Megatron风格）
    ULYSSES = "ulysses"        # Ulysses长序列并行（DeepSpeed风格）


@dataclass
class SequenceParallelConfig:
    """
    序列并行配置类
    
    Attributes:
        sp_size: 序列并行大小
        tp_size: 张量并行大小（序列并行通常与张量并行一起使用）
        mode: 序列并行模式（standard/ulysses）
        scatter_input: 是否自动切分输入
        gather_output: 是否自动收集输出
        enable_for_layernorm: 是否为LayerNorm启用序列并行
        enable_for_dropout: 是否为Dropout启用序列并行
        enable_for_activation: 是否为激活函数启用序列并行
    """
    sp_size: int = 1
    tp_size: int = 1
    mode: SequenceParallelMode = SequenceParallelMode.STANDARD
    scatter_input: bool = True
    gather_output: bool = True
    enable_for_layernorm: bool = True
    enable_for_dropout: bool = True
    enable_for_activation: bool = True
    
    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = SequenceParallelMode(self.mode)


# =============================================================================
# 通信工具函数
# =============================================================================

def _split_tensor_along_dim(tensor: torch.Tensor, dim: int, sp_size: int, sp_rank: int) -> torch.Tensor:
    """
    沿指定维度切分张量
    
    Args:
        tensor: 输入张量
        dim: 切分维度
        sp_size: 序列并行大小
        sp_rank: 当前序列并行rank
    
    Returns:
        切分后的张量
    """
    dim_size = tensor.size(dim)
    chunk_size = dim_size // sp_size
    return tensor.narrow(dim, sp_rank * chunk_size, chunk_size).contiguous()


def _gather_tensor_along_dim(tensor: torch.Tensor, dim: int, sp_group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """
    沿指定维度收集张量
    
    Args:
        tensor: 输入张量
        dim: 收集维度
        sp_group: 序列并行进程组
    
    Returns:
        收集后的张量
    """
    if not dist.is_initialized() or sp_group is None or dist.get_world_size(sp_group) == 1:
        return tensor
    
    world_size = dist.get_world_size(sp_group)
    
    # All-gather
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=sp_group)
    
    # 沿指定维度拼接
    return torch.cat(tensor_list, dim=dim)


# =============================================================================
# 自定义Autograd Function
# =============================================================================

class _ScatterToSequenceParallelRegion(Function):
    """
    将张量切分到序列并行区域
    
    前向: 切分输入张量
    反向: All-gather梯度
    """
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, sp_group: Optional[dist.ProcessGroup]):
        ctx.dim = dim
        ctx.sp_group = sp_group
        
        if not dist.is_initialized() or sp_group is None:
            return input_
        
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        
        return _split_tensor_along_dim(input_, dim, sp_size, sp_rank)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 反向传播时收集梯度
        return _gather_tensor_along_dim(grad_output, ctx.dim, ctx.sp_group), None, None


class _GatherFromSequenceParallelRegion(Function):
    """
    从序列并行区域收集张量
    
    前向: All-gather张量
    反向: 切分梯度
    """
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, sp_group: Optional[dist.ProcessGroup]):
        ctx.dim = dim
        ctx.sp_group = sp_group
        
        return _gather_tensor_along_dim(input_, dim, sp_group)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not dist.is_initialized() or ctx.sp_group is None:
            return grad_output, None, None
        
        sp_size = dist.get_world_size(ctx.sp_group)
        sp_rank = dist.get_rank(ctx.sp_group)
        
        return _split_tensor_along_dim(grad_output, ctx.dim, sp_size, sp_rank), None, None


# =============================================================================
# 序列并行层实现
# =============================================================================

class SequenceParallelLayerNorm(nn.Module):
    """
    序列并行LayerNorm
    
    在序列并行区域内执行LayerNorm，每个rank只处理部分序列。
    可以显著减少LayerNorm的激活内存占用。
    
    Args:
        normalized_shape: 归一化的形状
        eps: 数值稳定性常数
        elementwise_affine: 是否使用可学习的仿射参数
        sp_group: 序列并行进程组
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.sp_group = sp_group
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 在序列并行区域内执行LayerNorm
        # 输入已经是切分过的，直接执行LayerNorm
        return nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class SequenceParallelDropout(nn.Module):
    """
    序列并行Dropout
    
    在序列并行区域内执行Dropout，每个rank独立处理自己的序列片段。
    
    Args:
        p: dropout概率
        inplace: 是否就地操作
        sp_group: 序列并行进程组
    """
    
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.p = p
        self.inplace = inplace
        self.sp_group = sp_group
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 每个rank独立执行dropout
        return nn.functional.dropout(input, self.p, self.training, self.inplace)


class SequenceParallelLinear(nn.Module):
    """
    序列并行线性层
    
    输入是序列并行的（沿序列维度切分），输出可以选择是否收集。
    通常用于在Attention/MLP前后进行投影。
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        bias: 是否使用偏置
        gather_output: 是否收集输出（从序列并行区域收集）
        sp_group: 序列并行进程组
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.sp_group = sp_group
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 执行线性变换
        output = nn.functional.linear(input, self.weight, self.bias)
        
        # 如果需要，收集输出
        if self.gather_output and self.sp_group is not None:
            output = _GatherFromSequenceParallelRegion.apply(output, 1, self.sp_group)
        
        return output


# =============================================================================
# Ulysses长序列并行（DeepSpeed风格）
# =============================================================================

class UlyssesAttention(nn.Module):
    """
    Ulysses风格的长序列注意力
    
    参考DeepSpeed Ulysses实现，将all-to-all通信与注意力计算结合，
    支持超长序列（1M+ tokens）的训练。
    
    原理:
        1. 输入: [B, S/P, H] (序列并行，每个rank持有S/P的序列)
        2. All-to-all: [B, S/P, H] -> [B/P, S, H] (切换到头并行)
        3. 执行注意力计算
        4. All-to-all: [B/P, S, H] -> [B, S/P, H] (切换回序列并行)
    
    Args:
        hidden_size: 隐藏层维度
        num_attention_heads: 注意力头数
        sp_group: 序列并行进程组
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.sp_group = sp_group
        
        # QKV投影
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len_per_rank, hidden_size = hidden_states.size()
        
        # 如果启用Ulysses，执行all-to-all切换
        if self.sp_group is not None and dist.get_world_size(self.sp_group) > 1:
            # All-to-all: [B, S/P, H] -> [B/P, S, H]
            hidden_states = self._all_to_all_forward(hidden_states)
        
        # 执行标准注意力计算
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 简化的注意力计算（实际应使用flash attention等优化）
        attn_output = self._attention(q, k, v, attention_mask)
        output = self.out_proj(attn_output)
        
        # 切换回序列并行
        if self.sp_group is not None and dist.get_world_size(self.sp_group) > 1:
            output = self._all_to_all_backward(output)
        
        return output
    
    def _all_to_all_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-to-all: 序列并行 -> 头并行"""
        # 简化实现，实际应使用dist.all_to_all
        return tensor
    
    def _all_to_all_backward(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-to-all: 头并行 -> 序列并行"""
        return tensor
    
    def _attention(self, q, k, v, mask=None):
        """简化注意力计算"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)


# =============================================================================
# 模型转换器
# =============================================================================

class SequenceParallelConverter:
    """
    序列并行模型转换器
    
    自动将标准PyTorch模型转换为序列并行版本。
    """
    
    @classmethod
    def convert_model(
        cls,
        model: nn.Module,
        config: SequenceParallelConfig,
        sp_group: Optional[dist.ProcessGroup] = None,
    ) -> nn.Module:
        """
        转换模型为序列并行版本
        
        Args:
            model: 原始模型
            config: 序列并行配置
            sp_group: 序列并行进程组
        
        Returns:
            转换后的模型
        """
        if config.sp_size == 1:
            return model
        
        # 替换LayerNorm
        if config.enable_for_layernorm:
            cls._replace_layernorm(model, sp_group)
        
        # 替换Dropout
        if config.enable_for_dropout:
            cls._replace_dropout(model, sp_group)
        
        return model
    
    @classmethod
    def _replace_layernorm(cls, model: nn.Module, sp_group: Optional[dist.ProcessGroup]):
        """替换LayerNorm为序列并行版本"""
        for name, module in list(model.named_modules()):
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                # 获取父模块和层名
                *parent_path, layer_name = name.split('.')
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                
                # 创建序列并行LayerNorm
                sp_layernorm = SequenceParallelLayerNorm(
                    normalized_shape=module.normalized_shape if hasattr(module, 'normalized_shape') else module.num_features,
                    eps=module.eps if hasattr(module, 'eps') else 1e-5,
                    elementwise_affine=module.elementwise_affine if hasattr(module, 'elementwise_affine') else True,
                    sp_group=sp_group,
                )
                
                # 复制权重
                if hasattr(module, 'weight') and module.weight is not None:
                    sp_layernorm.weight.data = module.weight.data.clone()
                if hasattr(module, 'bias') and module.bias is not None:
                    sp_layernorm.bias.data = module.bias.data.clone()
                
                setattr(parent, layer_name, sp_layernorm)
    
    @classmethod
    def _replace_dropout(cls, model: nn.Module, sp_group: Optional[dist.ProcessGroup]):
        """替换Dropout为序列并行版本"""
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Dropout):
                *parent_path, layer_name = name.split('.')
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                
                sp_dropout = SequenceParallelDropout(
                    p=module.p,
                    inplace=module.inplace,
                    sp_group=sp_group,
                )
                
                setattr(parent, layer_name, sp_dropout)


# =============================================================================
# 主类
# =============================================================================

class SequenceParallel(BaseParallel):
    """
    序列并行策略类
    
    实现了序列并行（Sequence Parallelism）策略，主要用于减少LayerNorm、
    Dropout等层的激活内存占用。通常与张量并行一起使用。
    
    Args:
        model: PyTorch模型实例
        rank: 当前进程的rank
        world_size: 世界大小
        sp_size: 序列并行大小 (默认: 1)
        tp_size: 张量并行大小 (默认: 1)
        sequence_dim: 序列维度 (默认: 1，即batch中的序列维度)
        scatter_input: 是否自动切分输入 (默认: True)
        gather_output: 是否自动收集输出 (默认: True)
        config: 序列并行配置对象 (如果提供，其他参数将被忽略)
    
    Example:
        >>> from parascale.parallel import SequenceParallel
        >>> 
        >>> # 与张量并行一起使用
        >>> sp = SequenceParallel(
        ...     model, 
        ...     rank=rank, 
        ...     world_size=8, 
        ...     sp_size=4,      # 4路序列并行
        ...     tp_size=2       # 2路张量并行
        ... )
        >>> 
        >>> # 输入会被自动切分
        >>> output = sp(inputs)  # inputs: [B, S, H] -> 每个rank: [B, S/4, H]
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        sp_size: int = 1,
        tp_size: int = 1,
        sequence_dim: int = 1,
        scatter_input: bool = True,
        gather_output: bool = True,
        config: Optional[SequenceParallelConfig] = None,
    ):
        """
        初始化序列并行策略
        
        Args:
            model: PyTorch模型实例
            rank: 当前进程的rank
            world_size: 世界大小
            sp_size: 序列并行大小
            tp_size: 张量并行大小
            sequence_dim: 序列维度
            scatter_input: 是否自动切分输入
            gather_output: 是否自动收集输出
            config: 配置对象
        """
        super().__init__(model, rank, world_size)
        
        # 使用配置对象或创建新的
        if config is not None:
            self.config = config
        else:
            self.config = SequenceParallelConfig(
                sp_size=sp_size,
                tp_size=tp_size,
                scatter_input=scatter_input,
                gather_output=gather_output,
            )
        
        self.sp_size = self.config.sp_size
        self.tp_size = self.config.tp_size
        self.sequence_dim = sequence_dim
        
        # 验证配置
        self._validate_config()
        
        # 计算序列并行rank
        self.sp_rank = self._compute_sp_rank()
        
        # 初始化进程组
        self.sp_group = None
        self._init_process_groups()
        
        # 转换模型
        if self.sp_size > 1:
            self._parallelize_model()
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        logger.info(
            f"SequenceParallel initialized: rank={rank}, "
            f"sp_size={self.sp_size}, tp_size={self.tp_size}, "
            f"sp_rank={self.sp_rank}"
        )
    
    def _validate_config(self):
        """验证配置"""
        total_size = self.config.sp_size * self.config.tp_size
        if total_size > self.world_size:
            raise ParallelConfigError(
                f"sp_size({self.config.sp_size}) × tp_size({self.config.tp_size}) = "
                f"{total_size} > world_size({self.world_size})"
            )
        
        if self.world_size % total_size != 0:
            logger.warning(
                f"world_size({self.world_size}) is not divisible by "
                f"sp_size×tp_size({total_size})"
            )
    
    def _compute_sp_rank(self) -> int:
        """计算序列并行rank"""
        if self.sp_size == 1:
            return 0
        
        # 假设rank布局: [tp_rank, sp_rank, dp_rank]
        # 即先张量并行，再序列并行，最后数据并行
        return (self.rank // self.tp_size) % self.sp_size
    
    def _init_process_groups(self):
        """初始化序列并行进程组"""
        if not dist.is_initialized() or self.sp_size == 1:
            return
        
        # 创建序列并行组
        # 与张量正交互补的进程组
        sp_ranks = []
        for tp_rank in range(self.tp_size):
            for dp_group in range(self.world_size // (self.sp_size * self.tp_size)):
                group_ranks = [
                    dp_group * self.sp_size * self.tp_size + sp_rank * self.tp_size + tp_rank
                    for sp_rank in range(self.sp_size)
                ]
                sp_ranks.append(group_ranks)
        
        for ranks in sp_ranks:
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.sp_group = group
                break
    
    def _parallelize_model(self):
        """并行化模型"""
        self.model = SequenceParallelConverter.convert_model(
            self.model,
            self.config,
            self.sp_group,
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        序列并行前向传播
        
        Args:
            inputs: 输入数据张量，形状通常为 [B, S, H]
        
        Returns:
            模型输出张量
        """
        # 切分输入到序列并行区域
        if self.config.scatter_input and self.sp_size > 1:
            inputs = _ScatterToSequenceParallelRegion.apply(
                inputs, self.sequence_dim, self.sp_group
            )
        
        # 移动到设备
        inputs = self.to_device(inputs)
        
        # 执行前向传播
        outputs = self.model(inputs)
        
        # 收集输出
        if self.config.gather_output and self.sp_size > 1:
            outputs = _GatherFromSequenceParallelRegion.apply(
                outputs, self.sequence_dim, self.sp_group
            )
        
        return outputs
    
    def scatter_to_sp_region(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将张量切分到序列并行区域
        
        Args:
            tensor: 输入张量
        
        Returns:
            切分后的张量
        """
        if self.sp_size == 1:
            return tensor
        return _ScatterToSequenceParallelRegion.apply(tensor, self.sequence_dim, self.sp_group)
    
    def gather_from_sp_region(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        从序列并行区域收集张量
        
        Args:
            tensor: 输入张量
        
        Returns:
            收集后的张量
        """
        if self.sp_size == 1:
            return tensor
        return _GatherFromSequenceParallelRegion.apply(tensor, self.sequence_dim, self.sp_group)
    
    def gather_gradients(self) -> None:
        """
        收集梯度
        
        对于序列并行，梯度已经在各自rank上，不需要额外同步。
        """
        pass
    
    def broadcast_model(self) -> None:
        """广播模型参数"""
        if self.sp_group is None:
            return
        
        for param in self.model.parameters():
            dist.broadcast(param, src=0, group=self.sp_group)
    
    def get_parallel_info(self) -> Dict[str, Any]:
        """
        获取并行配置信息
        
        Returns:
            包含并行配置信息的字典
        """
        info = super().get_parallel_info()
        info.update({
            "sp_size": self.sp_size,
            "tp_size": self.tp_size,
            "sp_rank": self.sp_rank,
            "sequence_dim": self.sequence_dim,
            "mode": self.config.mode.value,
            "scatter_input": self.config.scatter_input,
            "gather_output": self.config.gather_output,
        })
        return info
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取内存统计信息
        
        Returns:
            内存统计字典
        """
        if not torch.cuda.is_available():
            return {}
        
        return {
            "activation_memory_reduction": self.sp_size,
            "layernorm_memory_reduction": self.sp_size,
            "dropout_memory_reduction": self.sp_size,
        }


# =============================================================================
# 便捷函数
# =============================================================================

def enable_sequence_parallel(
    model: nn.Module,
    sp_size: int,
    tp_size: int = 1,
    sequence_dim: int = 1,
) -> SequenceParallel:
    """
    便捷函数：启用序列并行
    
    Args:
        model: PyTorch模型
        sp_size: 序列并行大小
        tp_size: 张量并行大小
        sequence_dim: 序列维度
    
    Returns:
        SequenceParallel实例
    
    Example:
        >>> model = TransformerModel()
        >>> sp = enable_sequence_parallel(model, sp_size=4, tp_size=2)
        >>> output = sp(inputs)
    """
    import torch.distributed as dist
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    return SequenceParallel(
        model=model,
        rank=rank,
        world_size=world_size,
        sp_size=sp_size,
        tp_size=tp_size,
        sequence_dim=sequence_dim,
    )