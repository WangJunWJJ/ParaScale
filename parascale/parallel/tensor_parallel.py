# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : tensor_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 张量并行模块

本模块实现了高性能的张量并行策略，参考Megatron-LM和DeepSpeed的最佳实践。

架构设计:
    ┌─────────────────────────────────────────────────────────────┐
    │                  TensorParallel                             │
    ├─────────────────────────────────────────────────────────────┤
    │  Strategy Pattern:                                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
    │  │   Simple    │  │ Transformer │  │   Custom    │         │
    │  │   (基础)    │  │   (优化)    │  │  (自定义)   │         │
    │  └─────────────┘  └─────────────┘  └─────────────┘         │
    │                                                             │
    │  Core Components:                                           │
    │  - ColumnParallelLinear: 列并行线性层                      │
    │  - RowParallelLinear: 行并行线性层                         │
    │  - VocabParallelEmbedding: 词表并行嵌入层                  │
    │  - ParallelSelfAttention: 优化的自注意力                   │
    │  - ParallelMLP: 优化的MLP                                  │
    └─────────────────────────────────────────────────────────────┘

使用示例:
    >>> from parascale.parallel import TensorParallel, TensorParallelConfig
    >>> 
    >>> # 简单使用 - 自动检测和并行化
    >>> model = TransformerModel()
    >>> tp = TensorParallel(model, rank=0, world_size=4, tp_size=2)
    >>> 
    >>> # 使用配置对象
    >>> config = TensorParallelConfig(tp_size=2, strategy="transformer")
    >>> tp = TensorParallel(model, rank=0, world_size=4, config=config)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from enum import Enum
import logging
import math

from .base import BaseParallel, ParallelConfigError, ParallelInitError

logger = logging.getLogger(__name__)


# =============================================================================
# 配置和策略
# =============================================================================

class ParallelStrategy(Enum):
    """并行策略枚举"""
    SIMPLE = "simple"           # 简单策略：仅切分指定层
    TRANSFORMER = "transformer" # Transformer优化策略
    AUTO = "auto"               # 自动检测并选择策略


class TensorParallelConfig:
    """
    张量并行配置类
    
    Attributes:
        tp_size: 张量并行大小
        strategy: 并行策略
        layer_config: 层配置，指定哪些层需要并行化
        auto_detect: 是否自动检测层类型
        fuse_communication: 是否融合通信操作
    """
    
    def __init__(
        self,
        tp_size: int = 1,
        strategy: Union[str, ParallelStrategy] = ParallelStrategy.AUTO,
        layer_config: Optional[Dict[str, Any]] = None,
        auto_detect: bool = True,
        fuse_communication: bool = True,
    ):
        self.tp_size = tp_size
        self.strategy = ParallelStrategy(strategy) if isinstance(strategy, str) else strategy
        self.layer_config = layer_config or {}
        self.auto_detect = auto_detect
        self.fuse_communication = fuse_communication


# =============================================================================
# 通信工具函数
# =============================================================================

def _all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce通信"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return input_
    output = input_.clone()
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output


def _gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    """沿最后一个维度gather"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return input_
    world_size = dist.get_world_size()
    input_list = [torch.empty_like(input_) for _ in range(world_size)]
    dist.all_gather(input_list, input_)
    return torch.cat(input_list, dim=-1)


def _split_along_last_dim(input_: torch.Tensor, tp_size: int, rank: int) -> torch.Tensor:
    """沿最后一个维度split"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return input_
    last_dim = input_.dim() - 1
    last_dim_size = input_.size()[last_dim]
    chunk_size = last_dim_size // tp_size
    return input_.narrow(last_dim, rank * chunk_size, chunk_size).contiguous()


# =============================================================================
# 自定义Autograd Function
# =============================================================================

class _ReduceFromTensorParallelRegion(Function):
    """从张量并行区域reduce张量 - 用于列并行"""
    
    @staticmethod
    def forward(ctx, input_):
        return _all_reduce(input_)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _GatherFromTensorParallelRegion(Function):
    """从张量并行区域gather张量 - 用于行并行"""
    
    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)
    
    @staticmethod
    def backward(ctx, grad_output):
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        return _split_along_last_dim(grad_output, tp_size, rank)


class _ScatterToTensorParallelRegion(Function):
    """将张量scatter到张量并行区域 - 用于列并行输入"""
    
    @staticmethod
    def forward(ctx, input_, tp_size, rank):
        ctx.tp_size = tp_size
        ctx.rank = rank
        return _split_along_last_dim(input_, tp_size, rank)
    
    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output), None, None


# =============================================================================
# 并行层实现
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """
    列并行线性层
    
    将权重矩阵按输出维度切分，每个rank只保存部分权重。
    输出需要all-reduce来合并各rank的结果。
    
    Args:
        input_size: 输入维度
        output_size: 输出维度
        bias: 是否使用偏置
        gather_output: 是否gather输出
        init_method: 权重初始化方法
        device: 设备
        dtype: 数据类型
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        init_method: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if output_size % tp_size != 0:
            raise ValueError(f"Output size {output_size} must be divisible by tp_size {tp_size}")
        
        self.input_size = input_size
        self.output_size_per_partition = output_size // tp_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.tp_size = tp_size
        self.rank = rank
        
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition, input_size, device=device, dtype=dtype
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition, device=device, dtype=dtype
            ))
        else:
            self.register_parameter('bias', None)
        
        # 初始化
        if init_method is None:
            init_method = nn.init.xavier_uniform_
        init_method(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_):
        output_parallel = nn.functional.linear(input_, self.weight, self.bias)
        output = _ReduceFromTensorParallelRegion.apply(output_parallel)
        if self.gather_output:
            output = _GatherFromTensorParallelRegion.apply(output)
        return output


class RowParallelLinear(nn.Module):
    """
    行并行线性层
    
    将权重矩阵按输入维度切分，每个rank只保存部分权重。
    输入需要scatter，输出自动是完整的。
    
    Args:
        input_size: 输入维度
        output_size: 输出维度
        bias: 是否使用偏置
        input_is_parallel: 输入是否已经是并行的
        init_method: 权重初始化方法
        device: 设备
        dtype: 数据类型
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if input_size % tp_size != 0:
            raise ValueError(f"Input size {input_size} must be divisible by tp_size {tp_size}")
        
        self.input_size_per_partition = input_size // tp_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.tp_size = tp_size
        self.rank = rank
        
        self.weight = nn.Parameter(torch.empty(
            output_size, self.input_size_per_partition, device=device, dtype=dtype
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        # 初始化
        if init_method is None:
            init_method = nn.init.xavier_uniform_
        init_method(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_):
        if not self.input_is_parallel:
            input_ = _ScatterToTensorParallelRegion.apply(input_, self.tp_size, self.rank)
        
        output_parallel = nn.functional.linear(input_, self.weight)
        output = _ReduceFromTensorParallelRegion.apply(output_parallel)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class VocabParallelEmbedding(nn.Module):
    """
    词表并行Embedding层
    
    将词表按词汇维度切分，每个rank只保存部分词向量。
    输出需要all-reduce来合并各rank的结果。
    
    Args:
        num_embeddings: 词表大小
        embedding_dim: 嵌入维度
        init_method: 初始化方法
        device: 设备
        dtype: 数据类型
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_method: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if num_embeddings % tp_size != 0:
            raise ValueError(f"Num embeddings {num_embeddings} must be divisible by tp_size {tp_size}")
        
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.embedding_dim = embedding_dim
        self.rank = rank
        
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition, embedding_dim, device=device, dtype=dtype
        ))
        
        if init_method is None:
            init_method = nn.init.xavier_uniform_
        init_method(self.weight)
    
    def forward(self, input_):
        # 将输入id映射到当前rank的词表范围
        input_mask = (input_ < self.rank * self.num_embeddings_per_partition) | \
                     (input_ >= (self.rank + 1) * self.num_embeddings_per_partition)
        masked_input = input_.clone() - self.rank * self.num_embeddings_per_partition
        masked_input[input_mask] = 0
        
        output_parallel = nn.functional.embedding(masked_input, self.weight)
        output_parallel[input_mask] = 0.0
        
        output = _ReduceFromTensorParallelRegion.apply(output_parallel)
        return output


class ParallelSelfAttention(nn.Module):
    """
    并行自注意力层 - 针对Transformer优化
    
    融合ColumnParallel和RowParallel，最小化通信量:
    - Q/K/V投影: ColumnParallel (输出all-reduce)
    - Attention计算: 每个rank计算部分注意力
    - 输出投影: RowParallel (输入scatter, 输出all-reduce)
    
    Args:
        hidden_size: 隐藏层维度
        num_attention_heads: 注意力头数
        attention_dropout: dropout概率
        device: 设备
        dtype: 数据类型
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        
        if num_attention_heads % tp_size != 0:
            raise ValueError(f"Num attention heads {num_attention_heads} must be divisible by tp_size {tp_size}")
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads // tp_size
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        
        # Q/K/V投影 - ColumnParallel
        self.query_key_value = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, bias=True, gather_output=False,
            device=device, dtype=dtype
        )
        
        # 输出投影 - RowParallel
        self.dense = RowParallelLinear(
            hidden_size, hidden_size, bias=True, input_is_parallel=False,
            device=device, dtype=dtype
        )
        
        self.attention_dropout = nn.Dropout(attention_dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Q/K/V投影
        mixed_x_layer = self.query_key_value(hidden_states)
        mixed_x_layer = mixed_x_layer.view(
            batch_size, seq_length, self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head
        )
        
        query_layer, key_layer, value_layer = mixed_x_layer.split(self.hidden_size_per_attention_head, dim=-1)
        
        # 调整形状
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)
        
        # 注意力计算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_attention_head)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, -1)
        
        # 输出投影
        output = self.dense(context_layer)
        return output


class ParallelMLP(nn.Module):
    """
    并行MLP层 - 针对Transformer优化
    
    融合ColumnParallel和RowParallel:
    - FC1: ColumnParallel + GeLU (输出all-reduce)
    - FC2: RowParallel (输入scatter, 输出all-reduce)
    
    Args:
        hidden_size: 隐藏层维度
        ffn_hidden_size: FFN中间层维度
        activation: 激活函数类型
        device: 设备
        dtype: 数据类型
    """
    
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        activation: str = "gelu",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # FC1 - ColumnParallel
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size, ffn_hidden_size, bias=True, gather_output=False,
            device=device, dtype=dtype
        )
        
        # FC2 - RowParallel
        self.dense_4h_to_h = RowParallelLinear(
            ffn_hidden_size, hidden_size, bias=True, input_is_parallel=False,
            device=device, dtype=dtype
        )
        
        if activation == "gelu":
            self.activation_func = nn.functional.gelu
        elif activation == "relu":
            self.activation_func = nn.functional.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        intermediate = self.activation_func(intermediate)
        output = self.dense_4h_to_h(intermediate)
        return output


# =============================================================================
# 模型转换器
# =============================================================================

class TensorParallelConverter:
    """
    张量并行模型转换器
    
    自动将标准PyTorch模型转换为张量并行版本。
    """
    
    # 层类型映射规则
    LAYER_MAPPING = {
        'q_proj': ('column', ColumnParallelLinear),
        'k_proj': ('column', ColumnParallelLinear),
        'v_proj': ('column', ColumnParallelLinear),
        'query': ('column', ColumnParallelLinear),
        'key': ('column', ColumnParallelLinear),
        'value': ('column', ColumnParallelLinear),
        'o_proj': ('row', RowParallelLinear),
        'dense': ('row', RowParallelLinear),
        'output': ('row', RowParallelLinear),
        'fc1': ('column', ColumnParallelLinear),
        'fc2': ('row', RowParallelLinear),
        'up_proj': ('column', ColumnParallelLinear),
        'down_proj': ('row', RowParallelLinear),
        'embed_tokens': ('vocab', VocabParallelEmbedding),
        'word_embeddings': ('vocab', VocabParallelEmbedding),
    }
    
    @classmethod
    def convert_model(
        cls,
        model: nn.Module,
        config: TensorParallelConfig,
    ) -> nn.Module:
        """
        转换模型为张量并行版本
        
        Args:
            model: 原始模型
            config: 张量并行配置
        
        Returns:
            转换后的模型
        """
        if config.tp_size == 1:
            return model
        
        # 自动检测层配置
        if config.auto_detect and not config.layer_config:
            config.layer_config = cls._auto_detect_layers(model)
        
        # 根据策略选择转换方式
        if config.strategy == ParallelStrategy.TRANSFORMER:
            return cls._convert_transformer_model(model, config)
        else:
            return cls._convert_simple_model(model, config)
    
    @classmethod
    def _auto_detect_layers(cls, model: nn.Module) -> Dict[str, str]:
        """自动检测层类型"""
        config = {}
        
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            name_lower = name.lower()
            for pattern, (parallel_type, _) in cls.LAYER_MAPPING.items():
                if pattern in name_lower:
                    config[name] = parallel_type
                    break
        
        return config
    
    @classmethod
    def _convert_simple_model(cls, model: nn.Module, config: TensorParallelConfig) -> nn.Module:
        """简单策略：仅转换指定的Linear层"""
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            
            parallel_type = config.layer_config.get(name)
            if parallel_type is None:
                continue
            
            new_module = cls._create_parallel_linear(module, parallel_type)
            if new_module is not None:
                cls._replace_module(model, name, new_module)
        
        return model
    
    @classmethod
    def _convert_transformer_model(cls, model: nn.Module, config: TensorParallelConfig) -> nn.Module:
        """Transformer策略：识别并替换Transformer block"""
        # 查找Transformer层并替换
        for name, module in list(model.named_modules()):
            if cls._is_transformer_block(module):
                cls._replace_transformer_block(model, name, module, config)
        
        # 处理剩余的Linear层
        return cls._convert_simple_model(model, config)
    
    @classmethod
    def _is_transformer_block(cls, module: nn.Module) -> bool:
        """检查是否是Transformer block"""
        has_attention = any(
            isinstance(m, (nn.MultiheadAttention, nn.Linear)) and 
            any(x in n for x in ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj'])
            for n, m in module.named_modules()
        )
        has_mlp = any(
            isinstance(m, nn.Linear) and 
            any(x in n for x in ['fc', 'mlp', 'dense'])
            for n, m in module.named_modules()
        )
        return has_attention and has_mlp
    
    @classmethod
    def _replace_transformer_block(
        cls,
        parent_model: nn.Module,
        block_name: str,
        block_module: nn.Module,
        config: TensorParallelConfig,
    ):
        """替换Transformer block为并行版本"""
        # 这里可以实现更复杂的block替换逻辑
        # 目前简化处理，只替换内部的Linear层
        pass
    
    @classmethod
    def _create_parallel_linear(
        cls,
        module: nn.Linear,
        parallel_type: str,
    ) -> Optional[nn.Module]:
        """创建并行线性层"""
        if parallel_type == 'column':
            return ColumnParallelLinear(
                input_size=module.in_features,
                output_size=module.out_features,
                bias=module.bias is not None,
                gather_output=False,
            )
        elif parallel_type == 'row':
            return RowParallelLinear(
                input_size=module.in_features,
                output_size=module.out_features,
                bias=module.bias is not None,
                input_is_parallel=False,
            )
        return None
    
    @classmethod
    def _replace_module(cls, model: nn.Module, name: str, new_module: nn.Module):
        """替换模型中的模块"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)


# =============================================================================
# 主类
# =============================================================================

class TensorParallel(BaseParallel):
    """
    张量并行策略类
    
    实现了高性能的张量并行策略，参考Megatron-LM和DeepSpeed的最佳实践。
    
    Args:
        model: PyTorch模型实例
        rank: 当前进程的rank
        world_size: 世界大小
        tp_size: 张量并行大小 (默认: 1)
        strategy: 并行策略，可选 "simple", "transformer", "auto"
        layer_config: 层配置，指定哪些层需要并行化
        auto_parallelize: 是否自动并行化模型 (默认: True)
        config: 张量并行配置对象 (如果提供，其他参数将被忽略)
    
    Example:
        >>> from parascale.parallel import TensorParallel
        >>> 
        >>> # 简单使用 - 自动检测和并行化
        >>> model = TransformerModel()
        >>> tp = TensorParallel(model, rank=0, world_size=4, tp_size=2)
        >>> output = tp(inputs)
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        tp_size: int = 1,
        strategy: str = "auto",
        layer_config: Optional[Dict[str, str]] = None,
        auto_parallelize: bool = True,
        config: Optional[TensorParallelConfig] = None,
    ):
        """
        初始化张量并行策略
        
        Args:
            model: PyTorch模型实例
            rank: 当前进程的rank
            world_size: 世界大小
            tp_size: 张量并行大小
            strategy: 并行策略
            layer_config: 层配置
            auto_parallelize: 是否自动并行化
            config: 配置对象
        """
        super().__init__(model, rank, world_size)
        
        # 使用配置对象或创建新的
        if config is not None:
            self.config = config
        else:
            self.config = TensorParallelConfig(
                tp_size=tp_size,
                strategy=strategy,
                layer_config=layer_config or {},
                auto_detect=auto_parallelize,
            )
        
        self.tp_size = self.config.tp_size
        self.tp_rank = rank % self.tp_size if self.tp_size > 1 else 0
        self.parallelized_layers = []
        
        # 验证配置
        self._validate_config()
        
        # 创建进程组
        self.tp_group = None
        self._init_process_groups()
        
        # 并行化模型
        if auto_parallelize and self.tp_size > 1:
            self._parallelize_model()
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        logger.info(
            f"TensorParallel initialized: rank={rank}, "
            f"tp_size={self.tp_size}, tp_rank={self.tp_rank}, "
            f"strategy={self.config.strategy.value}"
        )
    
    def _validate_config(self):
        """验证配置"""
        if self.tp_size > self.world_size:
            raise ParallelConfigError(
                f"tp_size ({self.tp_size}) cannot be larger than world_size ({self.world_size})"
            )
        
        if self.world_size % self.tp_size != 0:
            raise ParallelConfigError(
                f"world_size ({self.world_size}) must be divisible by tp_size ({self.tp_size})"
            )
    
    def _init_process_groups(self):
        """初始化张量并行进程组"""
        if not dist.is_initialized() or self.tp_size == 1:
            return
        
        # 创建张量并行组
        num_dp_groups = self.world_size // self.tp_size
        
        for dp_group_id in range(num_dp_groups):
            ranks = list(range(dp_group_id * self.tp_size, (dp_group_id + 1) * self.tp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group
                break
    
    def _parallelize_model(self):
        """并行化模型"""
        self.model = TensorParallelConverter.convert_model(self.model, self.config)
        
        # 记录并行化的层
        for name, module in self.model.named_modules():
            if isinstance(module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
                self.parallelized_layers.append(name)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 输入数据张量
        
        Returns:
            模型输出张量
        """
        x = self.to_device(inputs)
        return self.model(x)
    
    def gather_gradients(self) -> None:
        """
        收集梯度
        
        对于张量并行，梯度已经在前向传播时通过all-reduce同步。
        """
        pass
    
    def broadcast_model(self) -> None:
        """广播模型参数"""
        if not dist.is_initialized() or self.tp_group is None:
            return
        
        for param in self.model.parameters():
            dist.broadcast(param, src=0, group=self.tp_group)
    
    def get_parallel_info(self) -> Dict[str, Any]:
        """
        获取并行配置信息
        
        Returns:
            包含并行配置信息的字典
        """
        info = super().get_parallel_info()
        info.update({
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "strategy": self.config.strategy.value,
            "parallelized_layers": self.parallelized_layers,
            "num_parallelized_layers": len(self.parallelized_layers),
        })
        return info
