# -*- coding: utf-8 -*-
# @Time    : 2026/3/23
# @Author  : Jun Wang
# @File    : cann_ops.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale CANN 算子适配模块

本模块实现了 PyTorch 算子到华为昇腾 CANN (Compute Architecture for Neural Networks)
算子的映射和适配，确保关键算子在 Ascend 平台上的正确执行和性能优化。

核心功能：
- 算子映射：将 PyTorch 标准算子映射到 CANN 等效算子
- 量化算子：利用昇腾 AI Core 的 INT8/INT4 指令加速
- 融合算子：支持多算子融合以减少内存访问
- 自动降级：当 CANN 不支持时自动回退到 PyTorch 实现

适配算子列表：
- Linear (全连接层)
- Conv2d (卷积层)
- LayerNorm (层归一化)
- Softmax (软最大值)
- Gelu (高斯误差线性单元)
- 量化/反量化算子
- 分布式通信算子
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Any, Callable
import logging
import os

logger = logging.getLogger(__name__)

# 检测 Ascend 环境是否可用
ASCEND_AVAILABLE = False
TORCH_NPU_AVAILABLE = False

try:
    import torch_npu
    TORCH_NPU_AVAILABLE = True
    ASCEND_AVAILABLE = torch.npu.is_available()
    if ASCEND_AVAILABLE:
        logger.info(f"Ascend NPU 可用，设备数量: {torch.npu.device_count()}")
except ImportError:
    logger.warning("torch_npu 未安装，Ascend 功能将不可用")
except Exception as e:
    logger.warning(f"Ascend 初始化失败: {e}")


class AscendDeviceManager:
    """
    Ascend 设备管理器
    
    提供统一的 Ascend 设备管理接口，包括设备初始化、内存管理、
    流管理等功能。
    
    Attributes:
        device_id: 当前设备 ID
        device_name: 设备名称
        total_memory: 总显存大小（字节）
        available_memory: 可用显存大小（字节）
    
    Example:
        >>> manager = AscendDeviceManager(device_id=0)
        >>> manager.initialize()
        >>> print(f"设备: {manager.device_name}")
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device_id: int = 0):
        """
        初始化 Ascend 设备管理器
        
        Args:
            device_id: 设备 ID，默认为 0
        """
        if self._initialized:
            return
        
        self.device_id = device_id
        self.device_name = "Unknown"
        self.total_memory = 0
        self.available_memory = 0
        self.compute_stream = None
        self.comm_stream = None
        self._initialized = True
    
    def initialize(self) -> bool:
        """
        初始化 Ascend 设备
        
        Returns:
            初始化是否成功
        """
        if not ASCEND_AVAILABLE:
            logger.warning("Ascend 不可用，将使用 CPU 模式")
            return False
        
        try:
            torch.npu.set_device(self.device_id)
            self.device_name = torch.npu.get_device_name(self.device_id)
            self.total_memory = torch.npu.get_device_properties(self.device_id).total_memory
            self.available_memory = self.total_memory
            
            # 创建计算流和通信流
            self.compute_stream = torch.npu.Stream()
            self.comm_stream = torch.npu.Stream()
            
            logger.info(f"Ascend 设备初始化成功: {self.device_name}")
            return True
        except Exception as e:
            logger.error(f"Ascend 设备初始化失败: {e}")
            return False
    
    def synchronize(self) -> None:
        """同步所有流"""
        if ASCEND_AVAILABLE:
            torch.npu.synchronize()
    
    def get_memory_info(self) -> Tuple[int, int]:
        """
        获取内存信息
        
        Returns:
            (已用内存, 总内存) 元组
        """
        if ASCEND_AVAILABLE:
            allocated = torch.npu.memory_allocated(self.device_id)
            total = self.total_memory
            return allocated, total
        return 0, 0
    
    def empty_cache(self) -> None:
        """清空缓存"""
        if ASCEND_AVAILABLE:
            torch.npu.empty_cache()


class CANNOperatorRegistry:
    """
    CANN 算子注册表
    
    管理所有已注册的 CANN 算子映射，支持自动查找和调用。
    
    Example:
        >>> registry = CANNOperatorRegistry()
        >>> linear_op = registry.get_operator('linear')
        >>> output = linear_op(input, weight, bias)
    """
    
    _operators = {}
    _fallbacks = {}
    
    @classmethod
    def register(cls, name: str, operator: Callable, fallback: Optional[Callable] = None):
        """
        注册算子
        
        Args:
            name: 算子名称
            operator: CANN 算子实现
            fallback: 回退实现（当 CANN 不可用时）
        """
        cls._operators[name] = operator
        if fallback is not None:
            cls._fallbacks[name] = fallback
    
    @classmethod
    def get_operator(cls, name: str) -> Callable:
        """
        获取算子
        
        Args:
            name: 算子名称
        
        Returns:
            算子实现函数
        """
        if ASCEND_AVAILABLE and name in cls._operators:
            return cls._operators[name]
        elif name in cls._fallbacks:
            return cls._fallbacks[name]
        else:
            raise KeyError(f"未找到算子: {name}")
    
    @classmethod
    def has_operator(cls, name: str) -> bool:
        """检查算子是否存在"""
        return name in cls._operators or name in cls._fallbacks


def cann_linear(input: torch.Tensor, weight: torch.Tensor, 
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    CANN 优化的线性层算子
    
    利用昇腾 AI Core 的矩阵计算单元加速线性变换。
    
    Args:
        input: 输入张量，形状为 (batch_size, in_features)
        weight: 权重张量，形状为 (out_features, in_features)
        bias: 偏置张量，形状为 (out_features,)
    
    Returns:
        输出张量，形状为 (batch_size, out_features)
    """
    if ASCEND_AVAILABLE:
        try:
            # 使用 torch_npu 的优化实现
            return torch_npu.npu_linear(input, weight, bias)
        except (AttributeError, RuntimeError):
            # 降级到标准实现
            pass
    
    # 标准实现
    return F.linear(input, weight, bias)


def cann_linear_quantized(input: torch.Tensor, weight_quantized: torch.Tensor,
                          scale: torch.Tensor, zero_point: torch.Tensor,
                          bias: Optional[torch.Tensor] = None,
                          weight_bits: int = 4) -> torch.Tensor:
    """
    CANN 优化的量化线性层算子
    
    利用昇腾 AI Core 的 INT8/INT4 指令加速量化矩阵乘法。
    
    Args:
        input: 输入张量 (FP16/FP32)
        weight_quantized: 量化后的权重 (INT4/INT8)
        scale: 缩放因子
        zero_point: 零点
        bias: 偏置
        weight_bits: 权重量化位数 (4 或 8)
    
    Returns:
        输出张量
    """
    if ASCEND_AVAILABLE and weight_bits in [4, 8]:
        try:
            # 使用 CANN 量化算子
            return torch_npu.npu_quantized_matmul(
                input, weight_quantized, scale, zero_point, bias
            )
        except (AttributeError, RuntimeError):
            pass
    
    # 降级实现：反量化后计算
    weight = (weight_quantized.float() - zero_point) * scale
    return F.linear(input, weight, bias)


def cann_conv2d(input: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                stride: Tuple[int, int] = (1, 1),
                padding: Tuple[int, int] = (0, 0),
                dilation: Tuple[int, int] = (1, 1),
                groups: int = 1) -> torch.Tensor:
    """
    CANN 优化的 2D 卷积算子
    
    利用昇腾 AI Core 的 Cube 计算单元加速卷积运算。
    
    Args:
        input: 输入张量 (N, C, H, W)
        weight: 权重张量 (out_channels, in_channels // groups, kH, kW)
        bias: 偏置张量
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
    
    Returns:
        输出张量
    """
    if ASCEND_AVAILABLE:
        try:
            return torch_npu.npu_conv2d(
                input, weight, bias, stride, padding, dilation, groups
            )
        except (AttributeError, RuntimeError):
            pass
    
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


def cann_layer_norm(input: torch.Tensor, normalized_shape: List[int],
                    weight: Optional[torch.Tensor] = None,
                    bias: Optional[torch.Tensor] = None,
                    eps: float = 1e-5) -> torch.Tensor:
    """
    CANN 优化的层归一化算子
    
    Args:
        input: 输入张量
        normalized_shape: 归一化维度
        weight: 缩放参数
        bias: 偏移参数
        eps: 数值稳定性常数
    
    Returns:
        归一化后的张量
    """
    if ASCEND_AVAILABLE:
        try:
            return torch_npu.npu_layer_norm(input, normalized_shape, weight, bias, eps)
        except (AttributeError, RuntimeError):
            pass
    
    return F.layer_norm(input, normalized_shape, weight, bias, eps)


def cann_softmax(input: torch.Tensor, dim: int = -1,
                 dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    CANN 优化的 Softmax 算子
    
    Args:
        input: 输入张量
        dim: 计算维度
        dtype: 输出数据类型
    
    Returns:
        Softmax 结果
    """
    if ASCEND_AVAILABLE:
        try:
            return torch_npu.npu_softmax(input, dim, dtype)
        except (AttributeError, RuntimeError):
            pass
    
    return F.softmax(input, dim=dim, dtype=dtype)


def cann_gelu(input: torch.Tensor) -> torch.Tensor:
    """
    CANN 优化的 GELU 激活函数
    
    使用昇腾优化的近似算法实现。
    
    Args:
        input: 输入张量
    
    Returns:
        GELU 结果
    """
    if ASCEND_AVAILABLE:
        try:
            return torch_npu.npu_gelu(input)
        except (AttributeError, RuntimeError):
            pass
    
    return F.gelu(input)


def cann_silu(input: torch.Tensor) -> torch.Tensor:
    """
    CANN 优化的 SiLU/Swish 激活函数
    
    Args:
        input: 输入张量
    
    Returns:
        SiLU 结果
    """
    if ASCEND_AVAILABLE:
        try:
            return torch_npu.npu_silu(input)
        except (AttributeError, RuntimeError):
            pass
    
    return F.silu(input)


def cann_quantize_per_tensor(input: torch.Tensor, 
                              bits: int = 8,
                              symmetric: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CANN 优化的逐张量量化
    
    Args:
        input: 输入张量
        bits: 量化位数 (4 或 8)
        symmetric: 是否对称量化
    
    Returns:
        (量化后的张量, scale, zero_point)
    """
    if bits == 4:
        qmin, qmax = -8, 7
    elif bits == 8:
        qmin, qmax = -128, 127
    else:
        raise ValueError(f"不支持的量化位数: {bits}")
    
    if ASCEND_AVAILABLE:
        try:
            quantized, scale, zero_point = torch_npu.npu_quantize_per_tensor(
                input, bits, symmetric
            )
            return quantized, scale, zero_point
        except (AttributeError, RuntimeError):
            pass
    
    # 降级实现
    if symmetric:
        max_abs = torch.max(torch.abs(input))
        scale = max_abs / qmax
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.zeros_like(scale)
    else:
        min_val = torch.min(input)
        max_val = torch.max(input)
        scale = (max_val - min_val) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = qmin - min_val / scale
    
    quantized = torch.clamp(torch.round(input / scale + zero_point), qmin, qmax)
    return quantized.to(torch.int8), scale, zero_point


def cann_dequantize(quantized: torch.Tensor, 
                    scale: torch.Tensor,
                    zero_point: torch.Tensor) -> torch.Tensor:
    """
    CANN 优化的反量化
    
    Args:
        quantized: 量化后的张量
        scale: 缩放因子
        zero_point: 零点
    
    Returns:
        反量化后的张量
    """
    if ASCEND_AVAILABLE:
        try:
            return torch_npu.npu_dequantize(quantized, scale, zero_point)
        except (AttributeError, RuntimeError):
            pass
    
    return (quantized.float() - zero_point) * scale


def cann_fused_add_layernorm(input: torch.Tensor, 
                              residual: torch.Tensor,
                              normalized_shape: List[int],
                              weight: torch.Tensor,
                              bias: torch.Tensor,
                              eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CANN 融合 Add + LayerNorm 算子
    
    将残差加法和层归一化融合为单个算子，减少内存访问。
    
    Args:
        input: 输入张量
        residual: 残差张量
        normalized_shape: 归一化维度
        weight: 缩放参数
        bias: 偏移参数
        eps: 数值稳定性常数
    
    Returns:
        (归一化结果, 残差和)
    """
    if ASCEND_AVAILABLE:
        try:
            norm_out, add_out = torch_npu.npu_fused_add_layernorm(
                input, residual, normalized_shape, weight, bias, eps
            )
            return norm_out, add_out
        except (AttributeError, RuntimeError):
            pass
    
    # 降级实现
    add_out = input + residual
    norm_out = F.layer_norm(add_out, normalized_shape, weight, bias, eps)
    return norm_out, add_out


def cann_fused_bias_gelu(input: torch.Tensor, 
                          bias: torch.Tensor) -> torch.Tensor:
    """
    CANN 融合 Bias + GELU 算子
    
    Args:
        input: 输入张量
        bias: 偏置张量
    
    Returns:
        融合计算结果
    """
    if ASCEND_AVAILABLE:
        try:
            return torch_npu.npu_fused_bias_gelu(input, bias)
        except (AttributeError, RuntimeError):
            pass
    
    return F.gelu(input + bias)


def cann_fused_softmax_dropout(input: torch.Tensor,
                                dropout_p: float = 0.0,
                                training: bool = True,
                                dim: int = -1) -> torch.Tensor:
    """
    CANN 融合 Softmax + Dropout 算子
    
    Args:
        input: 输入张量
        dropout_p: Dropout 概率
        training: 是否训练模式
        dim: Softmax 维度
    
    Returns:
        融合计算结果
    """
    if ASCEND_AVAILABLE and training:
        try:
            return torch_npu.npu_fused_softmax_dropout(input, dropout_p, dim)
        except (AttributeError, RuntimeError):
            pass
    
    output = F.softmax(input, dim=dim)
    if training and dropout_p > 0:
        output = F.dropout(output, p=dropout_p, training=training)
    return output


class CANNLinear(nn.Module):
    """
    CANN 优化的线性层
    
    自动选择 CANN 或 PyTorch 实现，支持量化训练。
    
    Args:
        in_features: 输入特征数
        out_features: 输出特征数
        bias: 是否使用偏置
        quantized: 是否使用量化
        weight_bits: 权重量化位数
    
    Example:
        >>> linear = CANNLinear(768, 3072, quantized=True, weight_bits=4)
        >>> output = linear(input)
    """
    
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True,
                 quantized: bool = False,
                 weight_bits: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantized = quantized
        self.weight_bits = weight_bits
        
        if quantized:
            # 量化权重存储
            self.weight_quantized = nn.Parameter(
                torch.zeros(out_features, in_features, dtype=torch.int8),
                requires_grad=False
            )
            self.weight_scale = nn.Parameter(
                torch.ones(out_features, 1),
                requires_grad=False
            )
            self.weight_zero_point = nn.Parameter(
                torch.zeros(out_features, 1),
                requires_grad=False
            )
            # FP32 主权重用于训练
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features),
                requires_grad=True
            )
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize_weight(self):
        """量化权重"""
        if self.quantized:
            quantized, scale, zp = cann_quantize_per_tensor(
                self.weight.data, bits=self.weight_bits
            )
            self.weight_quantized.data = quantized
            self.weight_scale.data = scale.view(-1, 1)
            self.weight_zero_point.data = zp.view(-1, 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantized and not self.training:
            self.quantize_weight()
            return cann_linear_quantized(
                input, self.weight_quantized, self.weight_scale,
                self.weight_zero_point, self.bias, self.weight_bits
            )
        return cann_linear(input, self.weight, self.bias)


class CANNLayerNorm(nn.Module):
    """
    CANN 优化的层归一化
    
    Args:
        normalized_shape: 归一化维度
        eps: 数值稳定性常数
        elementwise_affine: 是否学习缩放和偏移参数
    """
    
    def __init__(self, normalized_shape: Union[int, List[int]],
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cann_layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class CANNAttention(nn.Module):
    """
    CANN 优化的注意力模块
    
    融合 QKV 投影、注意力计算和输出投影。
    
    Args:
        hidden_size: 隐藏层大小
        num_attention_heads: 注意力头数
        attention_dropout: 注意力 Dropout 概率
        output_dropout: 输出 Dropout 概率
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 attention_dropout: float = 0.0,
                 output_dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        
        assert hidden_size % num_attention_heads == 0, \
            "hidden_size 必须能被 num_attention_heads 整除"
        
        self.query = CANNLinear(hidden_size, hidden_size)
        self.key = CANNLinear(hidden_size, hidden_size)
        self.value = CANNLinear(hidden_size, hidden_size)
        self.output = CANNLinear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(output_dropout)
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV 投影
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # 重塑为多头形式
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax + Dropout
        attention_probs = cann_fused_softmax_dropout(
            attention_scores, self.attention_dropout, self.training
        )
        
        # 注意力输出
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        # 输出投影
        output = self.output(context)
        output = self.dropout(output)
        
        return output


class CANNMLP(nn.Module):
    """
    CANN 优化的 MLP 模块
    
    融合线性层、激活函数和 Dropout。
    
    Args:
        hidden_size: 隐藏层大小
        intermediate_size: 中间层大小
        hidden_dropout: Dropout 概率
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_dropout: float = 0.0):
        super().__init__()
        self.dense_h_to_4h = CANNLinear(hidden_size, intermediate_size)
        self.dense_4h_to_h = CANNLinear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = cann_gelu(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CANNTransformerLayer(nn.Module):
    """
    CANN 优化的 Transformer 层
    
    融合注意力、MLP 和残差连接。
    
    Args:
        hidden_size: 隐藏层大小
        num_attention_heads: 注意力头数
        intermediate_size: 中间层大小
        attention_dropout: 注意力 Dropout 概率
        hidden_dropout: 隐藏层 Dropout 概率
        layer_norm_eps: 层归一化 epsilon
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 intermediate_size: int,
                 attention_dropout: float = 0.0,
                 hidden_dropout: float = 0.0,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        self.attention = CANNAttention(hidden_size, num_attention_heads, attention_dropout)
        self.mlp = CANNMLP(hidden_size, intermediate_size, hidden_dropout)
        self.input_layernorm = CANNLayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = CANNLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN 架构
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# 注册算子
def _register_operators():
    """注册所有 CANN 算子"""
    CANNOperatorRegistry.register('linear', cann_linear, F.linear)
    CANNOperatorRegistry.register('linear_quantized', cann_linear_quantized, None)
    CANNOperatorRegistry.register('conv2d', cann_conv2d, F.conv2d)
    CANNOperatorRegistry.register('layer_norm', cann_layer_norm, F.layer_norm)
    CANNOperatorRegistry.register('softmax', cann_softmax, F.softmax)
    CANNOperatorRegistry.register('gelu', cann_gelu, F.gelu)
    CANNOperatorRegistry.register('silu', cann_silu, F.silu)
    CANNOperatorRegistry.register('quantize', cann_quantize_per_tensor, None)
    CANNOperatorRegistry.register('dequantize', cann_dequantize, None)
    CANNOperatorRegistry.register('fused_add_layernorm', cann_fused_add_layernorm, None)
    CANNOperatorRegistry.register('fused_bias_gelu', cann_fused_bias_gelu, None)
    CANNOperatorRegistry.register('fused_softmax_dropout', cann_fused_softmax_dropout, None)


# 模块加载时注册算子
_register_operators()

# 导入 math 模块
import math
