# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : fake_quantize.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 伪量化模块

本模块实现了伪量化层（FakeQuantize），在训练过程中模拟量化-反量化过程，
使模型能够适应量化后的推理环境。
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import QuantizationConfig
from .observers import BaseObserver, MinMaxObserver, MovingAverageObserver


class FakeQuantize(nn.Module):
    """
    伪量化层
    
    在训练过程中模拟量化-反量化过程，使模型能够学习适应量化误差。
    这是量化感知训练（QAT）的核心组件。
    
    工作流程：
    1. 使用观察器收集输入数据的统计信息（min/max）
    2. 计算量化参数（scale 和 zero_point）
    3. 量化：x_q = round((x - zero_point) / scale)
    4. 钳制：x_q = clamp(x_q, qmin, qmax)
    5. 反量化：x_dq = x_q * scale + zero_point
    6. 使用反量化后的值继续前向传播
    
    在训练过程中，使用 Straight-Through Estimator (STE) 来传递梯度，
    使得量化操作可微。
    
    Attributes:
        config: 量化配置
        observer: 观察器实例
        scale: 量化缩放因子
        zero_point: 量化零点
        fake_quant_enabled: 是否启用伪量化
        observer_enabled: 是否启用观察器
    
    Example:
        >>> config = QuantizationConfig(bits=8, scheme="symmetric")
        >>> fake_quant = FakeQuantize(config)
        >>> x = torch.randn(32, 3, 224, 224)
        >>> x_quantized = fake_quant(x)
    """
    
    def __init__(self, config: QuantizationConfig):
        """
        初始化伪量化层
        
        Args:
            config: 量化配置实例
        """
        super().__init__()
        self.config = config
        
        # 创建观察器
        if config.observer_type == "minmax":
            self.observer = MinMaxObserver(config)
        elif config.observer_type == "moving_average":
            self.observer = MovingAverageObserver(config)
        else:
            raise ValueError(f"不支持的观察器类型: {config.observer_type}")
        
        # 注册量化参数为 buffer
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        
        # 控制标志
        self.fake_quant_enabled = True
        self.observer_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            伪量化后的张量
        """
        # 收集统计信息
        if self.observer_enabled:
            self.observer.update(x.detach())
        
        # 如果不启用伪量化，直接返回输入
        if not self.fake_quant_enabled:
            return x
        
        # 计算量化参数
        if self.training:
            # 训练时动态计算
            scale, zero_point = self.observer.calculate_qparams()
            self.scale = scale
            self.zero_point = zero_point
        
        # 执行伪量化
        return self._fake_quantize(x, self.scale, self.zero_point)
    
    def _fake_quantize(
        self, 
        x: torch.Tensor, 
        scale: torch.Tensor, 
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        执行伪量化操作
        
        Args:
            x: 输入张量
            scale: 量化缩放因子
            zero_point: 量化零点
        
        Returns:
            伪量化后的张量
        """
        qmin, qmax = self.config.get_qmin_qmax()
        
        # 量化
        x_quant = torch.round(x / scale + zero_point)
        
        # 钳制到量化范围
        x_quant = torch.clamp(x_quant, qmin, qmax)
        
        # 反量化
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant
    
    def enable_fake_quant(self, enabled: bool = True) -> None:
        """
        启用/禁用伪量化
        
        Args:
            enabled: 是否启用
        """
        self.fake_quant_enabled = enabled
    
    def enable_observer(self, enabled: bool = True) -> None:
        """
        启用/禁用观察器
        
        Args:
            enabled: 是否启用
        """
        self.observer_enabled = enabled
    
    def calculate_qparams(self) -> tuple:
        """
        计算量化参数
        
        Returns:
            元组 (scale, zero_point)
        """
        return self.observer.calculate_qparams()
    
    def reset_observer(self) -> None:
        """重置观察器状态"""
        self.observer.reset()


class FakeQuantizedLinear(nn.Linear):
    """
    伪量化线性层
    
    在标准 Linear 层的基础上添加输入和权重的伪量化。
    
    Attributes:
        activation_fake_quant: 输入激活值的伪量化器
        weight_fake_quant: 权重的伪量化器
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        config: Optional[QuantizationConfig] = None
    ):
        """
        初始化伪量化线性层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置
            config: 量化配置
        """
        super().__init__(in_features, out_features, bias)
        
        if config is None:
            config = QuantizationConfig()
        
        # 创建伪量化器
        self.activation_fake_quant = FakeQuantize(config)
        self.weight_fake_quant = FakeQuantize(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        # 伪量化输入和权重
        x = self.activation_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        
        # 标准线性运算
        return nn.functional.linear(x, w, self.bias)


class FakeQuantizedConv2d(nn.Conv2d):
    """
    伪量化卷积层
    
    在标准 Conv2d 层的基础上添加输入和权重的伪量化。
    
    Attributes:
        activation_fake_quant: 输入激活值的伪量化器
        weight_fake_quant: 权重的伪量化器
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        config: Optional[QuantizationConfig] = None
    ):
        """
        初始化伪量化卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 空洞率
            groups: 分组数
            bias: 是否使用偏置
            padding_mode: 填充模式
            config: 量化配置
        """
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )
        
        if config is None:
            config = QuantizationConfig()
        
        # 创建伪量化器
        self.activation_fake_quant = FakeQuantize(config)
        self.weight_fake_quant = FakeQuantize(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        # 伪量化输入和权重
        x = self.activation_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        
        # 标准卷积运算
        return self._conv_forward(x, w, self.bias)
