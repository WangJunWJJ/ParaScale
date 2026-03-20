# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : observers.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 量化观察器模块

本模块实现了用于统计激活值和权重范围的观察器，
为计算量化参数（scale 和 zero_point）提供数据支持。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import QuantizationConfig


class BaseObserver:
    """
    观察器基类
    
    用于统计张量的数值范围（min/max），为量化参数计算提供数据。
    
    Attributes:
        config: 量化配置
        min_val: 观察到的最小值
        max_val: 观察到的最大值
    """
    
    def __init__(self, config: QuantizationConfig):
        """
        初始化观察器
        
        Args:
            config: 量化配置实例
        """
        self.config = config
        self.min_val: Optional[torch.Tensor] = None
        self.max_val: Optional[torch.Tensor] = None
    
    def update(self, x: torch.Tensor) -> None:
        """
        更新观察到的数值范围
        
        Args:
            x: 输入张量
        """
        raise NotImplementedError
    
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算量化参数（scale 和 zero_point）
        
        Returns:
            元组 (scale, zero_point)
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("观察器尚未观察到任何数据")
        
        qmin, qmax = self.config.get_qmin_qmax()
        
        if self.config.scheme == "symmetric":
            # 对称量化：zero_point = 0
            max_abs = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
            scale = max_abs / qmax
            zero_point = torch.zeros_like(scale)
        else:
            # 非对称量化
            scale = (self.max_val - self.min_val) / (qmax - qmin)
            zero_point = qmin - self.min_val / scale
        
        # 防止 scale 为 0
        scale = torch.clamp(scale, min=1e-8)
        
        return scale, zero_point
    
    def reset(self) -> None:
        """重置观察器状态"""
        self.min_val = None
        self.max_val = None


class MinMaxObserver(BaseObserver):
    """
    MinMax 观察器
    
    记录观察到的最小值和最大值，用于计算量化参数。
    适用于静态范围的数据。
    
    Example:
        >>> observer = MinMaxObserver(config)
        >>> for data in dataloader:
        ...     observer.update(data)
        >>> scale, zero_point = observer.calculate_qparams()
    """
    
    def update(self, x: torch.Tensor) -> None:
        """
        更新观察到的数值范围
        
        Args:
            x: 输入张量
        """
        if self.config.per_channel and x.dim() > 1:
            # 逐通道统计：在除了第 0 维（batch）的其他维度上求 min/max
            dims = list(range(1, x.dim()))
            min_val = torch.min(x, dim=dims[0], keepdim=True)[0]
            max_val = torch.max(x, dim=dims[0], keepdim=True)[0]
            for dim in dims[1:]:
                min_val = torch.min(min_val, dim=dim, keepdim=True)[0]
                max_val = torch.max(max_val, dim=dim, keepdim=True)[0]
        else:
            # 逐张量统计
            min_val = torch.min(x)
            max_val = torch.max(x)
        
        if self.min_val is None:
            self.min_val = min_val
            self.max_val = max_val
        else:
            self.min_val = torch.min(self.min_val, min_val)
            self.max_val = torch.max(self.max_val, max_val)


class MovingAverageObserver(BaseObserver):
    """
    移动平均观察器
    
    使用移动平均来平滑观察到的数值范围，
    适用于动态范围的数据。
    
    Attributes:
        ratio: 移动平均比例，新数据的权重
    
    Example:
        >>> observer = MovingAverageObserver(config, ratio=0.1)
        >>> for data in dataloader:
        ...     observer.update(data)
        >>> scale, zero_point = observer.calculate_qparams()
    """
    
    def __init__(self, config: QuantizationConfig):
        """
        初始化移动平均观察器
        
        Args:
            config: 量化配置实例
        """
        super().__init__(config)
        self.ratio = config.moving_average_ratio
    
    def update(self, x: torch.Tensor) -> None:
        """
        使用移动平均更新观察到的数值范围
        
        Args:
            x: 输入张量
        """
        if self.config.per_channel and x.dim() > 1:
            # 逐通道统计
            dims = list(range(1, x.dim()))
            min_val = torch.min(x, dim=dims[0], keepdim=True)[0]
            max_val = torch.max(x, dim=dims[0], keepdim=True)[0]
            for dim in dims[1:]:
                min_val = torch.min(min_val, dim=dim, keepdim=True)[0]
                max_val = torch.max(max_val, dim=dim, keepdim=True)[0]
        else:
            # 逐张量统计
            min_val = torch.min(x)
            max_val = torch.max(x)
        
        if self.min_val is None:
            self.min_val = min_val
            self.max_val = max_val
        else:
            # 移动平均更新
            self.min_val = self.ratio * min_val + (1 - self.ratio) * self.min_val
            self.max_val = self.ratio * max_val + (1 - self.ratio) * self.max_val
