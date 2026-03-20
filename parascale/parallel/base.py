# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : base.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 并行策略基类模块

本模块定义了所有并行策略的抽象基类，
提供公共的参数验证、初始化和设备管理功能。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BaseParallel(ABC):
    """
    并行策略抽象基类
    
    定义了所有并行策略必须实现的接口，并提供公共的参数验证
    和初始化逻辑。所有具体的并行策略类（DataParallel、
    TensorParallel、PipelineParallel 等）都应继承此类。
    
    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank（进程标识）
        world_size: 世界大小（参与训练的进程总数）
        device: 当前进程使用的设备
    
    Example:
        >>> class MyParallel(BaseParallel):
        ...     def forward(self, inputs):
        ...         return self.model(inputs)
    """
    
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        """
        初始化并行策略基类
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank，取值范围 [0, world_size-1]
            world_size: 世界大小，表示参与训练的进程总数
        
        Raises:
            ValueError: 当参数不合法时抛出
        """
        self._validate_params(model, rank, world_size)
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    def _validate_params(self, model: nn.Module, rank: int, world_size: int) -> None:
        """
        验证初始化参数的合法性
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank
            world_size: 世界大小
        
        Raises:
            ValueError: 当任何参数不合法时抛出
        """
        if model is None:
            raise ValueError("Model cannot be None")
        if rank < 0:
            raise ValueError("Rank must be non-negative")
        if world_size <= 0:
            raise ValueError("World size must be positive")
        if rank >= world_size:
            raise ValueError(f"Rank {rank} cannot be >= world size {world_size}")
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        前向传播（抽象方法）
        
        子类必须实现此方法，定义具体的并行前向传播逻辑。
        
        Args:
            inputs: 输入数据张量
        
        Returns:
            模型输出张量，某些并行策略可能返回 None
        """
        pass
    
    def gather_gradients(self) -> None:
        """
        收集梯度
        
        在数据并行和张量并行模式下，需要在参数更新前
        收集所有进程的梯度。默认为空实现，子类可以覆盖。
        """
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将张量移动到当前设备
        
        Args:
            tensor: 要移动的张量
        
        Returns:
            移动到当前设备后的张量
        """
        return tensor.to(self.device)
