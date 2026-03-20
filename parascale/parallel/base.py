# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : base.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 并行策略基类模块

本模块定义了所有并行策略的抽象基类，
提供公共的参数验证、初始化和设备管理功能。

架构设计:
    - BaseParallel: 所有并行策略的抽象基类
    - 子类必须实现 forward() 方法
    - 子类可以选择性覆盖 gather_gradients()、broadcast_model() 等方法
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseParallel(ABC):
    """
    并行策略抽象基类

    定义了所有并行策略必须实现的接口，并提供公共的参数验证
    和初始化逻辑。所有具体的并行策略类（DataParallel、
    TensorParallel、PipelineParallel 等）都应继承此类。

    抽象层次设计:
        1. forward(): 必须实现，定义前向传播逻辑
        2. gather_gradients(): 可选覆盖，梯度收集逻辑
        3. broadcast_model(): 可选覆盖，模型广播逻辑

    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank（进程标识）
        world_size: 世界大小（参与训练的进程总数）
        device: 当前进程使用的设备
        parallel_config: 并行策略配置信息

    Example:
        >>> class MyParallel(BaseParallel):
        ...     def forward(self, inputs):
        ...         return self.model(inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        parallel_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化并行策略基类

        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank，取值范围 [0, world_size-1]
            world_size: 世界大小，表示参与训练的进程总数
            parallel_config: 并行策略配置信息，可选

        Raises:
            ValueError: 当参数不合法时抛出
            TypeError: 当 model 不是 nn.Module 实例时抛出
        """
        self._validate_params(model, rank, world_size)
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.parallel_config = parallel_config or {}
        self.device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )
        self._is_initialized = False

    @staticmethod
    def _validate_params(
        model: nn.Module,
        rank: int,
        world_size: int
    ) -> None:
        """
        验证初始化参数的合法性

        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank
            world_size: 世界大小

        Raises:
            TypeError: 当 model 不是 nn.Module 实例时抛出
            ValueError: 当任何参数不合法时抛出
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be an nn.Module instance, got {type(model)}"
            )
        if rank < 0:
            raise ValueError(f"Rank must be non-negative, got {rank}")
        if world_size <= 0:
            raise ValueError(f"World size must be positive, got {world_size}")
        if rank >= world_size:
            raise ValueError(
                f"Rank {rank} cannot be >= world size {world_size}"
            )

    def validate_distributed_state(self) -> None:
        """
        验证分布式训练状态

        确保在分布式环境下正确初始化。
        子类可以覆盖此方法以添加特定检查。

        Raises:
            RuntimeError: 当分布式状态不合法时抛出
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                f"Distributed environment not initialized for rank {self.rank}. "
                "Please call initialize_distributed() first."
            )

        actual_rank = dist.get_rank()
        actual_world_size = dist.get_world_size()

        if actual_rank != self.rank:
            raise RuntimeError(
                f"Rank mismatch: expected {self.rank}, got {actual_rank}"
            )
        if actual_world_size != self.world_size:
            raise RuntimeError(
                f"World size mismatch: expected {self.world_size}, got {actual_world_size}"
            )

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

        Raises:
            RuntimeError: 当分布式状态不合法时抛出
        """
        pass

    def broadcast_model(self) -> None:
        """
        广播模型参数

        将 rank 0 的模型参数广播到所有其他进程。
        默认为空实现，子类可以覆盖。

        Raises:
            RuntimeError: 当分布式状态不合法时抛出
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

    def get_parallel_info(self) -> Dict[str, Any]:
        """
        获取并行配置信息

        Returns:
            包含并行配置信息的字典
        """
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "device": str(self.device),
            "parallel_type": self.__class__.__name__,
            "config": self.parallel_config,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"rank={self.rank}, "
            f"world_size={self.world_size}, "
            f"device={self.device})"
        )