# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : data_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 数据并行模块

本模块实现了数据并行策略，将数据分割到不同 GPU，
每个 GPU 运行完整的模型副本，通过梯度同步实现并行训练。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed
from typing import Optional, Any
from .base import BaseParallel


class DataParallel(BaseParallel):
    """
    数据并行策略类
    
    数据并行是最常用的并行策略，每个 GPU 持有完整的模型副本，
    处理不同的数据子集。在反向传播时，通过 all-reduce 操作
    同步所有 GPU 的梯度。
    
    工作流程：
    1. 将数据分割到不同 GPU
    2. 每个 GPU 独立执行前向和反向传播
    3. 通过 all-reduce 同步梯度
    4. 所有 GPU 使用相同的梯度更新参数
    
    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank
        world_size: 世界大小
        device: 当前设备
    
    Example:
        >>> model = SimpleModel()
        >>> dp = DataParallel(model, rank=0, world_size=2)
        >>> output = dp.forward(inputs)
    """
    
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        """
        初始化数据并行策略
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank
            world_size: 世界大小（GPU 数量）
        """
        super().__init__(model, rank, world_size)
        self.model.to(self.device)
    
    def broadcast_model(self) -> None:
        """
        广播模型参数
        
        将 rank 0 的模型参数广播到所有其他进程，
        确保所有进程的模型初始化状态一致。
        通常在训练开始前调用一次。
        """
        for param in self.model.parameters():
            dist.broadcast(param, src=0)
    
    def gather_gradients(self) -> None:
        """
        收集并平均梯度
        
        使用 all-reduce 操作收集所有进程的梯度，
        并计算平均值。这是数据并行的核心操作，
        确保所有进程使用相同的梯度更新参数。
        """
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size
    
    def prepare_dataloader(
        self, 
        dataset: Any, 
        batch_size: int, 
        shuffle: bool = True, 
        num_workers: int = 0
    ) -> torch.utils.data.DataLoader:
        """
        准备分布式数据加载器
        
        创建一个使用 DistributedSampler 的数据加载器，
        自动将数据分割到不同进程。
        
        Args:
            dataset: 数据集实例
            batch_size: 批次大小（每个 GPU 的批次大小）
            shuffle: 是否打乱数据，默认为 True
            num_workers: 数据加载线程数，默认为 0
        
        Returns:
            配置好的分布式数据加载器
        
        Example:
            >>> dataset = CIFAR10(root='./data', train=True)
            >>> dataloader = dp.prepare_dataloader(dataset, batch_size=32)
        """
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            shuffle=shuffle
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler, 
            num_workers=num_workers
        )
        return dataloader
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        将输入数据移动到当前设备，然后执行模型前向传播。
        
        Args:
            inputs: 输入数据张量
        
        Returns:
            模型输出张量
        """
        inputs = self.to_device(inputs)
        return self.model(inputs)
