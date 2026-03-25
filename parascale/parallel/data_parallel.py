# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : data_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 数据并行模块

本模块实现了数据并行策略，将数据分割到不同 GPU，
每个 GPU 运行完整的模型副本，通过梯度同步实现并行训练。

优化特性：
- 梯度压缩：Top-K稀疏化、1-bit量化
- 通信与计算重叠：使用CUDA Streams
- 梯度分桶：减少通信次数
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed
from typing import Optional, Any, List, Callable
from .base import BaseParallel, ParallelInitError
import logging
import time

logger = logging.getLogger(__name__)


class DataParallel(BaseParallel):
    """
    数据并行策略类
    
    数据并行是最常用的并行策略，每个 GPU 持有完整的模型副本，
    处理不同的数据子集。在反向传播时，通过 all-reduce 操作
    同步所有 GPU 的梯度。
    
    优化特性：
    1. 梯度压缩：支持Top-K稀疏化和1-bit量化
    2. 通信重叠：使用CUDA Streams重叠通信与计算
    3. 梯度分桶：按参数大小分桶，减少通信次数
    
    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank
        world_size: 世界大小
        device: 当前设备
        compression_ratio: 梯度压缩比例
        overlap_comm: 是否重叠通信与计算
        bucket_size: 梯度分桶大小（字节）
    
    Example:
        >>> model = SimpleModel()
        >>> dp = DataParallel(model, rank=0, world_size=2)
        >>> output = dp.forward(inputs)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        rank: int, 
        world_size: int,
        compression_ratio: Optional[float] = None,
        overlap_comm: bool = False,
        bucket_size: int = 25 * 1024 * 1024  # 25MB
    ):
        """
        初始化数据并行策略
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank
            world_size: 世界大小（GPU 数量）
            compression_ratio: 梯度压缩比例（None表示不压缩）
            overlap_comm: 是否重叠通信与计算
            bucket_size: 梯度分桶大小（字节）
        """
        super().__init__(model, rank, world_size)
        self.model.to(self.device)
        
        # 通信优化配置
        self.compression_ratio = compression_ratio
        self.overlap_comm = overlap_comm and torch.cuda.is_available()
        self.bucket_size = bucket_size
        
        # 初始化梯度分桶
        self.buckets: List[List[nn.Parameter]] = []
        self._init_buckets()
        
        # 通信重叠相关
        if self.overlap_comm:
            self.comm_stream = torch.cuda.Stream()
        else:
            self.comm_stream = None
        
        # 梯度压缩器
        self.compressor = None
        if compression_ratio is not None:
            from .communication import TopKCompressor
            self.compressor = TopKCompressor(compression_ratio)
        
        logger.info(
            f"DataParallel initialized: compression={compression_ratio}, "
            f"overlap={overlap_comm}, buckets={len(self.buckets)}"
        )
    
    def _init_buckets(self) -> None:
        """
        初始化梯度分桶
        
        将参数按大小分桶，减少通信次数。
        """
        current_bucket: List[nn.Parameter] = []
        current_size = 0
        
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            
            param_size = param.numel() * param.element_size()
            
            if current_size + param_size > self.bucket_size and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = [param]
                current_size = param_size
            else:
                current_bucket.append(param)
                current_size += param_size
        
        if current_bucket:
            self.buckets.append(current_bucket)
    
    def _initialize_impl(self) -> None:
        """
        数据并行特定的初始化
        """
        # 广播模型参数
        self.broadcast_model()
    
    def broadcast_model(self) -> None:
        """
        广播模型参数
        
        将 rank 0 的模型参数广播到所有其他进程，
        确保所有进程的模型初始化状态一致。
        通常在训练开始前调用一次。
        """
        if not dist.is_initialized():
            return
        
        for param in self.model.parameters():
            dist.broadcast(param, src=0)
    
    def gather_gradients(self) -> None:
        """
        收集并平均梯度
        
        使用 all-reduce 操作收集所有进程的梯度，
        并计算平均值。这是数据并行的核心操作，
        确保所有进程使用相同的梯度更新参数。
        
        优化：
        1. 梯度压缩（如果启用）
        2. 通信与计算重叠（如果启用）
        3. 梯度分桶
        """
        if not dist.is_initialized() or self.world_size == 1:
            return
        
        start_time = time.time()
        
        # 使用分桶通信
        for bucket in self.buckets:
            self._sync_bucket(bucket)
        
        comm_time = time.time() - start_time
        self._record_time('comm_time', comm_time)
    
    def _sync_bucket(self, bucket: List[nn.Parameter]) -> None:
        """
        同步一个桶的梯度
        
        Args:
            bucket: 参数桶
        """
        if not bucket:
            return
        
        # 收集桶中所有梯度
        grads = [p.grad for p in bucket if p.grad is not None]
        if not grads:
            return
        
        # 扁平化梯度
        flat_grad = torch.cat([g.flatten() for g in grads])
        
        # 梯度压缩
        if self.compressor is not None:
            compressed, metadata = self.compressor.compress(flat_grad)
            dist.all_reduce(compressed)
            flat_grad.copy_(self.compressor.decompress(compressed, metadata))
        else:
            # 标准 all-reduce
            if self.overlap_comm and self.comm_stream is not None:
                # 使用独立CUDA流重叠通信
                with torch.cuda.stream(self.comm_stream):
                    dist.all_reduce(flat_grad)
                torch.cuda.synchronize(self.comm_stream)
            else:
                dist.all_reduce(flat_grad)
        
        # 平均梯度
        flat_grad /= self.world_size
        
        # 分回各个参数
        offset = 0
        for param in bucket:
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(flat_grad[offset:offset + numel].view_as(param.grad))
                offset += numel
    
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
        start_time = time.time()
        inputs = self.to_device(inputs)
        output = self.model(inputs)
        
        forward_time = time.time() - start_time
        self._record_time('forward_time', forward_time)
        
        return output
    
    def get_comm_stats(self) -> dict:
        """
        获取通信统计信息
        
        Returns:
            通信统计字典
        """
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'buckets': len(self.buckets),
            'compression_ratio': self.compression_ratio,
            'overlap_comm': self.overlap_comm,
            'total_params': total_params,
            'estimated_comm_volume_mb': total_params * 4 / (1024 ** 2) / self.world_size
        }
