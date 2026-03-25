# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : communication.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 通信优化模块

本模块实现了分布式训练中的通信优化技术，包括：
- 梯度压缩: Top-K稀疏化、1-bit Adam、误差补偿
- 通信计算重叠: 使用CUDA Streams实现通信与计算重叠
- All-Reduce优化: Ring-AllReduce、Tree-AllReduce

这些优化可以显著减少通信开销，提高分布式训练效率。
"""

import torch
import torch.distributed as dist
from typing import Optional, List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class GradientCompressor:
    """
    梯度压缩器基类
    
    用于减少分布式训练中的通信量。
    """
    
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, any]:
        """
        压缩张量
        
        Args:
            tensor: 要压缩的张量
        
        Returns:
            (压缩后的数据, 元数据)
        """
        raise NotImplementedError
    
    def decompress(self, compressed: torch.Tensor, metadata: any) -> torch.Tensor:
        """
        解压缩张量
        
        Args:
            compressed: 压缩后的数据
            metadata: 元数据
        
        Returns:
            解压缩后的张量
        """
        raise NotImplementedError


class TopKCompressor(GradientCompressor):
    """
    Top-K 稀疏化梯度压缩器
    
    只保留梯度中绝对值最大的K个元素，其余置零。
    可以显著减少通信量（10-100倍）。
    
    Attributes:
        compression_ratio: 压缩比例 (0-1)
        error_feedback: 是否启用误差补偿
        residual_errors: 累积的误差
    
    Example:
        >>> compressor = TopKCompressor(compression_ratio=0.01)
        >>> compressed, metadata = compressor.compress(gradient)
        >>> decompressed = compressor.decompress(compressed, metadata)
    """
    
    def __init__(self, compression_ratio: float = 0.01, error_feedback: bool = True):
        """
        初始化Top-K压缩器
        
        Args:
            compression_ratio: 压缩比例，保留的元素比例
            error_feedback: 是否启用误差补偿
        """
        if not 0 < compression_ratio <= 1:
            raise ValueError(f"compression_ratio must be in (0, 1], got {compression_ratio}")
        
        self.compression_ratio = compression_ratio
        self.error_feedback = error_feedback
        self.residual_errors = {}
    
    def compress(self, tensor: torch.Tensor, param_id: int = 0) -> Tuple[torch.Tensor, dict]:
        """
        使用Top-K稀疏化压缩张量
        
        Args:
            tensor: 要压缩的梯度张量
            param_id: 参数标识符（用于误差补偿）
        
        Returns:
            (压缩后的值, 元数据包含索引和形状)
        """
        # 添加误差补偿
        if self.error_feedback:
            if param_id not in self.residual_errors:
                self.residual_errors[param_id] = torch.zeros_like(tensor)
            tensor = tensor + self.residual_errors[param_id]
        
        # 计算K值
        numel = tensor.numel()
        k = max(1, int(numel * self.compression_ratio))
        
        # 获取Top-K索引
        flat_tensor = tensor.flatten()
        _, indices = torch.topk(torch.abs(flat_tensor), k)
        
        # 提取Top-K值
        values = flat_tensor[indices]
        
        # 记录误差
        if self.error_feedback:
            reconstructed = torch.zeros_like(flat_tensor)
            reconstructed[indices] = values
            self.residual_errors[param_id] = flat_tensor - reconstructed
        
        metadata = {
            'indices': indices,
            'shape': tensor.shape,
            'numel': numel,
        }
        
        return values, metadata
    
    def decompress(self, compressed: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        解压缩Top-K稀疏化张量
        
        Args:
            compressed: 压缩后的值
            metadata: 包含索引和形状的元数据
        
        Returns:
            解压缩后的张量
        """
        indices = metadata['indices']
        shape = metadata['shape']
        numel = metadata['numel']
        
        # 重建张量
        tensor = torch.zeros(numel, device=compressed.device, dtype=compressed.dtype)
        tensor[indices] = compressed
        
        return tensor.view(shape)
    
    def get_compression_stats(self) -> dict:
        """获取压缩统计信息"""
        return {
            'compression_ratio': self.compression_ratio,
            'error_feedback': self.error_feedback,
            'num_tracked_params': len(self.residual_errors),
        }


class OneBitAdamCompressor(GradientCompressor):
    """
    1-bit Adam 梯度压缩器
    
    将梯度量化为1-bit（正负号），大幅减少通信量。
    使用误差补偿保持收敛性。
    
    Reference:
        "1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed"
    
    Attributes:
        error_feedback: 是否启用误差补偿
        residual_errors: 累积的误差
    """
    
    def __init__(self, error_feedback: bool = True):
        """
        初始化1-bit压缩器
        
        Args:
            error_feedback: 是否启用误差补偿
        """
        self.error_feedback = error_feedback
        self.residual_errors = {}
    
    def compress(self, tensor: torch.Tensor, param_id: int = 0) -> Tuple[torch.Tensor, dict]:
        """
        使用1-bit量化压缩张量
        
        Args:
            tensor: 要压缩的梯度张量
            param_id: 参数标识符
        
        Returns:
            (符号位, 元数据包含尺度和形状)
        """
        # 添加误差补偿
        if self.error_feedback:
            if param_id not in self.residual_errors:
                self.residual_errors[param_id] = torch.zeros_like(tensor)
            tensor = tensor + self.residual_errors[param_id]
        
        # 计算尺度
        scale = torch.abs(tensor).mean()
        
        # 提取符号位
        signs = (tensor >= 0).to(torch.uint8)
        
        # 记录误差
        if self.error_feedback:
            reconstructed = (signs.float() * 2 - 1) * scale
            self.residual_errors[param_id] = tensor - reconstructed
        
        metadata = {
            'scale': scale,
            'shape': tensor.shape,
        }
        
        return signs, metadata
    
    def decompress(self, compressed: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        解压缩1-bit量化张量
        
        Args:
            compressed: 符号位
            metadata: 包含尺度和形状的元数据
        
        Returns:
            解压缩后的张量
        """
        scale = metadata['scale']
        shape = metadata['shape']
        
        # 重建张量
        tensor = (compressed.float() * 2 - 1) * scale
        
        return tensor.view(shape)


class CommunicationOverlap:
    """
    通信计算重叠管理器
    
    使用CUDA Streams实现通信与计算的重叠，
    隐藏通信延迟，提高训练效率。
    
    Attributes:
        compute_stream: 计算CUDA流
        comm_stream: 通信CUDA流
        overlap_enabled: 是否启用重叠
    
    Example:
        >>> overlap = CommunicationOverlap()
        >>> with overlap.compute_context():
        ...     loss.backward()
        >>> overlap.sync_gradients_async(parameters)
    """
    
    def __init__(self, overlap_enabled: bool = True):
        """
        初始化通信重叠管理器
        
        Args:
            overlap_enabled: 是否启用通信计算重叠
        """
        self.overlap_enabled = overlap_enabled and torch.cuda.is_available()
        
        if self.overlap_enabled:
            self.compute_stream = torch.cuda.Stream()
            self.comm_stream = torch.cuda.Stream()
            self.events = []
        else:
            self.compute_stream = None
            self.comm_stream = None
            self.events = []
    
    def compute_context(self):
        """
        获取计算上下文管理器
        
        Returns:
            CUDA流上下文
        """
        if self.overlap_enabled:
            return torch.cuda.stream(self.compute_stream)
        else:
            # 返回一个虚拟上下文
            from contextlib import nullcontext
            return nullcontext()
    
    def sync_gradients_async(
        self,
        parameters: List[torch.nn.Parameter],
        compressor: Optional[GradientCompressor] = None
    ) -> List[torch.cuda.Event]:
        """
        异步同步梯度
        
        Args:
            parameters: 要同步的参数列表
            compressor: 可选的梯度压缩器
        
        Returns:
            CUDA事件列表，用于等待同步完成
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            # 非分布式环境或单进程，无需同步
            return []
        
        if not self.overlap_enabled:
            # 同步执行
            for param in parameters:
                if param.grad is not None:
                    if compressor is not None:
                        compressed, metadata = compressor.compress(param.grad)
                        dist.all_reduce(compressed)
                        param.grad.copy_(compressor.decompress(compressed, metadata))
                    else:
                        dist.all_reduce(param.grad)
            return []
        
        events = []
        
        with torch.cuda.stream(self.comm_stream):
            for i, param in enumerate(parameters):
                if param.grad is not None:
                    # 等待计算完成
                    if self.compute_stream:
                        event = torch.cuda.Event()
                        event.record(self.compute_stream)
                        event.wait(self.comm_stream)
                        events.append(event)
                    
                    # 执行通信
                    if compressor is not None:
                        compressed, metadata = compressor.compress(param.grad, param_id=i)
                        dist.all_reduce(compressed)
                        param.grad.copy_(compressor.decompress(compressed, metadata))
                    else:
                        dist.all_reduce(param.grad)
        
        return events
    
    def wait_for_communication(self, events: List[torch.cuda.Event]) -> None:
        """
        等待通信完成
        
        Args:
            events: CUDA事件列表
        """
        if not self.overlap_enabled:
            return
        
        torch.cuda.synchronize(self.comm_stream)
    
    def all_reduce_bucket(
        self,
        bucket: List[torch.Tensor],
        compressor: Optional[GradientCompressor] = None
    ) -> None:
        """
        对一个桶执行all-reduce
        
        Args:
            bucket: 张量桶
            compressor: 可选的梯度压缩器
        """
        if not bucket:
            return
        
        # 扁平化桶
        flat = torch.cat([t.flatten() for t in bucket])
        
        # 压缩并通信
        if compressor is not None:
            compressed, metadata = compressor.compress(flat)
            dist.all_reduce(compressed)
            flat.copy_(compressor.decompress(compressed, metadata))
        else:
            dist.all_reduce(flat)
        
        # 分回各个张量
        offset = 0
        for tensor in bucket:
            numel = tensor.numel()
            tensor.copy_(flat[offset:offset + numel].view_as(tensor))
            offset += numel


class RingAllReduce:
    """
    Ring All-Reduce 实现
    
    使用环形拓扑进行all-reduce，比传统的tree-based all-reduce
    在某些网络拓扑下更高效。
    
    Reference:
        "Baidu's Ring All-Reduce"
    """
    
    def __init__(self, world_size: int, rank: int):
        """
        初始化Ring All-Reduce
        
        Args:
            world_size: 世界大小
            rank: 当前rank
        """
        self.world_size = world_size
        self.rank = rank
        self.next_rank = (rank + 1) % world_size
        self.prev_rank = (rank - 1 + world_size) % world_size
    
    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        执行reduce-scatter操作
        
        Args:
            tensor: 输入张量
        
        Returns:
            分片后的结果
        """
        chunk_size = tensor.numel() // self.world_size
        chunks = [tensor[i * chunk_size:(i + 1) * chunk_size].clone() 
                  for i in range(self.world_size)]
        
        # Ring reduce-scatter
        recv_buffer = torch.zeros_like(chunks[0])
        
        for step in range(self.world_size - 1):
            send_idx = (self.rank - step + self.world_size) % self.world_size
            recv_idx = (self.rank - step - 1 + self.world_size) % self.world_size
            
            # 发送和接收
            send_req = dist.isend(chunks[send_idx], self.next_rank)
            recv_req = dist.irecv(recv_buffer, self.prev_rank)
            
            recv_req.wait()
            send_req.wait()
            
            chunks[recv_idx] += recv_buffer
        
        return chunks[self.rank]
    
    def all_gather(self, shard: torch.Tensor, output: torch.Tensor) -> None:
        """
        执行all-gather操作
        
        Args:
            shard: 输入分片
            output: 输出张量
        """
        chunk_size = shard.numel()
        chunks = [output[i * chunk_size:(i + 1) * chunk_size] 
                  for i in range(self.world_size)]
        
        # 先放入自己的分片
        chunks[self.rank].copy_(shard)
        
        recv_buffer = torch.zeros_like(shard)
        
        for step in range(self.world_size - 1):
            send_idx = (self.rank - step + self.world_size) % self.world_size
            recv_idx = (self.rank - step - 1 + self.world_size) % self.world_size
            
            send_req = dist.isend(chunks[send_idx], self.next_rank)
            recv_req = dist.irecv(recv_buffer, self.prev_rank)
            
            recv_req.wait()
            send_req.wait()
            
            chunks[recv_idx].copy_(recv_buffer)


class OptimizedCommunicator:
    """
    优化的通信器
    
    整合所有通信优化技术的统一接口。
    
    Attributes:
        compressor: 梯度压缩器
        overlap: 通信计算重叠管理器
        ring_allreduce: Ring All-Reduce实例
        bucket_size: 梯度分桶大小
    
    Example:
        >>> comm = OptimizedCommunicator(
        ...     compression_ratio=0.01,
        ...     overlap_enabled=True
        ... )
        >>> comm.synchronize_gradients(model.parameters())
    """
    
    def __init__(
        self,
        compression_ratio: Optional[float] = None,
        overlap_enabled: bool = True,
        bucket_size: int = 25 * 1024 * 1024,  # 25MB
        use_ring_allreduce: bool = False
    ):
        """
        初始化优化通信器
        
        Args:
            compression_ratio: 梯度压缩比例，None表示不压缩
            overlap_enabled: 是否启用通信计算重叠
            bucket_size: 梯度分桶大小（字节）
            use_ring_allreduce: 是否使用ring all-reduce
        """
        # 初始化梯度压缩器
        if compression_ratio is not None and compression_ratio < 1.0:
            self.compressor = TopKCompressor(compression_ratio)
            logger.info(f"Gradient compression enabled: ratio={compression_ratio}")
        else:
            self.compressor = None
        
        # 初始化通信重叠
        self.overlap = CommunicationOverlap(overlap_enabled)
        
        # 初始化ring all-reduce
        if use_ring_allreduce and dist.is_initialized():
            self.ring_allreduce = RingAllReduce(
                dist.get_world_size(),
                dist.get_rank()
            )
        else:
            self.ring_allreduce = None
        
        self.bucket_size = bucket_size
    
    def synchronize_gradients(
        self,
        parameters: List[torch.nn.Parameter],
        async_op: bool = False
    ) -> Optional[List[torch.cuda.Event]]:
        """
        同步梯度
        
        Args:
            parameters: 要同步的参数列表
            async_op: 是否异步执行
        
        Returns:
            如果异步执行，返回CUDA事件列表
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return None
        
        # 分桶
        buckets = self._create_buckets(parameters)
        
        if async_op and self.overlap.overlap_enabled:
            # 异步执行
            events = []
            for bucket in buckets:
                bucket_events = self.overlap.sync_gradients_async(
                    bucket,
                    self.compressor
                )
                events.extend(bucket_events)
            return events
        else:
            # 同步执行
            for bucket in buckets:
                self.overlap.all_reduce_bucket(bucket, self.compressor)
            return None
    
    def _create_buckets(
        self,
        parameters: List[torch.nn.Parameter]
    ) -> List[List[torch.Tensor]]:
        """
        将参数分桶
        
        Args:
            parameters: 参数列表
        
        Returns:
            桶列表
        """
        buckets = []
        current_bucket = []
        current_size = 0
        
        for param in parameters:
            if param.grad is None:
                continue
            
            param_size = param.grad.numel() * param.grad.element_size()
            
            if current_size + param_size > self.bucket_size and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [param.grad]
                current_size = param_size
            else:
                current_bucket.append(param.grad)
                current_size += param_size
        
        if current_bucket:
            buckets.append(current_bucket)
        
        return buckets
    
    def wait(self, events: Optional[List[torch.cuda.Event]] = None) -> None:
        """
        等待通信完成
        
        Args:
            events: CUDA事件列表
        """
        if events is not None:
            self.overlap.wait_for_communication(events)
        else:
            if dist.is_initialized():
                dist.barrier()
    
    def get_stats(self) -> dict:
        """获取通信统计信息"""
        return {
            'compression_enabled': self.compressor is not None,
            'compression_stats': self.compressor.get_compression_stats() if self.compressor else None,
            'overlap_enabled': self.overlap.overlap_enabled,
            'ring_allreduce_enabled': self.ring_allreduce is not None,
            'bucket_size_mb': self.bucket_size / (1024 ** 2),
        }
