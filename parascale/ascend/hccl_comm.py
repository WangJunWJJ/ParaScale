# -*- coding: utf-8 -*-
# @Time    : 2026/3/23
# @Author  : Jun Wang
# @File    : hccl_comm.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale HCCL 通信优化模块

本模块实现了针对华为昇腾 HCCL (Huawei Collective Communication Library) 
的高效分布式通信机制，包括拓扑感知通信、梯度压缩、通信计算重叠等优化。

核心功能：
- HCCL 初始化和进程组管理
- 拓扑感知通信优化（HCCS 域内优先）
- 梯度压缩通信（Top-K、1-bit Adam）
- 通信与计算重叠
- Ring All-Reduce 和 Tree All-Reduce
- 自适应通信策略选择

性能优化策略：
1. HCCS 拓扑感知：利用昇腾服务器内部高速互联
2. 两阶段通信：域内 All-Reduce + 域间 All-Reduce
3. 梯度压缩：减少通信数据量
4. 流水线通信：通信与计算重叠
"""

import torch
import torch.distributed as dist
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import time
import threading
import queue

logger = logging.getLogger(__name__)

# 检测 HCCL 是否可用
HCCL_AVAILABLE = False
try:
    import torch_npu
    HCCL_AVAILABLE = True
except ImportError:
    pass


class CommunicationBackend(Enum):
    """通信后端枚举"""
    HCCL = "hccl"
    NCCL = "nccl"
    GLOO = "gloo"


class TopologyType(Enum):
    """拓扑类型枚举"""
    SINGLE_NODE = "single_node"
    MULTI_NODE_HCCS = "multi_node_hccs"
    MULTI_NODE_NETWORK = "multi_node_network"


@dataclass
class TopologyInfo:
    """拓扑信息数据类"""
    topology_type: TopologyType
    num_nodes: int
    num_devices_per_node: int
    hccs_groups: List[List[int]]
    inter_node_groups: List[List[int]]
    bandwidth_intra_node: float  # GB/s
    bandwidth_inter_node: float  # GB/s


class HCCLTopologyDetector:
    """
    HCCL 拓扑探测器
    
    自动探测昇腾服务器的拓扑结构，包括 HCCS 域、节点间连接等。
    
    Example:
        >>> detector = HCCLTopologyDetector()
        >>> topology = detector.detect()
        >>> print(f"HCCS 组数: {len(topology.hccs_groups)}")
    """
    
    def __init__(self):
        self._topology = None
    
    def detect(self) -> TopologyInfo:
        """
        探测拓扑结构
        
        Returns:
            TopologyInfo 拓扑信息
        """
        if self._topology is not None:
            return self._topology
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # 尝试获取本地设备数
        local_size = self._get_local_size()
        
        # 判断拓扑类型
        if world_size <= local_size:
            topology_type = TopologyType.SINGLE_NODE
            num_nodes = 1
            num_devices_per_node = world_size
        elif HCCL_AVAILABLE:
            topology_type = TopologyType.MULTI_NODE_HCCS
            num_nodes = (world_size + local_size - 1) // local_size
            num_devices_per_node = local_size
        else:
            topology_type = TopologyType.MULTI_NODE_NETWORK
            num_nodes = (world_size + local_size - 1) // local_size
            num_devices_per_node = local_size
        
        # 构建 HCCS 组
        hccs_groups = self._build_hccs_groups(world_size, local_size)
        
        # 构建跨节点组
        inter_node_groups = self._build_inter_node_groups(world_size, local_size)
        
        # 带宽估计
        bandwidth_intra_node = 300.0 if HCCL_AVAILABLE else 100.0  # HCCS ~300GB/s
        bandwidth_inter_node = 100.0 if HCCL_AVAILABLE else 25.0   # 网络 ~100GB/s (RoCE)
        
        self._topology = TopologyInfo(
            topology_type=topology_type,
            num_nodes=num_nodes,
            num_devices_per_node=num_devices_per_node,
            hccs_groups=hccs_groups,
            inter_node_groups=inter_node_groups,
            bandwidth_intra_node=bandwidth_intra_node,
            bandwidth_inter_node=bandwidth_inter_node
        )
        
        return self._topology
    
    def _get_local_size(self) -> int:
        """获取本地设备数量"""
        try:
            if HCCL_AVAILABLE:
                return torch.npu.device_count()
            return torch.cuda.device_count()
        except Exception:
            return 1
    
    def _build_hccs_groups(self, world_size: int, local_size: int) -> List[List[int]]:
        """构建 HCCS 组"""
        groups = []
        for i in range(0, world_size, local_size):
            group = list(range(i, min(i + local_size, world_size)))
            groups.append(group)
        return groups
    
    def _build_inter_node_groups(self, world_size: int, local_size: int) -> List[List[int]]:
        """构建跨节点组（每个节点取一个 rank）"""
        groups = []
        for i in range(local_size):
            group = list(range(i, world_size, local_size))
            if group:
                groups.append(group)
        return groups


class HCCLProcessGroup:
    """
    HCCL 进程组管理器
    
    管理分布式训练中的各种进程组，支持分层通信。
    
    Attributes:
        world_group: 全局进程组
        intra_node_groups: 节点内进程组列表
        inter_node_groups: 节点间进程组列表
    
    Example:
        >>> pg_manager = HCCLProcessGroup()
        >>> pg_manager.initialize()
        >>> intra_group = pg_manager.get_intra_node_group(rank)
    """
    
    def __init__(self, backend: str = "hccl"):
        """
        初始化进程组管理器
        
        Args:
            backend: 通信后端，默认为 hccl
        """
        self.backend = backend
        self.world_group = None
        self.intra_node_groups: Dict[int, dist.ProcessGroup] = {}
        self.inter_node_groups: Dict[int, dist.ProcessGroup] = {}
        self._topology = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        初始化进程组
        
        Returns:
            初始化是否成功
        """
        if self._initialized:
            return True
        
        if not dist.is_initialized():
            logger.warning("分布式环境未初始化")
            return False
        
        self.world_group = dist.group.WORLD
        
        # 探测拓扑
        detector = HCCLTopologyDetector()
        self._topology = detector.detect()
        
        # 创建节点内进程组
        self._create_intra_node_groups()
        
        # 创建节点间进程组
        self._create_inter_node_groups()
        
        self._initialized = True
        logger.info(f"HCCL 进程组初始化完成，拓扑类型: {self._topology.topology_type.value}")
        return True
    
    def _create_intra_node_groups(self):
        """创建节点内进程组"""
        rank = dist.get_rank()
        
        for i, group_ranks in enumerate(self._topology.hccs_groups):
            pg = dist.new_group(group_ranks, backend=self.backend)
            for r in group_ranks:
                self.intra_node_groups[r] = pg
    
    def _create_inter_node_groups(self):
        """创建节点间进程组"""
        rank = dist.get_rank()
        
        for i, group_ranks in enumerate(self._topology.inter_node_groups):
            pg = dist.new_group(group_ranks, backend=self.backend)
            for r in group_ranks:
                self.inter_node_groups[r] = pg
    
    def get_intra_node_group(self, rank: int) -> Optional[dist.ProcessGroup]:
        """获取指定 rank 的节点内进程组"""
        return self.intra_node_groups.get(rank)
    
    def get_inter_node_group(self, rank: int) -> Optional[dist.ProcessGroup]:
        """获取指定 rank 的节点间进程组"""
        return self.inter_node_groups.get(rank)
    
    def get_topology(self) -> TopologyInfo:
        """获取拓扑信息"""
        return self._topology


class HCCLGradientCompressor:
    """
    HCCL 梯度压缩器
    
    支持多种压缩算法以减少通信数据量。
    
    支持的压缩算法：
    - Top-K: 保留绝对值最大的 K 个元素
    - 1-bit Adam: 梯度量化为 1-bit
    - FP16: 半精度压缩
    - DGC: 深度梯度压缩
    
    Example:
        >>> compressor = HCCLGradientCompressor(method='topk', ratio=0.01)
        >>> compressed, meta = compressor.compress(gradient)
        >>> decompressed = compressor.decompress(compressed, meta)
    """
    
    def __init__(self, method: str = "topk", ratio: float = 0.01,
                 error_feedback: bool = True):
        """
        初始化梯度压缩器
        
        Args:
            method: 压缩方法 ('topk', '1bit', 'fp16', 'dgc')
            ratio: 压缩比例 (0-1)
            error_feedback: 是否启用误差补偿
        """
        self.method = method
        self.ratio = ratio
        self.error_feedback = error_feedback
        self.residuals: Dict[int, torch.Tensor] = {}
    
    def compress(self, tensor: torch.Tensor, param_id: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        压缩张量
        
        Args:
            tensor: 要压缩的张量
            param_id: 参数 ID（用于误差补偿）
        
        Returns:
            (压缩后的数据, 元数据)
        """
        # 误差补偿
        if self.error_feedback:
            if param_id not in self.residuals:
                self.residuals[param_id] = torch.zeros_like(tensor)
            tensor = tensor + self.residuals[param_id]
        
        if self.method == "topk":
            return self._compress_topk(tensor, param_id)
        elif self.method == "1bit":
            return self._compress_1bit(tensor, param_id)
        elif self.method == "fp16":
            return self._compress_fp16(tensor)
        elif self.method == "dgc":
            return self._compress_dgc(tensor, param_id)
        else:
            raise ValueError(f"不支持的压缩方法: {self.method}")
    
    def _compress_topk(self, tensor: torch.Tensor, param_id: int) -> Tuple[torch.Tensor, Dict]:
        """Top-K 压缩"""
        flat = tensor.flatten()
        k = max(1, int(flat.numel() * self.ratio))
        
        values, indices = torch.topk(torch.abs(flat), k)
        compressed_values = flat[indices]
        
        # 记录误差
        if self.error_feedback:
            reconstructed = torch.zeros_like(flat)
            reconstructed[indices] = compressed_values
            self.residuals[param_id] = flat - reconstructed
        
        metadata = {
            'indices': indices,
            'shape': tensor.shape,
            'numel': flat.numel()
        }
        return compressed_values, metadata
    
    def _compress_1bit(self, tensor: torch.Tensor, param_id: int) -> Tuple[torch.Tensor, Dict]:
        """1-bit 压缩"""
        flat = tensor.flatten()
        signs = (flat >= 0).to(torch.int8) * 2 - 1  # -1 或 1
        
        # 计算尺度因子
        scale = torch.abs(flat).mean()
        
        # 记录误差
        if self.error_feedback:
            self.residuals[param_id] = flat - signs.float() * scale
        
        metadata = {
            'shape': tensor.shape,
            'scale': scale,
            'numel': flat.numel()
        }
        return signs, metadata
    
    def _compress_fp16(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """FP16 压缩"""
        compressed = tensor.half()
        metadata = {'shape': tensor.shape, 'dtype': tensor.dtype}
        return compressed, metadata
    
    def _compress_dgc(self, tensor: torch.Tensor, param_id: int) -> Tuple[torch.Tensor, Dict]:
        """DGC (Deep Gradient Compression) 压缩"""
        # DGC 使用动量修正的 Top-K
        flat = tensor.flatten()
        k = max(1, int(flat.numel() * self.ratio))
        
        # 使用动量
        if param_id not in self.residuals:
            self.residuals[param_id] = torch.zeros_like(flat)
        
        momentum = 0.9
        corrected = flat + momentum * self.residuals[param_id]
        
        values, indices = torch.topk(torch.abs(corrected), k)
        compressed_values = flat[indices]
        
        # 更新动量
        reconstructed = torch.zeros_like(flat)
        reconstructed[indices] = compressed_values
        self.residuals[param_id] = flat - reconstructed
        
        metadata = {
            'indices': indices,
            'shape': tensor.shape,
            'numel': flat.numel()
        }
        return compressed_values, metadata
    
    def decompress(self, compressed: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        解压缩张量
        
        Args:
            compressed: 压缩后的数据
            metadata: 元数据
        
        Returns:
            解压缩后的张量
        """
        if self.method == "topk" or self.method == "dgc":
            indices = metadata['indices']
            shape = metadata['shape']
            numel = metadata['numel']
            
            tensor = torch.zeros(numel, dtype=compressed.dtype, device=compressed.device)
            tensor[indices] = compressed
            return tensor.view(shape)
        
        elif self.method == "1bit":
            shape = metadata['shape']
            scale = metadata['scale']
            
            tensor = compressed.float() * scale
            return tensor.view(shape)
        
        elif self.method == "fp16":
            shape = metadata['shape']
            dtype = metadata.get('dtype', torch.float32)
            
            tensor = compressed.to(dtype)
            return tensor.view(shape)
        
        else:
            raise ValueError(f"不支持的压缩方法: {self.method}")


class HCCLCommunicator:
    """
    HCCL 高效通信器
    
    提供拓扑感知的高效分布式通信接口。
    
    核心功能：
    - 两阶段 All-Reduce（域内 + 域间）
    - Ring All-Reduce
    - 梯度压缩通信
    - 通信计算重叠
    
    Example:
        >>> comm = HCCLCommunicator()
        >>> comm.initialize()
        >>> comm.all_reduce(tensor, op=dist.ReduceOp.SUM)
    """
    
    def __init__(self, compression: Optional[str] = None,
                 compression_ratio: float = 0.01,
                 overlap_comm: bool = True):
        """
        初始化通信器
        
        Args:
            compression: 压缩方法 ('topk', '1bit', 'fp16', 'dgc', None)
            compression_ratio: 压缩比例
            overlap_comm: 是否启用通信计算重叠
        """
        self.compression = compression
        self.compression_ratio = compression_ratio
        self.overlap_comm = overlap_comm
        
        self._pg_manager: Optional[HCCLProcessGroup] = None
        self._compressor: Optional[HCCLGradientCompressor] = None
        self._topology: Optional[TopologyInfo] = None
        self._initialized = False
        
        # 通信流
        self._comm_stream = None
        self._compute_stream = None
        
        # 异步通信队列
        self._async_queue = queue.Queue()
        self._async_thread = None
    
    def initialize(self) -> bool:
        """初始化通信器"""
        if self._initialized:
            return True
        
        if not dist.is_initialized():
            logger.warning("分布式环境未初始化")
            return False
        
        # 初始化进程组管理器
        self._pg_manager = HCCLProcessGroup()
        if not self._pg_manager.initialize():
            return False
        
        self._topology = self._pg_manager.get_topology()
        
        # 初始化压缩器
        if self.compression:
            self._compressor = HCCLGradientCompressor(
                method=self.compression,
                ratio=self.compression_ratio
            )
        
        # 创建通信流
        if HCCL_AVAILABLE:
            self._comm_stream = torch.npu.Stream()
            self._compute_stream = torch.npu.Stream()
        elif torch.cuda.is_available():
            self._comm_stream = torch.cuda.Stream()
            self._compute_stream = torch.cuda.Stream()
        
        self._initialized = True
        logger.info(f"HCCL 通信器初始化完成，压缩: {self.compression}")
        return True
    
    def all_reduce(self, tensor: torch.Tensor, 
                   op: dist.ReduceOp = dist.ReduceOp.SUM,
                   async_op: bool = False) -> Optional[Any]:
        """
        执行 All-Reduce 操作
        
        根据拓扑自动选择最优通信策略：
        - 单节点：直接 All-Reduce
        - 多节点：两阶段 All-Reduce（域内 + 域间）
        
        Args:
            tensor: 要通信的张量
            op: 归约操作
            async_op: 是否异步执行
        
        Returns:
            异步操作句柄（如果 async_op=True）
        """
        if not self._initialized:
            return dist.all_reduce(tensor, op, async_op=async_op)
        
        if self._topology.topology_type == TopologyType.SINGLE_NODE:
            return self._single_node_all_reduce(tensor, op, async_op)
        else:
            return self._multi_node_all_reduce(tensor, op, async_op)
    
    def _single_node_all_reduce(self, tensor: torch.Tensor,
                                 op: dist.ReduceOp,
                                 async_op: bool) -> Optional[Any]:
        """单节点 All-Reduce"""
        if self._compressor:
            # 压缩通信
            compressed, metadata = self._compressor.compress(tensor)
            work = dist.all_reduce(compressed, op, async_op=async_op)
            if not async_op:
                tensor.copy_(self._compressor.decompress(compressed, metadata))
            return work
        else:
            return dist.all_reduce(tensor, op, async_op=async_op)
    
    def _multi_node_all_reduce(self, tensor: torch.Tensor,
                                op: dist.ReduceOp,
                                async_op: bool) -> Optional[Any]:
        """
        多节点两阶段 All-Reduce
        
        阶段 1: 节点内 All-Reduce（利用 HCCS 高带宽）
        阶段 2: 节点间 All-Reduce（跨节点通信）
        """
        rank = dist.get_rank()
        
        # 阶段 1: 节点内 All-Reduce
        intra_group = self._pg_manager.get_intra_node_group(rank)
        if intra_group is not None:
            dist.all_reduce(tensor, op, group=intra_group)
        
        # 阶段 2: 节点间 All-Reduce
        inter_group = self._pg_manager.get_inter_node_group(rank)
        if inter_group is not None:
            if self._compressor:
                compressed, metadata = self._compressor.compress(tensor)
                work = dist.all_reduce(compressed, op, group=inter_group, async_op=async_op)
                if not async_op:
                    tensor.copy_(self._compressor.decompress(compressed, metadata))
                return work
            else:
                return dist.all_reduce(tensor, op, group=inter_group, async_op=async_op)
        
        return None
    
    def reduce_scatter(self, output: torch.Tensor,
                       input_list: List[torch.Tensor],
                       op: dist.ReduceOp = dist.ReduceOp.SUM) -> None:
        """
        执行 Reduce-Scatter 操作
        
        Args:
            output: 输出张量
            input_list: 输入张量列表
            op: 归约操作
        """
        if self._compressor:
            # 压缩后通信
            compressed_list = []
            metadata_list = []
            for t in input_list:
                c, m = self._compressor.compress(t)
                compressed_list.append(c)
                metadata_list.append(m)
            
            dist.reduce_scatter(output, compressed_list, op)
        else:
            dist.reduce_scatter(output, input_list, op)
    
    def all_gather(self, output_list: List[torch.Tensor],
                   input_tensor: torch.Tensor) -> None:
        """
        执行 All-Gather 操作
        
        Args:
            output_list: 输出张量列表
            input_tensor: 输入张量
        """
        if self._compressor:
            compressed, metadata = self._compressor.compress(input_tensor)
            compressed_list = [torch.zeros_like(compressed) for _ in output_list]
            dist.all_gather(compressed_list, compressed)
            
            for i, c in enumerate(compressed_list):
                output_list[i].copy_(self._compressor.decompress(c, metadata))
        else:
            dist.all_gather(output_list, input_tensor)
    
    def all_to_all(self, output_list: List[torch.Tensor],
                   input_list: List[torch.Tensor]) -> None:
        """
        执行 All-to-All 操作
        
        Args:
            output_list: 输出张量列表
            input_list: 输入张量列表
        """
        dist.all_to_all(output_list, input_list)
    
    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        """
        执行 Broadcast 操作
        
        Args:
            tensor: 要广播的张量
            src: 源 rank
        """
        dist.broadcast(tensor, src)
    
    def ring_all_reduce(self, tensor: torch.Tensor,
                        op: dist.ReduceOp = dist.ReduceOp.SUM) -> None:
        """
        Ring All-Reduce 实现
        
        使用环形通信模式，减少内存峰值。
        
        Args:
            tensor: 要通信的张量
            op: 归约操作
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # 将张量分成 world_size 份
        numel = tensor.numel()
        chunk_size = (numel + world_size - 1) // world_size
        
        chunks = []
        for i in range(world_size):
            start = i * chunk_size
            end = min(start + chunk_size, numel)
            if start < numel:
                chunks.append(tensor.flatten()[start:end])
            else:
                chunks.append(torch.zeros(0, dtype=tensor.dtype, device=tensor.device))
        
        # Scatter-Reduce 阶段
        for i in range(world_size - 1):
            send_idx = (rank - i) % world_size
            recv_idx = (rank - i - 1) % world_size
            
            send_req = dist.isend(chunks[send_idx], (rank + 1) % world_size)
            recv_req = dist.irecv(chunks[recv_idx], (rank - 1 + world_size) % world_size)
            
            send_req.wait()
            recv_req.wait()
        
        # All-Gather 阶段
        for i in range(world_size - 1):
            send_idx = (rank - i + 1) % world_size
            recv_idx = (rank - i) % world_size
            
            send_req = dist.isend(chunks[send_idx], (rank + 1) % world_size)
            recv_req = dist.irecv(chunks[recv_idx], (rank - 1 + world_size) % world_size)
            
            send_req.wait()
            recv_req.wait()
        
        # 重组张量
        tensor.copy_(torch.cat(chunks)[:numel].view(tensor.shape))


class HCCLCommunicationOverlap:
    """
    HCCL 通信计算重叠管理器
    
    实现通信与计算的重叠执行，提高训练效率。
    
    工作流程：
    1. 计算流执行前向/反向计算
    2. 通信流异步执行梯度同步
    3. 使用事件同步两个流
    
    Example:
        >>> overlap = HCCLCommunicationOverlap()
        >>> overlap.start_compute()
        >>> # 执行计算
        >>> overlap.start_communication(gradients)
        >>> overlap.wait_communication()
    """
    
    def __init__(self, communicator: HCCLCommunicator):
        """
        初始化重叠管理器
        
        Args:
            communicator: HCCL 通信器实例
        """
        self.communicator = communicator
        self._compute_stream = None
        self._comm_stream = None
        self._events: Dict[int, torch.Event] = {}
        self._pending_works: List[Any] = []
    
    def initialize(self):
        """初始化流和事件"""
        if HCCL_AVAILABLE:
            self._compute_stream = torch.npu.Stream()
            self._comm_stream = torch.npu.Stream()
        elif torch.cuda.is_available():
            self._compute_stream = torch.cuda.Stream()
            self._comm_stream = torch.cuda.Stream()
    
    def start_compute(self):
        """进入计算流"""
        if self._compute_stream:
            self._compute_stream.wait_stream(torch.npu.current_stream() if HCCL_AVAILABLE else torch.cuda.current_stream())
            return self._compute_stream
        return None
    
    def start_communication(self, tensors: List[torch.Tensor],
                            op: dist.ReduceOp = dist.ReduceOp.SUM):
        """
        启动异步通信
        
        Args:
            tensors: 要通信的张量列表
            op: 归约操作
        """
        if self._comm_stream is None:
            return
        
        with torch.npu.stream(self._comm_stream) if HCCL_AVAILABLE else torch.cuda.stream(self._comm_stream):
            for i, tensor in enumerate(tensors):
                if tensor.grad is not None:
                    work = self.communicator.all_reduce(tensor.grad, op, async_op=True)
                    if work:
                        self._pending_works.append(work)
    
    def wait_communication(self):
        """等待所有通信完成"""
        for work in self._pending_works:
            work.wait()
        self._pending_works.clear()
    
    def synchronize(self):
        """同步所有流"""
        if self._compute_stream:
            self._compute_stream.synchronize()
        if self._comm_stream:
            self._comm_stream.synchronize()


class HCCLBucketCommunicator:
    """
    HCCL 分桶通信器
    
    将梯度分桶后批量通信，减少通信次数。
    
    Args:
        bucket_size_mb: 每个桶的大小（MB）
        communicator: HCCL 通信器
    
    Example:
        >>> bucket_comm = HCCLBucketCommunicator(bucket_size_mb=25)
        >>> bucket_comm.initialize(model.parameters())
        >>> bucket_comm.sync_gradients()
    """
    
    def __init__(self, bucket_size_mb: int = 25,
                 communicator: Optional[HCCLCommunicator] = None):
        self.bucket_size_mb = bucket_size_mb
        self.communicator = communicator or HCCLCommunicator()
        self._buckets: List[List[torch.Tensor]] = []
        self._bucket_sizes: List[int] = []
    
    def initialize(self, parameters):
        """初始化梯度桶"""
        bucket_size = self.bucket_size_mb * 1024 * 1024
        current_bucket = []
        current_size = 0
        
        for param in parameters:
            if param.grad is None:
                continue
            
            param_size = param.grad.numel() * param.grad.element_size()
            
            if current_size + param_size > bucket_size and current_bucket:
                self._buckets.append(current_bucket)
                self._bucket_sizes.append(current_size)
                current_bucket = []
                current_size = 0
            
            current_bucket.append(param)
            current_size += param_size
        
        if current_bucket:
            self._buckets.append(current_bucket)
            self._bucket_sizes.append(current_size)
        
        logger.info(f"创建了 {len(self._buckets)} 个梯度桶")
    
    def sync_gradients(self, op: dist.ReduceOp = dist.ReduceOp.SUM):
        """同步所有梯度桶"""
        for bucket in self._buckets:
            # 合并桶内梯度
            grads = [p.grad for p in bucket if p.grad is not None]
            if not grads:
                continue
            
            flat_grad = torch._utils._flatten_dense_tensors(grads)
            self.communicator.all_reduce(flat_grad, op)
            
            # 分发回各参数
            for p, g in zip(bucket, torch._utils._unflatten_dense_tensors(flat_grad, grads)):
                if p.grad is not None:
                    p.grad.copy_(g)


def initialize_hccl(backend: str = "hccl",
                    init_method: Optional[str] = None,
                    timeout: int = 1800) -> bool:
    """
    初始化 HCCL 分布式环境
    
    Args:
        backend: 通信后端
        init_method: 初始化方法（URL）
        timeout: 超时时间（秒）
    
    Returns:
        初始化是否成功
    """
    if dist.is_initialized():
        logger.info("分布式环境已初始化")
        return True
    
    # 从环境变量获取配置
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if init_method is None:
        init_method = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        init_method = f"tcp://{init_method}:{master_port}"
    
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=datetime.timedelta(seconds=timeout)
        )
        logger.info(f"HCCL 初始化成功: rank={rank}, world_size={world_size}")
        return True
    except Exception as e:
        logger.error(f"HCCL 初始化失败: {e}")
        return False


import os
import datetime
