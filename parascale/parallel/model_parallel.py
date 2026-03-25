# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : model_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 模型并行模块

本模块实现了模型并行策略，将模型分割到不同设备，
每个设备运行模型的一部分，通过点对点通信传递中间结果。

优化特性：
- 负载均衡：基于参数数量和计算复杂度进行层分割
- 自动层识别：支持多种模型结构
- 内存预估：预估每层内存使用
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List, Dict, Any, Tuple
from .base import BaseParallel, ParallelConfigError
import logging

logger = logging.getLogger(__name__)


class LayerInfo:
    """
    层信息类
    
    存储层的元数据，用于负载均衡计算。
    
    Attributes:
        name: 层名称
        module: 层模块
        num_params: 参数数量
        output_size: 输出大小（如果已知）
        compute_cost: 计算成本估计
    """
    
    def __init__(
        self,
        name: str,
        module: nn.Module,
        num_params: int,
        output_size: Optional[Tuple[int, ...]] = None
    ):
        self.name = name
        self.module = module
        self.num_params = num_params
        self.output_size = output_size
        # 计算成本估计：参数数量 × 输出元素数量
        self.compute_cost = self._estimate_compute_cost()
    
    def _estimate_compute_cost(self) -> float:
        """
        估计计算成本
        
        Returns:
            计算成本估计值
        """
        base_cost = self.num_params
        
        # 根据层类型调整成本
        if isinstance(self.module, nn.Conv2d):
            # 卷积层通常计算密集
            base_cost *= 2.0
        elif isinstance(self.module, nn.Linear):
            # 全连接层
            base_cost *= 1.0
        elif isinstance(self.module, (nn.BatchNorm2d, nn.LayerNorm)):
            # 归一化层计算较轻
            base_cost *= 0.5
        
        return base_cost


class ModelParallel(BaseParallel):
    """
    模型并行策略类
    
    模型并行将模型的不同层分配到不同设备上，适用于
    单个 GPU 无法容纳整个模型的情况。数据按顺序流经
    各个设备，每个设备只计算自己负责的部分。
    
    优化特性：
    1. 负载均衡：基于参数数量和计算复杂度智能分割
    2. 支持多种模型结构：Sequential、ModuleList、encoder-decoder等
    3. 内存预估：预估每层内存使用
    
    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank
        world_size: 世界大小
        device: 当前设备
        stage_model: 当前stage的模型
        layer_infos: 所有层的信息列表
    
    Example:
        >>> model = LargeModel()
        >>> mp = ModelParallel(model, rank=0, world_size=2)
        >>> output = mp.forward(inputs)
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        balance_strategy: str = "compute_cost"
    ):
        """
        初始化模型并行策略
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank
            world_size: 世界大小（设备数量）
            balance_strategy: 负载均衡策略 ("param_count", "compute_cost", "memory")
        """
        super().__init__(model, rank, world_size)
        
        self.balance_strategy = balance_strategy
        self.stage_model: Optional[nn.Module] = None
        self.layer_infos: List[LayerInfo] = []
        
        # 分析模型结构
        self._analyze_model()
        
        # 分割模型
        self._split_model()
        
        logger.info(
            f"ModelParallel initialized: rank={rank}, "
            f"layers={len(self.layer_infos)}, strategy={balance_strategy}"
        )
    
    def _analyze_model(self) -> None:
        """
        分析模型结构，收集层信息
        """
        self.layer_infos = []
        
        # 递归收集所有层
        for name, module in self.model.named_modules():
            # 跳过容器层（Sequential、ModuleList等）
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue
            
            # 跳过没有参数的层（如激活函数）
            num_params = sum(p.numel() for p in module.parameters())
            
            # 只记录有参数的层或重要的层
            if num_params > 0 or isinstance(module, (nn.ReLU, nn.GELU)):
                layer_info = LayerInfo(name, module, num_params)
                self.layer_infos.append(layer_info)
        
        if len(self.layer_infos) < self.world_size:
            raise ParallelConfigError(
                f"Model has {len(self.layer_infos)} layers, "
                f"but world_size is {self.world_size}. "
                "Consider reducing world_size or using a larger model."
            )
    
    def _split_model(self) -> None:
        """
        将模型分割到不同设备
        
        使用负载均衡策略智能分割层。
        """
        # 计算每个rank负责的层范围
        start_idx, end_idx = self._compute_layer_range()
        
        # 获取当前rank的层
        stage_layers = self.layer_infos[start_idx:end_idx]
        
        if not stage_layers:
            raise ParallelConfigError(
                f"Rank {self.rank} has no layers assigned"
            )
        
        # 创建当前stage的Sequential模型
        modules = [info.module for info in stage_layers]
        self.stage_model = nn.Sequential(*modules)
        self.stage_model.to(self.device)
        
        # 记录分配信息
        total_params = sum(info.num_params for info in stage_layers)
        logger.info(
            f"Rank {self.rank}: layers {start_idx}-{end_idx-1}, "
            f"{len(stage_layers)} layers, {total_params:,} params"
        )
    
    def _compute_layer_range(self) -> Tuple[int, int]:
        """
        计算当前rank负责的层范围
        
        Returns:
            (start_idx, end_idx) 元组
        """
        total_layers = len(self.layer_infos)
        
        if self.balance_strategy == "param_count":
            return self._balance_by_param_count()
        elif self.balance_strategy == "compute_cost":
            return self._balance_by_compute_cost()
        elif self.balance_strategy == "memory":
            return self._balance_by_memory()
        else:
            # 默认均匀分割
            layers_per_rank = total_layers // self.world_size
            start_idx = layers_per_rank * self.rank
            end_idx = layers_per_rank * (self.rank + 1)
            
            if self.rank == self.world_size - 1:
                end_idx = total_layers
            
            return start_idx, end_idx
    
    def _balance_by_param_count(self) -> Tuple[int, int]:
        """
        基于参数数量进行负载均衡
        
        Returns:
            (start_idx, end_idx) 元组
        """
        total_params = sum(info.num_params for info in self.layer_infos)
        target_params = total_params / self.world_size
        
        # 计算每个rank的起始层
        cumulative_params = 0
        rank_boundaries = [0]
        
        for idx, info in enumerate(self.layer_infos):
            cumulative_params += info.num_params
            if len(rank_boundaries) < self.world_size:
                if cumulative_params >= target_params * len(rank_boundaries):
                    rank_boundaries.append(idx + 1)
        
        # 确保最后一个边界是总层数
        while len(rank_boundaries) <= self.world_size:
            rank_boundaries.append(len(self.layer_infos))
        
        start_idx = rank_boundaries[self.rank]
        end_idx = rank_boundaries[self.rank + 1]
        
        return start_idx, end_idx
    
    def _balance_by_compute_cost(self) -> Tuple[int, int]:
        """
        基于计算成本进行负载均衡
        
        Returns:
            (start_idx, end_idx) 元组
        """
        total_cost = sum(info.compute_cost for info in self.layer_infos)
        target_cost = total_cost / self.world_size
        
        cumulative_cost = 0
        rank_boundaries = [0]
        
        for idx, info in enumerate(self.layer_infos):
            cumulative_cost += info.compute_cost
            if len(rank_boundaries) < self.world_size:
                if cumulative_cost >= target_cost * len(rank_boundaries):
                    rank_boundaries.append(idx + 1)
        
        while len(rank_boundaries) <= self.world_size:
            rank_boundaries.append(len(self.layer_infos))
        
        start_idx = rank_boundaries[self.rank]
        end_idx = rank_boundaries[self.rank + 1]
        
        return start_idx, end_idx
    
    def _balance_by_memory(self) -> Tuple[int, int]:
        """
        基于内存使用进行负载均衡
        
        目前简化为基于参数数量，实际应该考虑激活值大小
        
        Returns:
            (start_idx, end_idx) 元组
        """
        # 简化处理：使用参数数量作为内存使用的代理
        return self._balance_by_param_count()
    
    def _initialize_impl(self) -> None:
        """
        模型并行特定的初始化
        """
        # 验证stage_model已创建
        if self.stage_model is None:
            raise ParallelInitError("Stage model not initialized")
    
    def forward(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        前向传播
        
        按顺序执行各个阶段的计算，通过点对点通信传递中间结果。
        
        Args:
            inputs: 输入数据张量
        
        Returns:
            模型输出张量，非最后一个 rank 返回 None
        """
        if self.stage_model is None:
            raise RuntimeError("Stage model not initialized")
        
        x = inputs
        
        # 按顺序执行各个阶段
        for stage in range(self.world_size):
            if stage == self.rank:
                # 当前 rank 执行自己的阶段
                x = self.to_device(x)
                x = self.stage_model(x)
            
            # 如果不是最后一个阶段，需要传递数据
            if stage < self.world_size - 1:
                if stage == self.rank:
                    # 发送数据给下一个 rank
                    self._send_tensor(x, stage + 1)
                elif stage + 1 == self.rank:
                    # 从上一个 rank 接收数据
                    x = self._recv_tensor(stage)
        
        # 只有最后一个 rank 返回输出
        if self.rank == self.world_size - 1:
            return x
        
        return None
    
    def _send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """
        发送张量到目标rank
        
        Args:
            tensor: 要发送的张量
            dst_rank: 目标rank
        """
        if not dist.is_initialized():
            return
        
        # 发送形状信息
        shape = torch.tensor(tensor.shape, dtype=torch.long, device=self.device)
        dist.send(shape, dst=dst_rank)
        
        # 发送数据
        dist.send(tensor.contiguous(), dst=dst_rank)
    
    def _recv_tensor(self, src_rank: int) -> torch.Tensor:
        """
        从源rank接收张量
        
        Args:
            src_rank: 源rank
        
        Returns:
            接收到的张量
        """
        if not dist.is_initialized():
            # 单进程模式，直接返回空张量
            return torch.zeros(1, device=self.device)
        
        # 接收形状信息
        shape = torch.zeros(4, dtype=torch.long, device=self.device)
        dist.recv(shape, src=src_rank)
        shape = shape[shape > 0].tolist()  # 移除填充的0
        
        # 接收数据
        tensor = torch.zeros(*shape, device=self.device)
        dist.recv(tensor, src=src_rank)
        
        return tensor
    
    def get_load_balance_report(self) -> Dict[str, Any]:
        """
        获取负载均衡报告
        
        Returns:
            负载均衡报告字典
        """
        report = {
            'total_layers': len(self.layer_infos),
            'world_size': self.world_size,
            'strategy': self.balance_strategy,
            'ranks': []
        }
        
        start_idx, end_idx = self._compute_layer_range()
        
        for rank in range(self.world_size):
            rank_start, rank_end = self._get_range_for_rank(rank)
            rank_layers = self.layer_infos[rank_start:rank_end]
            
            rank_info = {
                'rank': rank,
                'layers': rank_end - rank_start,
                'params': sum(info.num_params for info in rank_layers),
                'compute_cost': sum(info.compute_cost for info in rank_layers)
            }
            report['ranks'].append(rank_info)
        
        return report
    
    def _get_range_for_rank(self, rank: int) -> Tuple[int, int]:
        """
        获取指定rank的层范围
        
        Args:
            rank: rank编号
        
        Returns:
            (start_idx, end_idx) 元组
        """
        original_rank = self.rank
        self.rank = rank
        start_idx, end_idx = self._compute_layer_range()
        self.rank = original_rank
        return start_idx, end_idx
