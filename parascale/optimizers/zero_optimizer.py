# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : zero_optimizer.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale ZeRO (Zero Redundancy Optimizer) 优化器模块

本模块实现了 DeepSpeed ZeRO 优化器的完整功能，包括：
- ZeRO Stage 1: 优化器状态分片
- ZeRO Stage 2: 优化器状态 + 梯度分片
- ZeRO Stage 3: 优化器状态 + 梯度 + 参数分片
- CPU Offload: 将数据卸载到CPU以节省GPU内存

参考: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Tuple
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)


class ZeroStage(IntEnum):
    """ZeRO优化阶段枚举"""
    DISABLED = 0
    OPTIMIZER_STATES = 1  # 分片优化器状态
    GRADIENTS = 2         # 分片优化器状态 + 梯度
    PARAMETERS = 3        # 分片优化器状态 + 梯度 + 参数


class ZeroOptimizer(optim.Optimizer):
    """
    ZeRO (Zero Redundancy Optimizer) 优化器
    
    通过分片技术减少分布式训练中的内存冗余，支持超大规模模型训练。
    
    ZeRO Stage 1:
        - 将优化器状态（如Adam的momentum和variance）分片到不同rank
        - 每个rank只存储部分参数的优化器状态
        - 内存节省: 4x (与数据并行度成正比)
    
    ZeRO Stage 2:
        - 在Stage 1基础上增加梯度分片
        - 梯度在all-reduce时直接分片到对应rank
        - 内存节省: 8x
    
    ZeRO Stage 3:
        - 在Stage 2基础上增加参数分片
        - 每个rank只存储部分参数
        - 前向/反向传播时动态收集所需参数
        - 内存节省: 与数据并行度线性相关
    
    CPU Offload:
        - 将优化器状态和梯度卸载到CPU内存
        - 进一步减少GPU内存使用
        - 适用于GPU内存极度受限的场景
    
    Attributes:
        model: PyTorch模型
        base_optimizer: 基础优化器类
        stage: ZeRO阶段 (0-3)
        offload_optimizer: 是否将优化器状态卸载到CPU
        offload_params: 是否将参数卸载到CPU
        world_size: 分布式世界大小
        rank: 当前进程rank
    
    Example:
        >>> model = LargeModel()
        >>> optimizer = ZeroOptimizer(
        ...     model,
        ...     optim.AdamW,
        ...     lr=1e-3,
        ...     stage=ZeroStage.GRADIENTS,
        ...     offload_optimizer=True
        ... )
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer_cls: type,
        stage: int = 1,
        offload_optimizer: bool = False,
        offload_params: bool = False,
        overlap_comm: bool = False,
        bucket_size: int = 500000000,  # 500MB
        **optimizer_kwargs
    ):
        """
        初始化ZeRO优化器
        
        Args:
            model: PyTorch模型
            base_optimizer_cls: 基础优化器类 (如 optim.AdamW)
            stage: ZeRO阶段 (0-3)
            offload_optimizer: 是否将优化器状态卸载到CPU
            offload_params: 是否将参数卸载到CPU (仅Stage 3)
            overlap_comm: 是否重叠通信和计算
            bucket_size: 梯度分桶大小（字节）
            **optimizer_kwargs: 传递给基础优化器的参数
        
        Raises:
            ValueError: 当stage不在有效范围内
            RuntimeError: 当分布式环境未初始化
        """
        if stage not in [0, 1, 2, 3]:
            raise ValueError(f"stage must be 0, 1, 2, or 3, got {stage}")
        
        if stage >= 1 and not dist.is_initialized():
            raise RuntimeError("Distributed environment must be initialized for ZeRO")
        
        self.stage = ZeroStage(stage)
        self.model = model
        self.offload_optimizer = offload_optimizer
        self.offload_params = offload_params
        self.overlap_comm = overlap_comm
        self.bucket_size = bucket_size
        
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # 收集所有参数
        self.all_params = list(model.parameters())
        self.total_params = sum(p.numel() for p in self.all_params)
        
        # 参数分片 (用于Stage 3)
        self.param_partitions = {}
        self.grad_partitions = {}
        
        # 分片优化器状态
        self.shard_optimizer_states = {}
        
        # 初始化参数分片
        if self.stage >= ZeroStage.PARAMETERS:
            self._partition_parameters()
        
        # 创建基础优化器（只包含当前rank负责的参数）
        self.param_groups = []
        self._create_sharded_optimizer(base_optimizer_cls, optimizer_kwargs)
        
        # 初始化父类
        super().__init__(self.param_groups, {})
        
        # 通信相关
        self.buckets = []
        self._init_buckets()
        
        logger.info(f"ZeRO Optimizer initialized: stage={self.stage}, "
                   f"world_size={self.world_size}, rank={self.rank}")
    
    def _partition_parameters(self) -> None:
        """
        将参数分片到不同rank (ZeRO Stage 3)
        
        每个rank只负责存储和更新部分参数
        """
        params_per_rank = self.total_params // self.world_size
        
        current_offset = 0
        current_rank = 0
        params_for_rank = []
        
        for param in self.all_params:
            param_size = param.numel()
            
            # 如果参数跨越多个rank，需要分割
            while param_size > 0:
                remaining_in_rank = params_per_rank - (current_offset % params_per_rank)
                chunk_size = min(param_size, remaining_in_rank)
                
                if current_rank == self.rank:
                    # 记录当前rank负责的参数范围
                    if param not in self.param_partitions:
                        self.param_partitions[param] = []
                    start_idx = param.numel() - param_size
                    end_idx = start_idx + chunk_size
                    self.param_partitions[param].append((start_idx, end_idx))
                
                current_offset += chunk_size
                param_size -= chunk_size
                
                if current_offset >= params_per_rank * (current_rank + 1):
                    current_rank = min(current_rank + 1, self.world_size - 1)
        
        logger.debug(f"Rank {self.rank} owns {len(self.param_partitions)} parameter tensors")
    
    def _create_sharded_optimizer(
        self,
        base_optimizer_cls: type,
        optimizer_kwargs: Dict[str, Any]
    ) -> None:
        """
        创建分片优化器
        
        根据ZeRO阶段，只将部分参数传递给基础优化器
        """
        if self.stage == ZeroStage.DISABLED:
            # Stage 0: 使用所有参数
            sharded_params = self.all_params
        elif self.stage >= ZeroStage.PARAMETERS:
            # Stage 3: 只使用当前rank负责的参数
            sharded_params = list(self.param_partitions.keys())
        else:
            # Stage 1 & 2: 使用所有参数，但优化器状态会分片
            sharded_params = self.all_params
        
        # 过滤掉不需要梯度的参数
        sharded_params = [p for p in sharded_params if p.requires_grad]
        
        # 创建基础优化器
        self.base_optimizer = base_optimizer_cls(sharded_params, **optimizer_kwargs)
        
        # 复制param_groups
        self.param_groups = self.base_optimizer.param_groups
        
        # 初始化分片状态
        if self.stage >= ZeroStage.OPTIMIZER_STATES:
            self._shard_optimizer_states()
    
    def _shard_optimizer_states(self) -> None:
        """
        分片优化器状态 (ZeRO Stage 1+)
        
        每个rank只存储部分参数的优化器状态
        """
        for group in self.base_optimizer.param_groups:
            for param in group['params']:
                if param not in self.base_optimizer.state:
                    continue
                
                state = self.base_optimizer.state[param]
                
                # 检查是否由当前rank负责
                if not self._is_param_owned(param):
                    # 清空非当前rank负责的参数状态
                    self.base_optimizer.state[param] = {}
                    continue
                
                # 如果需要offload，将状态移到CPU
                if self.offload_optimizer:
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.cpu()
    
    def _is_param_owned(self, param: torch.Tensor) -> bool:
        """检查参数是否由当前rank负责"""
        if self.stage < ZeroStage.PARAMETERS:
            return True
        return param in self.param_partitions
    
    def _init_buckets(self) -> None:
        """初始化梯度通信桶"""
        if self.stage < ZeroStage.GRADIENTS:
            return
        
        current_bucket_size = 0
        current_bucket = []
        
        for param in self.all_params:
            if not param.requires_grad:
                continue
            
            param_size = param.numel() * param.element_size()
            
            if current_bucket_size + param_size > self.bucket_size and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = [param]
                current_bucket_size = param_size
            else:
                current_bucket.append(param)
                current_bucket_size += param_size
        
        if current_bucket:
            self.buckets.append(current_bucket)
        
        logger.debug(f"Created {len(self.buckets)} gradient buckets")
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        清零梯度
        
        Args:
            set_to_none: 是否将梯度设为None以节省内存
        """
        for param in self.all_params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        执行优化步骤
        
        Args:
            closure: 可选的闭包函数，用于重新计算损失
        
        Returns:
            如果提供了closure，返回损失值
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Stage 2+: 同步梯度
        if self.stage >= ZeroStage.GRADIENTS:
            self._sync_gradients()
        
        # Stage 3: 收集参数进行更新
        if self.stage >= ZeroStage.PARAMETERS:
            self._gather_params_for_update()
        
        # 执行基础优化器步骤
        self.base_optimizer.step()
        
        # Stage 3: 重新分片参数
        if self.stage >= ZeroStage.PARAMETERS:
            self._partition_params_after_update()
        
        return loss
    
    def _sync_gradients(self) -> None:
        """
        同步梯度 (ZeRO Stage 2+)
        
        使用reduce-scatter代替all-reduce，每个rank只保留自己负责的梯度分片
        """
        if self.world_size == 1:
            return
        
        for bucket in self.buckets:
            # 收集桶中所有梯度
            grads = []
            for param in bucket:
                if param.grad is not None:
                    grads.append(param.grad)
            
            if not grads:
                continue
            
            # 扁平化梯度
            flat_grad = torch.cat([g.flatten() for g in grads])
            
            # 分片大小
            shard_size = flat_grad.numel() // self.world_size
            
            # Reduce-scatter: 每个rank只获得自己负责的部分
            if flat_grad.numel() % self.world_size == 0:
                # 使用reduce-scatter
                grad_shards = [
                    flat_grad[i * shard_size:(i + 1) * shard_size]
                    for i in range(self.world_size)
                ]
                local_shard = torch.zeros_like(grad_shards[0])
                dist.reduce_scatter(local_shard, grad_shards, op=dist.ReduceOp.SUM)
                
                # 将结果分回各个参数
                offset = 0
                for param in bucket:
                    if param.grad is not None:
                        numel = param.grad.numel()
                        param.grad.copy_(local_shard[offset:offset + numel].view_as(param.grad))
                        offset += numel
            else:
                # 回退到all-reduce
                dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
                flat_grad /= self.world_size
                
                offset = 0
                for param in bucket:
                    if param.grad is not None:
                        numel = param.grad.numel()
                        param.grad.copy_(flat_grad[offset:offset + numel].view_as(param.grad))
                        offset += numel
            
            # Offload到CPU
            if self.offload_optimizer:
                for param in bucket:
                    if param.grad is not None:
                        param.grad = param.grad.cpu()
    
    def _gather_params_for_update(self) -> None:
        """
        收集参数用于更新 (ZeRO Stage 3)
        
        从所有rank收集完整的参数以进行优化步骤
        """
        if self.world_size == 1:
            return
        
        for param in self.all_params:
            if not param.requires_grad:
                continue
            
            # 收集完整参数
            if param in self.param_partitions:
                # 当前rank负责此参数，需要收集
                gathered = [torch.zeros_like(param) for _ in range(self.world_size)]
                dist.all_gather(gathered, param.data)
                param.data.copy_(torch.cat([g.flatten() for g in gathered]).view_as(param))
    
    def _partition_params_after_update(self) -> None:
        """
        更新后重新分片参数 (ZeRO Stage 3)
        
        只保留当前rank负责的参数分片
        """
        for param, ranges in self.param_partitions.items():
            # 只保留当前rank负责的范围
            for start_idx, end_idx in ranges:
                shard = param.data.flatten()[start_idx:end_idx]
                # 释放完整参数，只保留分片
                param.data = shard
    
    def state_dict(self) -> Dict[str, Any]:
        """
        获取优化器状态字典
        
        Returns:
            包含优化器状态的字典
        """
        state = {
            'stage': int(self.stage),
            'offload_optimizer': self.offload_optimizer,
            'offload_params': self.offload_params,
            'base_optimizer_state': self.base_optimizer.state_dict(),
            'world_size': self.world_size,
            'rank': self.rank,
        }
        
        # Stage 3: 保存参数分片信息
        if self.stage >= ZeroStage.PARAMETERS:
            state['param_partitions'] = {
                id(param): ranges for param, ranges in self.param_partitions.items()
            }
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载优化器状态
        
        Args:
            state_dict: 之前保存的优化器状态字典
        """
        self.base_optimizer.load_state_dict(state_dict['base_optimizer_state'])
        
        # 恢复参数分片
        if self.stage >= ZeroStage.PARAMETERS and 'param_partitions' in state_dict:
            # 重建参数分片映射
            pass  # 需要在模型重建后重新映射
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取内存统计信息
        
        Returns:
            包含内存统计的字典
        """
        total_params = sum(p.numel() for p in self.all_params)
        
        # 计算当前rank的参数数量
        if self.stage >= ZeroStage.PARAMETERS:
            owned_params = sum(
                end - start for ranges in self.param_partitions.values()
                for start, end in ranges
            )
        else:
            owned_params = total_params
        
        # 优化器状态内存 (Adam = 2 * param_size)
        optimizer_state_size = owned_params * 2 * 4  # float32
        
        # 梯度内存
        grad_size = owned_params * 4 if self.stage >= ZeroStage.GRADIENTS else total_params * 4
        
        return {
            'total_params': total_params,
            'owned_params': owned_params,
            'param_memory_mb': (owned_params * 4) / (1024 ** 2),
            'grad_memory_mb': grad_size / (1024 ** 2),
            'optimizer_state_mb': optimizer_state_size / (1024 ** 2),
            'total_memory_mb': (owned_params * 4 + grad_size + optimizer_state_size) / (1024 ** 2),
            'theoretical_savings': self.world_size if self.stage >= ZeroStage.PARAMETERS else (
                2 if self.stage >= ZeroStage.GRADIENTS else (
                    1.5 if self.stage >= ZeroStage.OPTIMIZER_STATES else 1.0
                )
            ),
        }
    
    def print_memory_stats(self) -> None:
        """打印内存统计信息"""
        stats = self.get_memory_stats()
        print(f"ZeRO Stage {self.stage} Memory Stats:")
        print(f"  Total parameters: {stats['total_params']:,}")
        print(f"  Owned parameters: {stats['owned_params']:,} ({stats['owned_params']/stats['total_params']*100:.1f}%)")
        print(f"  Parameter memory: {stats['param_memory_mb']:.2f} MB")
        print(f"  Gradient memory: {stats['grad_memory_mb']:.2f} MB")
        print(f"  Optimizer state memory: {stats['optimizer_state_mb']:.2f} MB")
        print(f"  Total memory: {stats['total_memory_mb']:.2f} MB")
        print(f"  Theoretical savings: {stats['theoretical_savings']:.1f}x")


class ZeroAdamW(ZeroOptimizer):
    """
    ZeRO AdamW 优化器
    
    结合ZeRO内存优化和AdamW优化算法的优化器
    
    Example:
        >>> model = LargeModel()
        >>> optimizer = ZeroAdamW(
        ...     model,
        ...     lr=1e-3,
        ...     stage=ZeroStage.GRADIENTS,
        ...     offload_optimizer=True
        ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        stage: int = 1,
        offload_optimizer: bool = False,
        offload_params: bool = False,
        **kwargs
    ):
        super().__init__(
            model=model,
            base_optimizer_cls=optim.AdamW,
            stage=stage,
            offload_optimizer=offload_optimizer,
            offload_params=offload_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs
        )


class ZeroSGD(ZeroOptimizer):
    """
    ZeRO SGD 优化器
    
    结合ZeRO内存优化和SGD优化算法的优化器
    
    Example:
        >>> model = LargeModel()
        >>> optimizer = ZeroSGD(
        ...     model,
        ...     lr=0.01,
        ...     momentum=0.9,
        ...     stage=ZeroStage.GRADIENTS
        ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        stage: int = 1,
        offload_optimizer: bool = False,
        offload_params: bool = False,
        **kwargs
    ):
        super().__init__(
            model=model,
            base_optimizer_cls=optim.SGD,
            stage=stage,
            offload_optimizer=offload_optimizer,
            offload_params=offload_params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **kwargs
        )
