# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : hybrid_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 3D混合并行模块

本模块实现了高性能的3D混合并行策略，支持数据并行、张量并行和流水线并行的组合。

架构设计:
    ┌─────────────────────────────────────────────────────────────┐
    │                  HybridParallel                             │
    ├─────────────────────────────────────────────────────────────┤
    │  Layer 1: Pipeline Parallel (PP)                            │
    │     - 模型按层分割到不同GPU                                  │
    │     - 1F1B调度优化                                          │
    │                                                             │
    │  Layer 2: Tensor Parallel (TP)                              │
    │     - 每层内部张量切分                                       │
    │     - 使用tensor_parallel的实现                             │
    │                                                             │
    │  Layer 3: Data Parallel (DP)                                │
    │     - 数据分割到不同节点                                     │
    │     - 梯度同步                                              │
    └─────────────────────────────────────────────────────────────┘

使用示例:
    >>> from parascale.parallel import HybridParallel, HybridParallelConfig
    >>> 
    >>> # 简单使用
    >>> model = TransformerModel()
    >>> hp = HybridParallel(model, rank=0, world_size=8, dp_size=2, tp_size=2, pp_size=2)
    >>> 
    >>> # 使用配置对象
    >>> config = HybridParallelConfig(dp_size=2, tp_size=2, pp_size=2)
    >>> hp = HybridParallel(model, rank=0, world_size=8, config=config)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

from .base import BaseParallel, ParallelConfigError, ParallelInitError
from .tensor_parallel import (
    TensorParallelConfig,
    TensorParallelConverter,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    ParallelStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 配置和策略
# =============================================================================

class PipelineSchedule(Enum):
    """流水线调度策略"""
    FILL_DRAIN = "fill_drain"           # 传统填充-排空
    ONE_FORWARD_ONE_BACKWARD = "1f1b"   # 1F1B调度
    INTERLEAVED_1F1B = "interleaved"    # 交错1F1B


@dataclass
class HybridParallelConfig:
    """
    3D并行配置类
    
    Attributes:
        dp_size: 数据并行大小
        tp_size: 张量并行大小
        pp_size: 流水线并行大小
        schedule: 流水线调度策略
        num_micro_batches: 微批次数量
        tp_config: 张量并行配置
        enable_overlap: 是否启用通信重叠
    """
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    schedule: PipelineSchedule = field(default_factory=lambda: PipelineSchedule.ONE_FORWARD_ONE_BACKWARD)
    num_micro_batches: int = 4
    tp_config: Optional[TensorParallelConfig] = None
    enable_overlap: bool = True
    
    def __post_init__(self):
        if self.tp_config is None:
            self.tp_config = TensorParallelConfig(
                tp_size=self.tp_size,
                strategy=ParallelStrategy.AUTO,
            )


# =============================================================================
# 流水线通信
# =============================================================================

class PipelineCommunicator:
    """
    流水线并行通信器
    
    处理流水线阶段间的张量发送和接收。
    """
    
    def __init__(
        self,
        pp_group: Optional[dist.ProcessGroup],
        pp_rank: int,
        pp_size: int,
        device: torch.device,
    ):
        self.pp_group = pp_group
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.device = device
    
    def send_forward(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """向前发送张量"""
        if self.pp_group is None:
            return
        
        shape = torch.tensor(tensor.shape, dtype=torch.long, device=self.device)
        shape_len = torch.tensor([len(tensor.shape)], dtype=torch.long, device=self.device)
        
        dist.send(shape_len, dst=dst_rank, group=self.pp_group)
        dist.send(shape, dst=dst_rank, group=self.pp_group)
        dist.send(tensor.contiguous(), dst=dst_rank, group=self.pp_group)
    
    def recv_forward(self, src_rank: int) -> torch.Tensor:
        """从前一阶段接收张量"""
        if self.pp_group is None:
            return torch.zeros(1, device=self.device)
        
        shape_len = torch.zeros(1, dtype=torch.long, device=self.device)
        dist.recv(shape_len, src=src_rank, group=self.pp_group)
        
        shape = torch.zeros(shape_len.item(), dtype=torch.long, device=self.device)
        dist.recv(shape, src=src_rank, group=self.pp_group)
        
        tensor = torch.zeros(*shape.tolist(), device=self.device)
        dist.recv(tensor, src=src_rank, group=self.pp_group)
        
        return tensor
    
    def send_backward(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """向后发送梯度"""
        self.send_forward(tensor, dst_rank)
    
    def recv_backward(self, src_rank: int) -> torch.Tensor:
        """从后一阶段接收梯度"""
        return self.recv_forward(src_rank)


# =============================================================================
# 1F1B 调度器
# =============================================================================

class PipelineScheduler:
    """
    1F1B流水线调度器
    
    实现1F1B (One Forward One Backward) 调度算法，优化流水线bubble。
    """
    
    def __init__(
        self,
        num_stages: int,
        stage_id: int,
        num_micro_batches: int,
        communicator: PipelineCommunicator,
    ):
        self.num_stages = num_stages
        self.stage_id = stage_id
        self.num_micro_batches = num_micro_batches
        self.communicator = communicator
        
        self.forward_outputs: Dict[int, torch.Tensor] = {}
        self.is_first_stage = (stage_id == 0)
        self.is_last_stage = (stage_id == num_stages - 1)
    
    def forward_step(
        self,
        micro_batch_id: int,
        stage_module: nn.Module,
        input_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        执行前向步骤
        
        Args:
            micro_batch_id: 微批次ID
            stage_module: 当前stage的模型
            input_tensor: 输入张量
        
        Returns:
            输出张量
        """
        if not self.is_first_stage and input_tensor is None:
            src_rank = self.stage_id - 1
            input_tensor = self.communicator.recv_forward(src_rank)
        
        output = stage_module(input_tensor)
        self.forward_outputs[micro_batch_id] = output
        
        if not self.is_last_stage:
            dst_rank = self.stage_id + 1
            self.communicator.send_forward(output, dst_rank)
        
        return output
    
    def backward_step(
        self,
        micro_batch_id: int,
        output_grad: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        执行反向步骤
        
        Args:
            micro_batch_id: 微批次ID
            output_grad: 输出梯度
        
        Returns:
            输入梯度
        """
        output = self.forward_outputs.pop(micro_batch_id, None)
        if output is None:
            return None
        
        if not self.is_last_stage and output_grad is None:
            src_rank = self.stage_id + 1
            output_grad = self.communicator.recv_backward(src_rank)
        
        if output_grad is not None:
            output.backward(output_grad)
        else:
            output.backward()
        
        input_grad = None
        
        if not self.is_first_stage and input_grad is not None:
            dst_rank = self.stage_id - 1
            self.communicator.send_backward(input_grad, dst_rank)
        
        return input_grad
    
    def run_schedule(
        self,
        stage_module: nn.Module,
        input_batches: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        执行完整的1F1B调度
        
        Args:
            stage_module: 当前stage的模型
            input_batches: 输入批次列表
        
        Returns:
            输出列表
        """
        outputs = []
        
        # 预热阶段
        num_warmup = min(self.num_stages - self.stage_id - 1, self.num_micro_batches)
        
        for i in range(num_warmup):
            if self.is_first_stage:
                output = self.forward_step(i, stage_module, input_batches[i])
            else:
                output = self.forward_step(i, stage_module, None)
            
            if self.is_last_stage:
                outputs.append(output)
        
        # 1F1B阶段
        num_1f1b = self.num_micro_batches - num_warmup
        
        for i in range(num_1f1b):
            forward_id = num_warmup + i
            if self.is_first_stage:
                output = self.forward_step(forward_id, stage_module, input_batches[forward_id])
            else:
                output = self.forward_step(forward_id, stage_module, None)
            
            if self.is_last_stage:
                outputs.append(output)
            
            backward_id = i
            self.backward_step(backward_id, None)
        
        # 冷却阶段
        for i in range(num_warmup):
            backward_id = num_1f1b + i
            self.backward_step(backward_id, None)
        
        return outputs


# =============================================================================
# 主类
# =============================================================================

class HybridParallel(BaseParallel):
    """
    3D混合并行策略类
    
    实现了数据并行、张量并行和流水线并行的组合，支持1F1B调度。
    
    Args:
        model: PyTorch模型实例
        rank: 当前进程的全局rank
        world_size: 总进程数
        dp_size: 数据并行大小 (默认: 1)
        tp_size: 张量并行大小 (默认: 1)
        pp_size: 流水线并行大小 (默认: 1)
        num_micro_batches: 微批次数量 (默认: 4)
        config: 混合并行配置对象 (如果提供，其他参数将被忽略)
    
    Example:
        >>> from parascale.parallel import HybridParallel
        >>> 
        >>> # 简单使用
        >>> model = TransformerModel()
        >>> hp = HybridParallel(model, rank=0, world_size=8, dp_size=2, tp_size=2, pp_size=2)
        >>> output = hp(inputs)
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        dp_size: int = 1,
        tp_size: int = 1,
        pp_size: int = 1,
        num_micro_batches: int = 4,
        config: Optional[HybridParallelConfig] = None,
    ):
        """
        初始化3D混合并行策略
        
        Args:
            model: PyTorch模型实例
            rank: 当前进程的全局rank
            world_size: 总进程数
            dp_size: 数据并行大小
            tp_size: 张量并行大小
            pp_size: 流水线并行大小
            num_micro_batches: 微批次数量
            config: 配置对象
        """
        super().__init__(model, rank, world_size)
        
        # 使用配置对象或创建新的
        if config is not None:
            self.config = config
        else:
            self.config = HybridParallelConfig(
                dp_size=dp_size,
                tp_size=tp_size,
                pp_size=pp_size,
                num_micro_batches=num_micro_batches,
            )
        
        # 验证配置
        self._validate_config()
        
        # 计算各维度的rank
        self._compute_parallel_ranks()
        
        # 初始化进程组
        self.dp_group = None
        self.tp_group = None
        self.pp_group = None
        self._init_process_groups()
        
        # 流水线相关
        self.is_first_stage = (self.pp_rank == 0)
        self.is_last_stage = (self.pp_rank == self.config.pp_size - 1)
        
        # 分割模型
        self.stage_module: Optional[nn.Module] = None
        self._partition_model()
        
        # 创建流水线调度器
        self.scheduler: Optional[PipelineScheduler] = None
        if self.config.pp_size > 1:
            self.communicator = PipelineCommunicator(
                self.pp_group, self.pp_rank, self.config.pp_size, self.device
            )
            self.scheduler = PipelineScheduler(
                self.config.pp_size,
                self.pp_rank,
                self.config.num_micro_batches,
                self.communicator,
            )
        
        logger.info(
            f"HybridParallel initialized: rank={rank}, "
            f"DP={self.config.dp_size}, TP={self.config.tp_size}, PP={self.config.pp_size}, "
            f"dp_rank={self.dp_rank}, tp_rank={self.tp_rank}, pp_rank={self.pp_rank}"
        )
    
    def _validate_config(self):
        """验证配置"""
        total_size = self.config.dp_size * self.config.tp_size * self.config.pp_size
        if total_size != self.world_size:
            raise ParallelConfigError(
                f"dp_size({self.config.dp_size}) × tp_size({self.config.tp_size}) × "
                f"pp_size({self.config.pp_size}) = {total_size} ≠ world_size({self.world_size})"
            )
        
        if self.config.num_micro_batches < self.config.pp_size:
            logger.warning(
                f"num_micro_batches({self.config.num_micro_batches}) < pp_size({self.config.pp_size}) "
                f"may cause pipeline bubble"
            )
    
    def _compute_parallel_ranks(self):
        """计算各维度的rank"""
        # rank = dp_rank * tp_size * pp_size + tp_rank * pp_size + pp_rank
        self.pp_rank = self.rank % self.config.pp_size
        remaining = self.rank // self.config.pp_size
        self.tp_rank = remaining % self.config.tp_size
        self.dp_rank = remaining // self.config.tp_size
    
    def _init_process_groups(self):
        """初始化所有进程组"""
        if not dist.is_initialized():
            return
        
        self._create_tensor_parallel_groups()
        self._create_pipeline_parallel_groups()
        self._create_data_parallel_groups()
    
    def _create_tensor_parallel_groups(self):
        """创建张量并行组"""
        tp_ranks = []
        for dp in range(self.config.dp_size):
            for pp in range(self.config.pp_size):
                group_ranks = [
                    dp * self.config.tp_size * self.config.pp_size + tp * self.config.pp_size + pp
                    for tp in range(self.config.tp_size)
                ]
                tp_ranks.append(group_ranks)
        
        for ranks in tp_ranks:
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group
                break
    
    def _create_pipeline_parallel_groups(self):
        """创建流水线并行组"""
        pp_ranks = []
        for dp in range(self.config.dp_size):
            for tp in range(self.config.tp_size):
                group_ranks = [
                    dp * self.config.tp_size * self.config.pp_size + tp * self.config.pp_size + pp
                    for pp in range(self.config.pp_size)
                ]
                pp_ranks.append(group_ranks)
        
        for ranks in pp_ranks:
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.pp_group = group
                break
    
    def _create_data_parallel_groups(self):
        """创建数据并行组"""
        dp_ranks = []
        for tp in range(self.config.tp_size):
            for pp in range(self.config.pp_size):
                group_ranks = [
                    dp * self.config.tp_size * self.config.pp_size + tp * self.config.pp_size + pp
                    for dp in range(self.config.dp_size)
                ]
                dp_ranks.append(group_ranks)
        
        for ranks in dp_ranks:
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.dp_group = group
                break
    
    def _partition_model(self):
        """分割模型"""
        if self.config.pp_size == 1:
            self.stage_module = self.model
        else:
            self._pipeline_partition()
        
        # 在当前stage内应用张量并行
        if self.config.tp_size > 1 and self.stage_module is not None:
            self._apply_tensor_parallel_to_stage()
        
        # 将模型移动到设备
        if self.stage_module is not None:
            self.stage_module.to(self.device)
    
    def _pipeline_partition(self):
        """按流水线并行分割模型"""
        all_layers = self._extract_layers()
        
        if len(all_layers) < self.config.pp_size:
            raise ParallelConfigError(
                f"Model has {len(all_layers)} layers, but pp_size is {self.config.pp_size}. "
                f"Consider reducing pp_size or using a larger model."
            )
        
        layers_per_stage = len(all_layers) // self.config.pp_size
        start_idx = layers_per_stage * self.pp_rank
        end_idx = layers_per_stage * (self.pp_rank + 1)
        
        if self.is_last_stage:
            end_idx = len(all_layers)
        
        stage_layers = all_layers[start_idx:end_idx]
        self.stage_module = nn.Sequential(*stage_layers)
    
    def _extract_layers(self) -> List[nn.Module]:
        """从模型中提取所有层"""
        layers: List[nn.Module] = []
        
        if hasattr(self.model, 'layers') and isinstance(self.model.layers, nn.ModuleList):
            layers = list(self.model.layers)
        elif hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
            if hasattr(self.model.encoder, 'layers'):
                layers.extend(list(self.model.encoder.layers))
            if hasattr(self.model.decoder, 'layers'):
                layers.extend(list(self.model.decoder.layers))
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = list(self.model.transformer.h)
        else:
            for name, module in self.model.named_children():
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    layers.extend(list(module.children()))
                elif isinstance(module, nn.Module):
                    layers.append(module)
        
        if not layers:
            layers = [self.model]
        
        return layers
    
    def _apply_tensor_parallel_to_stage(self):
        """在当前stage应用张量并行"""
        self.config.tp_config.tp_size = self.config.tp_size
        self.stage_module = TensorParallelConverter.convert_model(
            self.stage_module,
            self.config.tp_config,
        )
    
    def forward(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        3D混合并行前向传播
        
        Args:
            inputs: 输入数据张量
        
        Returns:
            模型输出张量（仅最后一个流水线阶段返回非None）
        """
        if self.config.pp_size == 1:
            x = self.to_device(inputs)
            return self.stage_module(x)
        else:
            return self._forward_with_pipeline(inputs)
    
    def _forward_with_pipeline(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """使用流水线并行前向传播"""
        micro_batches = self._split_into_micro_batches(inputs)
        outputs = self.scheduler.run_schedule(self.stage_module, micro_batches)
        
        if self.is_last_stage and outputs:
            return torch.cat(outputs, dim=0)
        return None
    
    def _split_into_micro_batches(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """将输入分割为微批次"""
        batch_size = inputs.size(0)
        micro_batch_size = batch_size // self.config.num_micro_batches
        
        micro_batches = []
        for i in range(self.config.num_micro_batches):
            start = i * micro_batch_size
            end = start + micro_batch_size if i < self.config.num_micro_batches - 1 else batch_size
            micro_batches.append(inputs[start:end])
        
        return micro_batches
    
    def gather_gradients(self) -> None:
        """
        收集梯度（数据并行）
        """
        if self.config.dp_size <= 1 or self.dp_group is None:
            return
        
        if self.stage_module is None:
            return
        
        for param in self.stage_module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
                param.grad /= self.config.dp_size
    
    def broadcast_model(self) -> None:
        """
        广播模型参数（数据并行）
        """
        if self.dp_group is None or self.stage_module is None:
            return
        
        for param in self.stage_module.parameters():
            dist.broadcast(param, src=0, group=self.dp_group)
    
    def get_stage_model(self) -> Optional[nn.Module]:
        """
        获取当前流水线阶段的模型
        
        Returns:
            当前阶段的模型
        """
        return self.stage_module
    
    def get_parallel_info(self) -> Dict[str, Any]:
        """
        获取并行配置信息
        
        Returns:
            包含并行配置信息的字典
        """
        info = super().get_parallel_info()
        info.update({
            "dp_size": self.config.dp_size,
            "tp_size": self.config.tp_size,
            "pp_size": self.config.pp_size,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "is_first_stage": self.is_first_stage,
            "is_last_stage": self.is_last_stage,
            "schedule": self.config.schedule.value,
            "num_micro_batches": self.config.num_micro_batches,
        })
        return info
