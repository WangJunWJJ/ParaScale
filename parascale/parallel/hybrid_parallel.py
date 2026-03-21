# -*- coding: utf-8 -*-
# @Time    : 2026/3/9 上午10:00
# @Author  : Jun Wang
# @File    : hybrid_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 3D混合并行模块

本模块实现了3D混合并行策略，将数据并行、张量并行和流水线并行组合在一起。
3D并行可以在大规模分布式训练中实现更高的效率和可扩展性。

3D并行架构:
- 数据并行 (Data Parallel): 在DP组内分割数据，每个组持有完整的模型副本
- 张量并行 (Tensor Parallel): 在TP组内分割张量（权重矩阵）
- 流水线并行 (Pipeline Parallel): 在PP组内按层分割模型

增强功能:
- 详细的错误处理和验证
- 自动进程组拓扑检测
- 内存使用监控
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, List, Dict, Any
from .base import BaseParallel, ParallelConfigError, ParallelInitError
import logging

logger = logging.getLogger(__name__)


class HybridParallelConfigError(ParallelConfigError):
    """3D并行配置错误"""
    pass


class HybridParallelInitError(ParallelInitError):
    """3D并行初始化错误"""
    pass


class HybridParallel(BaseParallel):
    """
    3D混合并行策略类
    
    实现了数据并行、张量并行和流水线并行的组合。
    通过创建多个进程组来管理不同维度的并行。
    
    增强功能：
    1. 详细的配置验证和错误提示
    2. 自动进程组拓扑检测
    3. 内存使用监控
    
    Attributes:
        model: PyTorch 模型实例
        rank: 全局rank
        world_size: 总进程数
        dp_size: 数据并行大小
        tp_size: 张量并行大小
        pp_size: 流水线并行大小
        dp_rank: 数据并行组内的rank
        tp_rank: 张量并行组内的rank
        pp_rank: 流水线并行组内的rank (即stage id)
        dp_group: 数据并行进程组
        tp_group: 张量并行进程组
        pp_group: 流水线并行进程组
        is_first_stage: 是否为流水线第一个阶段
        is_last_stage: 是否为流水线最后一个阶段
        stage_layers: 当前流水线阶段的层
    
    Example:
        >>> model = TransformerModel()
        >>> hp = HybridParallel(
        ...     model, rank=0, world_size=8,
        ...     dp_size=2, tp_size=2, pp_size=2
        ... )
        >>> output = hp.forward(inputs)
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        dp_size: int = 1,
        tp_size: int = 1,
        pp_size: int = 1,
        tensor_parallel_mode: str = "row",
        pipeline_chunks: int = 1
    ):
        """
        初始化3D混合并行策略
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的全局rank
            world_size: 总进程数
            dp_size: 数据并行大小
            tp_size: 张量并行大小
            pp_size: 流水线并行大小
            tensor_parallel_mode: 张量并行模式 ("row" 或 "column")
            pipeline_chunks: 流水线微批次数量
        
        Raises:
            HybridParallelConfigError: 当配置不合法时抛出
        """
        # 验证配置
        self._validate_config(
            model, rank, world_size, dp_size, tp_size, pp_size,
            tensor_parallel_mode, pipeline_chunks
        )
        
        super().__init__(model, rank, world_size)
        
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.tensor_parallel_mode = tensor_parallel_mode
        self.pipeline_chunks = pipeline_chunks
        
        # 初始化进程组
        self.dp_group = None
        self.tp_group = None
        self.pp_group = None
        
        try:
            self._init_process_groups()
        except Exception as e:
            raise HybridParallelInitError(
                f"Failed to initialize process groups: {e}. "
                f"Please check your distributed environment setup."
            )
        
        # 流水线相关属性
        self.is_first_stage = (self.pp_rank == 0)
        self.is_last_stage = (self.pp_rank == self.pp_size - 1)
        self.stage_layers: Optional[nn.Sequential] = None
        
        # 分割模型（流水线分割 + 张量并行分割）
        try:
            self._partition_model()
        except Exception as e:
            raise HybridParallelInitError(
                f"Failed to partition model: {e}. "
                f"Please check if your model is compatible with 3D parallelism."
            )
        
        # 将模型移动到当前设备
        if self.stage_layers is not None:
            self.stage_layers.to(self.device)
        
        logger.info(
            f"HybridParallel initialized: rank={rank}, "
            f"DP={dp_size}, TP={tp_size}, PP={pp_size}"
        )
    
    def _validate_config(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        dp_size: int,
        tp_size: int,
        pp_size: int,
        tensor_parallel_mode: str,
        pipeline_chunks: int
    ) -> None:
        """
        验证3D并行配置
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的全局rank
            world_size: 总进程数
            dp_size: 数据并行大小
            tp_size: 张量并行大小
            pp_size: 流水线并行大小
            tensor_parallel_mode: 张量并行模式
            pipeline_chunks: 流水线微批次数量
        
        Raises:
            HybridParallelConfigError: 当配置不合法时抛出
        """
        # 验证进程数匹配
        total_parallel_size = dp_size * tp_size * pp_size
        if total_parallel_size != world_size:
            raise HybridParallelConfigError(
                f"dp_size({dp_size}) × tp_size({tp_size}) × pp_size({pp_size}) "
                f"= {total_parallel_size} ≠ world_size({world_size}). "
                f"The product of parallel sizes must equal world_size."
            )
        
        # 验证并行大小
        if dp_size < 1 or tp_size < 1 or pp_size < 1:
            raise HybridParallelConfigError(
                f"Parallel sizes must be positive: dp_size={dp_size}, "
                f"tp_size={tp_size}, pp_size={pp_size}"
            )
        
        # 验证张量并行模式
        if tensor_parallel_mode not in ["row", "column"]:
            raise HybridParallelConfigError(
                f"tensor_parallel_mode must be 'row' or 'column', "
                f"got '{tensor_parallel_mode}'"
            )
        
        # 验证流水线分块
        if pipeline_chunks < 1:
            raise HybridParallelConfigError(
                f"pipeline_chunks must be positive, got {pipeline_chunks}"
            )
        
        if pp_size > 1 and pipeline_chunks < 2:
            raise HybridParallelConfigError(
                f"When using pipeline parallelism (pp_size={pp_size}), "
                f"pipeline_chunks must be >= 2 to enable micro-batch processing."
            )
        
        # 验证模型
        if not isinstance(model, nn.Module):
            raise HybridParallelConfigError(
                f"model must be an nn.Module instance, got {type(model)}"
            )
        
        # 验证rank
        if rank < 0 or rank >= world_size:
            raise HybridParallelConfigError(
                f"rank must be in range [0, {world_size}), got {rank}"
            )
    
    def _init_process_groups(self) -> None:
        """
        初始化3D并行的进程组
        
        创建三个维度的进程组：
        - 数据并行组 (dp_group): 相同TP rank和PP rank的进程
        - 张量并行组 (tp_group): 相同DP rank和PP rank的进程
        - 流水线并行组 (pp_group): 相同DP rank和TP rank的进程
        """
        # 计算当前进程在各维度的rank
        self.pp_rank = self.rank % self.pp_size
        remaining = self.rank // self.pp_size
        self.tp_rank = remaining % self.tp_size
        self.dp_rank = remaining // self.tp_size
        
        # 检查分布式环境是否已初始化
        if not dist.is_initialized():
            # 单进程模式：所有进程组设为None
            if self.rank == 0:
                logger.warning(
                    "[HybridParallel] Distributed environment not initialized, "
                    "running in single-process mode"
                )
            return
        
        try:
            # 创建张量并行组
            self._create_tensor_parallel_groups()
            
            # 创建流水线并行组
            self._create_pipeline_parallel_groups()
            
            # 创建数据并行组
            self._create_data_parallel_groups()
            
        except Exception as e:
            raise HybridParallelInitError(
                f"Failed to create process groups: {e}"
            )
    
    def _create_tensor_parallel_groups(self) -> None:
        """创建张量并行组"""
        tp_ranks = []
        for dp in range(self.dp_size):
            for pp in range(self.pp_size):
                group_ranks = [
                    dp * self.tp_size * self.pp_size + tp * self.pp_size + pp
                    for tp in range(self.tp_size)
                ]
                tp_ranks.append(group_ranks)
        
        self.tp_group = None
        for ranks in tp_ranks:
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group
    
    def _create_pipeline_parallel_groups(self) -> None:
        """创建流水线并行组"""
        pp_ranks = []
        for dp in range(self.dp_size):
            for tp in range(self.tp_size):
                group_ranks = [
                    dp * self.tp_size * self.pp_size + tp * self.pp_size + pp
                    for pp in range(self.pp_size)
                ]
                pp_ranks.append(group_ranks)
        
        self.pp_group = None
        for ranks in pp_ranks:
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.pp_group = group
    
    def _create_data_parallel_groups(self) -> None:
        """创建数据并行组"""
        dp_ranks = []
        for tp in range(self.tp_size):
            for pp in range(self.pp_size):
                group_ranks = [
                    dp * self.tp_size * self.pp_size + tp * self.pp_size + pp
                    for dp in range(self.dp_size)
                ]
                dp_ranks.append(group_ranks)
        
        self.dp_group = None
        for ranks in dp_ranks:
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.dp_group = group
    
    def _partition_model(self) -> None:
        """
        分割模型到流水线阶段，并在每个阶段内应用张量并行
        
        1. 首先按流水线并行分割模型层
        2. 然后在每个阶段的线性层上应用张量并行
        """
        # 提取所有层
        all_layers = self._extract_layers()
        
        if len(all_layers) < self.pp_size:
            raise HybridParallelConfigError(
                f"Model has {len(all_layers)} layers, but pp_size is {self.pp_size}. "
                f"Consider reducing pp_size or using a larger model."
            )
        
        # 计算当前流水线阶段的层范围
        layers_per_stage = len(all_layers) // self.pp_size
        start_idx = layers_per_stage * self.pp_rank
        end_idx = layers_per_stage * (self.pp_rank + 1)
        
        # 最后一个阶段包含剩余的层
        if self.is_last_stage:
            end_idx = len(all_layers)
        
        # 获取当前阶段的层
        stage_layers_list = all_layers[start_idx:end_idx]
        
        # 在当前阶段的层上应用张量并行
        if self.tp_size > 1:
            stage_layers_list = self._apply_tensor_parallel(stage_layers_list)
        
        # 创建当前阶段的Sequential模型
        self.stage_layers = nn.Sequential(*stage_layers_list)
        self.stage_layers.to(self.device)
    
    def _extract_layers(self) -> List[nn.Module]:
        """
        从模型中提取所有层
        
        Returns:
            层列表
        """
        layers: List[nn.Module] = []
        
        if hasattr(self.model, 'layers') and isinstance(self.model.layers, nn.ModuleList):
            layers = list(self.model.layers)
        elif hasattr(self.model, 'features'):
            if hasattr(self.model.features, 'children'):
                layers = list(self.model.features.children())
            else:
                layers = [self.model.features]
        elif hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
            enc_layers = list(self.model.encoder.children()) if hasattr(self.model.encoder, 'children') else [self.model.encoder]
            dec_layers = list(self.model.decoder.children()) if hasattr(self.model.decoder, 'children') else [self.model.decoder]
            layers = enc_layers + dec_layers
        else:
            for name, module in self.model.named_children():
                if isinstance(module, nn.Sequential):
                    layers.extend(list(module.children()))
                else:
                    layers.append(module)
        
        if not layers:
            modules = list(self.model.children())
            if modules:
                layers = modules
            else:
                layers = [self.model]
        
        return layers
    
    def _apply_tensor_parallel(self, layers: List[nn.Module]) -> List[nn.Module]:
        """
        在层列表上应用张量并行
        
        Args:
            layers: 层列表
        
        Returns:
            应用张量并行后的层列表
        """
        parallelized_layers = []
        
        for layer in layers:
            if isinstance(layer, nn.Linear):
                # 对线性层应用张量并行
                parallelized_layer = self._parallelize_linear(layer)
                parallelized_layers.append(parallelized_layer)
            elif isinstance(layer, nn.Sequential):
                # 递归处理Sequential
                new_seq_layers = self._apply_tensor_parallel(list(layer.children()))
                parallelized_layers.append(nn.Sequential(*new_seq_layers))
            else:
                parallelized_layers.append(layer)
        
        return parallelized_layers
    
    def _parallelize_linear(self, linear_layer: nn.Linear) -> nn.Linear:
        """
        对线性层应用张量并行
        
        Args:
            linear_layer: 要并行化的线性层
        
        Returns:
            并行化后的线性层
        """
        if self.tensor_parallel_mode == "row":
            return self._parallelize_row(linear_layer)
        else:
            return self._parallelize_column(linear_layer)
    
    def _parallelize_row(self, linear_layer: nn.Linear) -> nn.Linear:
        """
        行并行：按输出维度分割权重
        """
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        split_out_features = out_features // self.tp_size
        
        new_linear = nn.Linear(
            in_features,
            split_out_features,
            bias=linear_layer.bias is not None
        )
        
        with torch.no_grad():
            start_idx = split_out_features * self.tp_rank
            end_idx = split_out_features * (self.tp_rank + 1)
            new_linear.weight.copy_(linear_layer.weight[start_idx:end_idx, :])
            if linear_layer.bias is not None:
                new_linear.bias.copy_(linear_layer.bias[start_idx:end_idx])
        
        return new_linear
    
    def _parallelize_column(self, linear_layer: nn.Linear) -> nn.Linear:
        """
        列并行：按输入维度分割权重
        """
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        split_in_features = in_features // self.tp_size
        
        new_linear = nn.Linear(
            split_in_features,
            out_features,
            bias=linear_layer.bias is not None
        )
        
        with torch.no_grad():
            start_idx = split_in_features * self.tp_rank
            end_idx = split_in_features * (self.tp_rank + 1)
            new_linear.weight.copy_(linear_layer.weight[:, start_idx:end_idx])
            if linear_layer.bias is not None:
                new_linear.bias.copy_(linear_layer.bias)
        
        return new_linear
    
    def forward(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        3D混合并行前向传播
        
        Args:
            inputs: 输入数据张量
        
        Returns:
            模型输出张量（仅最后一个流水线阶段返回非None）
        """
        if self.pipeline_chunks > 1:
            return self._forward_micro_batches(inputs)
        else:
            return self._forward_single_batch(inputs)
    
    def _forward_single_batch(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        单批次前向传播
        """
        # 第一个流水线阶段：接收输入
        if self.is_first_stage:
            x = self.to_device(inputs)
        else:
            # 其他阶段：从前一阶段接收
            x = self._recv_tensor(self.pp_rank - 1)
        
        # 张量并行column模式：切分输入
        if self.tp_size > 1 and self.tensor_parallel_mode == "column":
            split_size = x.size(-1) // self.tp_size
            x = x[:, split_size * self.tp_rank : split_size * (self.tp_rank + 1)]
        
        # 执行当前阶段的层
        if self.stage_layers is not None:
            x = self.stage_layers(x)
        
        # 张量并行row模式：all-gather输出
        if self.tp_size > 1 and self.tensor_parallel_mode == "row":
            x = self._tensor_all_gather(x)
        # 张量并行column模式：all-reduce输出
        elif self.tp_size > 1 and self.tensor_parallel_mode == "column":
            x = self._tensor_all_reduce(x)
        
        # 如果不是最后一个阶段，发送给下一阶段
        if not self.is_last_stage:
            self._send_tensor(x, self.pp_rank + 1)
            return None
        
        return x
    
    def _forward_micro_batches(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        微批次前向传播
        """
        if not isinstance(inputs, torch.Tensor):
            raise HybridParallelConfigError("Chunked pipeline requires tensor input")
        
        batch_size = inputs.size(0)
        chunk_size = batch_size // self.pipeline_chunks
        
        outputs: List[torch.Tensor] = []
        
        for i in range(self.pipeline_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.pipeline_chunks - 1 else batch_size
            chunk_input = inputs[start_idx:end_idx]
            
            # 第一个流水线阶段
            if self.is_first_stage:
                x = self.to_device(chunk_input)
                x = x.view(x.size(0), -1)
                
                # 张量并行column模式：切分输入
                if self.tp_size > 1 and self.tensor_parallel_mode == "column":
                    split_size = x.size(-1) // self.tp_size
                    x = x[:, split_size * self.tp_rank : split_size * (self.tp_rank + 1)]
                
                if self.stage_layers is not None:
                    x = self.stage_layers(x)
                
                # 张量并行通信
                if self.tp_size > 1 and self.tensor_parallel_mode == "row":
                    x = self._tensor_all_gather(x)
                elif self.tp_size > 1 and self.tensor_parallel_mode == "column":
                    x = self._tensor_all_reduce(x)
                
                if not self.is_last_stage:
                    self._send_tensor(x, self.pp_rank + 1)
                else:
                    outputs.append(x)
            
            # 最后一个流水线阶段
            elif self.is_last_stage:
                x = self._recv_tensor(self.pp_rank - 1)
                
                if self.stage_layers is not None:
                    x = self.stage_layers(x)
                
                outputs.append(x)
            
            # 中间流水线阶段
            else:
                x = self._recv_tensor(self.pp_rank - 1)
                
                if self.stage_layers is not None:
                    x = self.stage_layers(x)
                
                self._send_tensor(x, self.pp_rank + 1)
        
        # 最后一个阶段拼接所有微批次的输出
        if self.is_last_stage and outputs:
            return torch.cat(outputs, dim=0)
        return None
    
    def _tensor_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        在张量并行组内执行all-gather
        """
        if self.tp_size == 1 or self.tp_group is None:
            return tensor
        
        gathered = [torch.zeros_like(tensor) for _ in range(self.tp_size)]
        dist.all_gather(gathered, tensor, group=self.tp_group)
        return torch.cat(gathered, dim=-1)
    
    def _tensor_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        在张量并行组内执行all-reduce
        """
        if self.tp_size == 1 or self.tp_group is None:
            return tensor
        
        output = tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output
    
    def _send_tensor(self, tensor: torch.Tensor, dst_pp_rank: int) -> None:
        """
        发送张量到目标流水线阶段
        """
        if not dist.is_initialized():
            return
        
        # 计算目标全局rank
        dst_global_rank = self.dp_rank * self.tp_size * self.pp_size + self.tp_rank * self.pp_size + dst_pp_rank
        
        # 准备形状信息
        shape_list = list(tensor.shape)
        shape_tensor = torch.tensor(shape_list, device=self.device, dtype=torch.long)
        shape_len = torch.tensor([len(shape_list)], device=self.device, dtype=torch.long)
        
        # 发送形状长度
        dist.send(shape_len, dst=dst_global_rank)
        # 发送形状张量
        dist.send(shape_tensor, dst=dst_global_rank)
        # 发送数据张量
        dist.send(tensor.contiguous(), dst=dst_global_rank)
    
    def _recv_tensor(self, src_pp_rank: int) -> torch.Tensor:
        """
        从源流水线阶段接收张量
        """
        if not dist.is_initialized():
            return torch.zeros(1, device=self.device)
        
        # 计算源全局rank
        src_global_rank = self.dp_rank * self.tp_size * self.pp_size + self.tp_rank * self.pp_size + src_pp_rank
        
        # 接收形状长度
        shape_len = torch.zeros(1, dtype=torch.long, device=self.device)
        dist.recv(shape_len, src=src_global_rank)
        
        # 接收形状张量
        shape_tensor = torch.zeros(shape_len.item(), dtype=torch.long, device=self.device)
        dist.recv(shape_tensor, src=src_global_rank)
        
        # 接收数据张量
        tensor = torch.zeros(*shape_tensor.tolist(), device=self.device)
        dist.recv(tensor, src=src_global_rank)
        
        return tensor
    
    def gather_gradients(self) -> None:
        """
        收集梯度（用于数据并行）
        """
        if self.dp_size <= 1 or self.dp_group is None:
            return
        
        for param in self.stage_layers.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
                param.grad /= self.dp_size
    
    def broadcast_model(self) -> None:
        """
        广播模型参数（用于数据并行）
        """
        if self.dp_group is None:
            return
        
        for param in self.stage_layers.parameters():
            dist.broadcast(param, src=0)
    
    def get_stage_model(self) -> nn.Module:
        """
        获取当前流水线阶段的模型
        """
        if self.stage_layers is None:
            raise RuntimeError("Stage layers not initialized")
        return self.stage_layers
    
    def get_parallel_info(self) -> Dict[str, Any]:
        """
        获取并行配置信息
        """
        return {
            "global_rank": self.rank,
            "world_size": self.world_size,
            "dp_size": self.dp_size,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "is_first_stage": self.is_first_stage,
            "is_last_stage": self.is_last_stage,
            "tensor_parallel_mode": self.tensor_parallel_mode,
            "pipeline_chunks": self.pipeline_chunks,
        }
