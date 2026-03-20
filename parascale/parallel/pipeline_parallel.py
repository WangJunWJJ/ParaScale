# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : pipeline_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 流水线并行模块

本模块实现了流水线并行策略，将模型按层分割到不同设备，
支持动态形状传输和微批次处理，提高设备利用率。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List
from .base import BaseParallel


class PipelineParallel(BaseParallel):
    """
    流水线并行策略类
    
    流水线并行将模型的不同层分配到不同设备上，形成多个阶段。
    数据按顺序流经各个阶段，每个阶段只计算自己负责的层。
    
    支持微批次（micro-batch）处理，将一个批次分割成多个微批次，
    不同微批次可以在不同阶段并行执行，提高设备利用率。
    
    核心特性：
    - 动态形状传输：支持任意形状的张量传输
    - 微批次处理：提高流水线利用率
    - 自动层提取：支持多种模型结构
    
    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank
        world_size: 世界大小（阶段数）
        chunks: 微批次数量
        stage_layers: 当前阶段的层
        is_first: 是否为第一个阶段
        is_last: 是否为最后一个阶段
        device: 当前设备
    
    Example:
        >>> model = PipelineModel()
        >>> pp = PipelineParallel(model, rank=0, world_size=2, chunks=4)
        >>> output = pp.forward(inputs)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        rank: int, 
        world_size: int, 
        chunks: int = 1
    ):
        """
        初始化流水线并行策略
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank（阶段编号）
            world_size: 世界大小（阶段总数）
            chunks: 微批次数量，默认为 1（不使用微批次）
        
        Raises:
            ValueError: 当 chunks 小于 1 时抛出
        """
        if chunks <= 0:
            raise ValueError("Chunks must be positive")
        
        super().__init__(model, rank, world_size)
        self.chunks = chunks
        self.stage_layers: Optional[nn.Sequential] = None
        self.is_first = (rank == 0)
        self.is_last = (rank == world_size - 1)
        
        # 分割模型
        self._partition_model()
    
    def _partition_model(self) -> None:
        """
        将模型分割到不同的流水线阶段
        
        从模型中提取所有层，然后按阶段数均匀分配。
        最后一个阶段可能包含更多的层。
        """
        all_layers = self._extract_layers()
        
        if len(all_layers) < self.world_size:
            raise ValueError(
                f"Model has {len(all_layers)} layers, but world_size is {self.world_size}"
            )
        
        # 计算每个阶段的层范围
        layers_per_stage = len(all_layers) // self.world_size
        start_idx = layers_per_stage * self.rank
        end_idx = layers_per_stage * (self.rank + 1)
        
        # 最后一个阶段包含剩余的层
        if self.is_last:
            end_idx = len(all_layers)
        
        # 创建当前阶段的模型
        self.stage_layers = nn.Sequential(*all_layers[start_idx:end_idx])
        self.stage_layers.to(self.device)
    
    def _extract_layers(self) -> List[nn.Module]:
        """
        从模型中提取所有层
        
        支持多种模型结构：
        - 带有 layers 属性的模型（如 nn.ModuleList）
        - 带有 features 属性的模型
        - encoder-decoder 结构的模型
        - 普通 Sequential 模型
        
        Returns:
            层列表
        """
        layers: List[nn.Module] = []
        
        # 处理带有 layers 属性的模型
        if hasattr(self.model, 'layers') and isinstance(self.model.layers, nn.ModuleList):
            layers = list(self.model.layers)
        # 处理带有 features 属性的模型
        elif hasattr(self.model, 'features'):
            if hasattr(self.model.features, 'children'):
                layers = list(self.model.features.children())
            else:
                layers = [self.model.features]
        # 处理 encoder-decoder 结构
        elif hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
            enc_layers = list(self.model.encoder.children()) if hasattr(self.model.encoder, 'children') else [self.model.encoder]
            dec_layers = list(self.model.decoder.children()) if hasattr(self.model.decoder, 'children') else [self.model.decoder]
            layers = enc_layers + dec_layers
        else:
            # 处理普通模型
            for name, module in self.model.named_children():
                if isinstance(module, nn.Sequential):
                    layers.extend(list(module.children()))
                else:
                    layers.append(module)
        
        # 如果没有提取到层，尝试直接使用模型的子模块
        if not layers:
            modules = list(self.model.children())
            if modules:
                layers = modules
            else:
                layers = [self.model]
        
        return layers
    
    def forward(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        前向传播
        
        根据是否使用微批次，选择不同的执行方式。
        
        Args:
            input_data: 输入数据张量
        
        Returns:
            模型输出张量，非最后一个阶段返回 None
        """
        if isinstance(input_data, torch.Tensor):
            input_data = self.to_device(input_data)
        
        if self.chunks > 1:
            return self._forward_micro_batches(input_data)
        else:
            return self._forward_single_batch(input_data)
    
    def _forward_single_batch(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        单批次前向传播
        
        不使用微批次，数据按顺序流经各个阶段。
        
        Args:
            input_data: 输入数据张量
        
        Returns:
            模型输出张量，非最后一个阶段返回 None
        """
        if self.is_first:
            # 第一个阶段：接收输入，处理后发送给下一阶段
            x = input_data
            if isinstance(x, torch.Tensor):
                x = x.view(x.size(0), -1)
            
            x = self.stage_layers(x)
            
            if not self.is_last:
                self._send_tensor(x, self.rank + 1)
                return None
            return x
        
        elif self.is_last:
            # 最后一个阶段：接收上一阶段的数据，处理后输出
            x = self._recv_tensor(self.rank - 1)
            x = self.stage_layers(x)
            return x
        
        else:
            # 中间阶段：接收数据，处理后发送给下一阶段
            x = self._recv_tensor(self.rank - 1)
            x = self.stage_layers(x)
            self._send_tensor(x, self.rank + 1)
            return None
    
    def _forward_micro_batches(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        微批次前向传播
        
        将输入数据分割成多个微批次，不同微批次可以在不同阶段并行执行。
        
        Args:
            input_data: 输入数据张量
        
        Returns:
            模型输出张量（最后一个阶段），其他阶段返回 None
        """
        if not isinstance(input_data, torch.Tensor):
            raise ValueError("Chunked pipeline requires tensor input")
        
        batch_size = input_data.size(0)
        chunk_size = batch_size // self.chunks
        
        outputs: List[torch.Tensor] = []
        
        # 逐个处理微批次
        for i in range(self.chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.chunks - 1 else batch_size
            chunk_input = input_data[start_idx:end_idx]
            
            if self.is_first:
                # 第一个阶段
                x = chunk_input.view(chunk_input.size(0), -1)
                x = self.stage_layers(x)
                
                if not self.is_last:
                    self._send_tensor(x, self.rank + 1)
                else:
                    outputs.append(x)
            
            elif self.is_last:
                # 最后一个阶段
                x = self._recv_tensor(self.rank - 1)
                x = self.stage_layers(x)
                outputs.append(x)
            
            else:
                # 中间阶段
                x = self._recv_tensor(self.rank - 1)
                x = self.stage_layers(x)
                self._send_tensor(x, self.rank + 1)
        
        # 最后一个阶段拼接所有微批次的输出
        if self.is_last and outputs:
            return torch.cat(outputs, dim=0)
        return None
    
    def _send_tensor(self, tensor: torch.Tensor, dst: int) -> None:
        """
        发送张量到目标 rank（支持动态形状）
        
        使用动态形状传输协议：
        1. 发送形状长度
        2. 发送形状张量
        3. 发送数据张量
        
        Args:
            tensor: 要发送的张量
            dst: 目标 rank
        
        Raises:
            RuntimeError: 发送失败时抛出
        """
        try:
            # 准备形状信息
            shape_list = list(tensor.shape)
            shape_tensor = torch.tensor(shape_list, device=self.device, dtype=torch.long)
            shape_len = torch.tensor([len(shape_list)], device=self.device, dtype=torch.long)
            
            # 发送形状长度
            dist.send(shape_len, dst=dst)
            # 发送形状张量
            dist.send(shape_tensor, dst=dst)
            # 发送数据张量
            dist.send(tensor.contiguous(), dst=dst)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to send tensor to rank {dst}: {e}")
    
    def _recv_tensor(self, src: int) -> torch.Tensor:
        """
        从源 rank 接收张量（支持动态形状）
        
        使用动态形状传输协议：
        1. 接收形状长度
        2. 接收形状张量
        3. 接收数据张量
        
        Args:
            src: 源 rank
        
        Returns:
            接收到的张量
        
        Raises:
            RuntimeError: 接收失败时抛出
        """
        try:
            # 接收形状长度
            shape_len = torch.zeros(1, dtype=torch.long, device=self.device)
            dist.recv(shape_len, src=src)
            
            # 接收形状张量
            shape_tensor = torch.zeros(shape_len.item(), dtype=torch.long, device=self.device)
            dist.recv(shape_tensor, src=src)
            
            # 接收数据张量
            tensor = torch.zeros(*shape_tensor.tolist(), device=self.device)
            dist.recv(tensor, src=src)
            
            return tensor
        except RuntimeError as e:
            raise RuntimeError(f"Failed to receive tensor from rank {src}: {e}")
    
    def get_stage_model(self) -> nn.Module:
        """
        获取当前阶段的模型
        
        Returns:
            当前阶段的模型（nn.Sequential）
        
        Raises:
            RuntimeError: 如果阶段模型未初始化
        """
        if self.stage_layers is None:
            raise RuntimeError("Stage layers not initialized")
        return self.stage_layers
