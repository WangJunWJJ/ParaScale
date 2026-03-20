# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : model_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 模型并行模块

本模块实现了模型并行策略，将模型分割到不同设备，
每个设备运行模型的一部分，通过点对点通信传递中间结果。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List
from .base import BaseParallel


class ModelParallel(BaseParallel):
    """
    模型并行策略类
    
    模型并行将模型的不同层分配到不同设备上，适用于
    单个 GPU 无法容纳整个模型的情况。数据按顺序流经
    各个设备，每个设备只计算自己负责的部分。
    
    工作流程：
    1. 将模型按层分割到不同设备
    2. 数据从第一个设备开始，依次流经所有设备
    3. 每个设备计算完成后，将结果发送给下一个设备
    4. 最后一个设备输出最终结果
    
    注意：模型并行的通信开销较大，适合模型非常大、
    无法放入单个 GPU 的情况。
    
    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank
        world_size: 世界大小
        device: 当前设备
    
    Example:
        >>> model = LargeModel()
        >>> mp = ModelParallel(model, rank=0, world_size=2)
        >>> output = mp.forward(inputs)
    """
    
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        """
        初始化模型并行策略
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank
            world_size: 世界大小（设备数量）
        """
        super().__init__(model, rank, world_size)
        self._split_model()
    
    def _split_model(self) -> None:
        """
        将模型分割到不同设备
        
        根据模型结构，将不同的层分配到不同的设备。
        支持以下模型结构：
        - 带有 layers 属性的模型（如 nn.ModuleList）
        - 带有 encoder/decoder 属性的模型
        - 普通 Sequential 模型
        """
        # 处理带有 layers 属性的模型
        if hasattr(self.model, 'layers'):
            total_layers = len(self.model.layers)
            layers_per_rank = total_layers // self.world_size
            start_idx = layers_per_rank * self.rank
            end_idx = layers_per_rank * (self.rank + 1)
            
            # 最后一个 rank 处理剩余的层
            if self.rank == self.world_size - 1:
                end_idx = total_layers
            
            current_layers = self.model.layers[start_idx:end_idx]
            sub_model = nn.Sequential(*current_layers)
            sub_model.to(self.device)
            
            setattr(self.model, f'stage_{self.rank}', sub_model)
            
        # 处理 encoder-decoder 结构的模型
        elif hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
            if self.rank == 0:
                self.model.encoder.to(self.device)
                setattr(self.model, 'stage_0', self.model.encoder)
            else:
                self.model.decoder.to(self.device)
                setattr(self.model, f'stage_{self.rank}', self.model.decoder)
        else:
            # 处理普通模型
            modules: List[tuple] = []
            for name, module in self.model.named_children():
                modules.append((name, module))
            
            if modules:
                modules_per_rank = len(modules) // self.world_size
                start_idx = modules_per_rank * self.rank
                end_idx = modules_per_rank * (self.rank + 1)
                
                if self.rank == self.world_size - 1:
                    end_idx = len(modules)
                
                current_modules = [module for _, module in modules[start_idx:end_idx]]
                sub_model = nn.Sequential(*current_modules)
                sub_model.to(self.device)
                
                setattr(self.model, f'stage_{self.rank}', sub_model)
    
    def forward(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        前向传播
        
        按顺序执行各个阶段的计算，通过点对点通信传递中间结果。
        
        Args:
            inputs: 输入数据张量
        
        Returns:
            模型输出张量，非最后一个 rank 返回 None
        """
        # 将输入展平（假设为图像数据）
        x = inputs.view(-1, 3072)  # 3072 = 3 * 32 * 32 (CIFAR10)
        
        # 按顺序执行各个阶段
        for stage in range(self.world_size):
            if stage == self.rank:
                # 当前 rank 执行自己的阶段
                if hasattr(self.model, f'stage_{self.rank}'):
                    stage_model = getattr(self.model, f'stage_{self.rank}')
                    x = self.to_device(x)
                    x = stage_model(x)
            
            # 如果不是最后一个阶段，需要传递数据
            if stage < self.world_size - 1:
                if stage == self.rank:
                    # 发送数据给下一个 rank
                    next_rank = stage + 1
                    try:
                        dist.send(x, dst=next_rank)
                    except RuntimeError as e:
                        raise RuntimeError(f"Failed to send data to rank {next_rank}: {e}")
                        
                elif stage + 1 == self.rank:
                    # 从上一个 rank 接收数据
                    prev_rank = stage
                    if hasattr(self.model, f'stage_{self.rank}'):
                        stage_model = getattr(self.model, f'stage_{self.rank}')
                        # 查找第一个线性层以确定输入维度
                        first_linear = None
                        for module in stage_model.modules():
                            if isinstance(module, nn.Linear):
                                first_linear = module
                                break
                        if first_linear:
                            batch_size = x.size(0)
                            in_features = first_linear.in_features
                            x = torch.empty(batch_size, in_features, device=self.device)
                        else:
                            x = torch.empty_like(x, device=self.device)
                    else:
                        x = torch.empty_like(x, device=self.device)
                    try:
                        dist.recv(x, src=prev_rank)
                    except RuntimeError as e:
                        raise RuntimeError(f"Failed to receive data from rank {prev_rank}: {e}")
        
        # 只有最后一个 rank 返回输出
        if self.rank == self.world_size - 1:
            return x
        
        return None
