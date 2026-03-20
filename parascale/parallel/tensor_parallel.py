# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : tensor_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 张量并行模块

本模块实现了张量并行策略，将模型的张量（权重矩阵）分割到不同 GPU，
支持梯度传播的分布式通信。参考 DeepSpeed 的张量并行实现原理。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function
from typing import Optional, List, Literal
from .base import BaseParallel


class AllGatherWithGradient(Function):
    """
    支持梯度传播的 All-Gather 操作
    
    在前向传播时，收集所有 rank 的张量并拼接；
    在反向传播时，将梯度切分并返回给对应的 rank。
    
    用于行并行模式，将各 rank 的输出拼接成完整输出。
    
    Example:
        >>> x = torch.randn(32, 128, device='cuda')
        >>> gathered = AllGatherWithGradient.apply(x, world_size)
        >>> # gathered.shape: (32, 128 * world_size)
    """
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        """
        前向传播：收集所有 rank 的张量并拼接
        
        Args:
            ctx: PyTorch 上下文对象，用于保存反向传播所需的信息
            tensor: 当前 rank 的张量
            world_size: 世界大小
        
        Returns:
            拼接后的张量，维度为 (batch_size, features * world_size)
        """
        ctx.world_size = world_size
        ctx.rank = dist.get_rank()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        反向传播：切分梯度并返回给对应的 rank
        
        Args:
            ctx: PyTorch 上下文对象
            grad_output: 拼接后张量的梯度
        
        Returns:
            元组 (当前 rank 的梯度, None)
        """
        split_size = grad_output.size(-1) // ctx.world_size
        grad_input = grad_output[:, split_size * ctx.rank : split_size * (ctx.rank + 1)]
        return grad_input, None


class AllReduceWithGradient(Function):
    """
    支持梯度传播的 All-Reduce 操作
    
    在前向传播时，对所有 rank 的张量求和；
    在反向传播时，对梯度进行 all-reduce 并平均。
    
    用于列并行模式，将各 rank 的部分输出求和得到完整输出。
    
    Example:
        >>> x = torch.randn(32, 128, device='cuda')
        >>> reduced = AllReduceWithGradient.apply(x, world_size)
        >>> # reduced 是所有 rank 张量的和
    """
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        """
        前向传播：对所有 rank 的张量求和
        
        Args:
            ctx: PyTorch 上下文对象
            tensor: 当前 rank 的张量
            world_size: 世界大小
        
        Returns:
            所有 rank 张量求和的结果
        """
        ctx.world_size = world_size
        output = tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        反向传播：对梯度进行 all-reduce 并平均
        
        Args:
            ctx: PyTorch 上下文对象
            grad_output: 输出的梯度
        
        Returns:
            元组 (平均后的梯度, None)
        """
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)
        return grad_input / ctx.world_size, None


class TensorParallel(BaseParallel):
    """
    张量并行策略类
    
    张量并行将模型的权重矩阵分割到不同 GPU 上，每个 GPU 只保存
    部分权重。支持两种并行模式：
    
    - 行并行（row）：按输出维度分割权重，输出需要 all-gather 拼接
    - 列并行（column）：按输入维度分割权重，输出需要 all-reduce 求和
    
    使用 torch.autograd.Function 实现支持梯度传播的分布式通信，
    解决了传统实现中梯度图断裂的问题。
    
    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank
        world_size: 世界大小
        mode: 张量并行模式（"row" 或 "column"）
        parallel_layers: 已并行化的层名称列表
        device: 当前设备
    
    Example:
        >>> model = SimpleModel()
        >>> tp = TensorParallel(model, rank=0, world_size=2, mode="row")
        >>> output = tp.forward(inputs)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        rank: int, 
        world_size: int, 
        mode: Literal["row", "column"] = "row"
    ):
        """
        初始化张量并行策略
        
        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank
            world_size: 世界大小（GPU 数量）
            mode: 张量并行模式，支持 "row"（行并行）或 "column"（列并行）
        
        Raises:
            ValueError: 当 mode 不是 "row" 或 "column" 时抛出
        """
        if mode not in ["row", "column"]:
            raise ValueError(f"Unsupported tensor parallel mode: {mode}")
        
        super().__init__(model, rank, world_size)
        self.mode = mode
        self.parallel_layers: List[str] = []
        
        # 并行化模型
        self._parallelize_model()
        self.model.to(self.device)
    
    def _parallelize_model(self) -> None:
        """
        并行化模型中的线性层
        
        遍历模型中的所有模块，找到符合条件的线性层并进行并行化。
        当前只并行化最后一层（fc2 或 fc），避免维度不匹配问题。
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 只并行化顶层的 fc2 或 fc 层
                if "." not in name and (name == "fc2" or name == "fc"):
                    parallelized_layer = self._parallelize_linear(module)
                    setattr(self.model, name, parallelized_layer)
                    self.parallel_layers.append(name)
    
    def _parallelize_linear(self, linear_layer: nn.Linear) -> nn.Linear:
        """
        并行化单个线性层
        
        根据并行模式，将线性层的权重分割到不同 rank。
        
        Args:
            linear_layer: 要并行化的线性层
        
        Returns:
            并行化后的新线性层
        """
        if self.mode == "row":
            return self._parallelize_row(linear_layer)
        else:
            return self._parallelize_column(linear_layer)
    
    def _parallelize_row(self, linear_layer: nn.Linear) -> nn.Linear:
        """
        行并行：按输出维度分割权重
        
        将权重矩阵按行（输出维度）分割，每个 rank 保存一部分行。
        输出需要通过 all-gather 拼接成完整结果。
        
        Args:
            linear_layer: 要并行化的线性层
        
        Returns:
            分割后的新线性层
        """
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        split_out_features = out_features // self.world_size
        
        # 创建新的线性层，输出维度为原来的 1/world_size
        new_linear = nn.Linear(
            in_features, 
            split_out_features, 
            bias=linear_layer.bias is not None
        )
        
        # 复制对应的权重切片
        with torch.no_grad():
            start_idx = split_out_features * self.rank
            end_idx = split_out_features * (self.rank + 1)
            new_linear.weight.copy_(linear_layer.weight[start_idx:end_idx, :])
            if linear_layer.bias is not None:
                new_linear.bias.copy_(linear_layer.bias[start_idx:end_idx])
        
        return new_linear
    
    def _parallelize_column(self, linear_layer: nn.Linear) -> nn.Linear:
        """
        列并行：按输入维度分割权重
        
        将权重矩阵按列（输入维度）分割，每个 rank 保存一部分列。
        输出需要通过 all-reduce 求和得到完整结果。
        
        Args:
            linear_layer: 要并行化的线性层
        
        Returns:
            分割后的新线性层
        """
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        split_in_features = in_features // self.world_size
        
        # 创建新的线性层，输入维度为原来的 1/world_size
        new_linear = nn.Linear(
            split_in_features, 
            out_features, 
            bias=linear_layer.bias is not None
        )
        
        # 复制对应的权重切片
        with torch.no_grad():
            start_idx = split_in_features * self.rank
            end_idx = split_in_features * (self.rank + 1)
            new_linear.weight.copy_(linear_layer.weight[:, start_idx:end_idx])
            if linear_layer.bias is not None:
                # 偏置不分割，但需要除以 world_size
                new_linear.bias.copy_(linear_layer.bias)
        
        return new_linear
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        根据并行模式执行前向传播：
        - 行并行：输出通过 all-gather 拼接
        - 列并行：输入先切分，输出通过 all-reduce 求和
        
        Args:
            inputs: 输入数据张量
        
        Returns:
            模型输出张量
        """
        x = self.to_device(inputs)
        
        # 列并行模式：先切分输入
        if self.mode == "column":
            split_size = x.size(-1) // self.world_size
            x = x[:, split_size * self.rank : split_size * (self.rank + 1)]
        
        # 执行模型前向传播
        x = self.model(x)
        
        # 行并行模式：all-gather 拼接输出
        if self.mode == "row":
            x = AllGatherWithGradient.apply(x, self.world_size)
        # 列并行模式：all-reduce 求和输出
        elif self.mode == "column":
            x = AllReduceWithGradient.apply(x, self.world_size)
        
        return x
