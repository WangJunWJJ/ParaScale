# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : tensor_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 张量并行模块

本模块实现了张量并行策略，将模型的张量（权重矩阵）分割到不同 GPU，
支持梯度传播的分布式通信。参考 DeepSpeed 的张量并行实现原理。

张量并行原理:
    - 行并行(row): 权重按输出维度分割，输出需all-gather
    - 列并行(column): 权重按输入维度分割，输入需scatter，输出需all-reduce
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function
from typing import Optional, List, Dict, Any, Literal
import logging

from .base import BaseParallel

logger = logging.getLogger(__name__)


class _AllGatherWithGradient(Function):
    """
    支持梯度传播的 All-Gather 操作

    用于行并行模式:
    - 前向: 收集所有 rank 的张量并拼接
    - 反向: 将梯度切分并返回给对应的 rank
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        ctx.world_size = world_size
        ctx.rank = rank
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        split_size = grad_output.size(-1) // ctx.world_size
        grad_input = grad_output[:, split_size * ctx.rank : split_size * (ctx.rank + 1)]
        return grad_input, None, None


class _AllReduceWithGradient(Function):
    """
    支持梯度传播的 All-Reduce 操作

    用于列并行模式:
    - 前向: 对所有 rank 的张量求和
    - 反向: 梯度直接传递给上游，无需特殊处理
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        ctx.world_size = world_size
        output = tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None


class TensorParallel(BaseParallel):
    """
    张量并行策略类

    张量并行将模型的权重矩阵分割到不同 GPU 上，每个 GPU 只保存
    部分权重。支持两种并行模式：

    - 行并行（row）：按输出维度分割权重，输出需要 all-gather 拼接
    - 列并行（column）：按输入维度分割权重，输出需要 all-reduce 求和

    梯度同步机制:
        - 行并行: 各 rank 计算部分输出，梯度通过 AllGather 反向自动切分
        - 列并行: 各 rank 计算部分输出，梯度通过 AllReduce 求和

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

        super().__init__(
            model, rank, world_size,
            parallel_config={"mode": mode}
        )
        self.mode = mode
        self.parallel_layers: List[str] = []
        self._layer_grad_handles: List[Any] = []

        self._parallelize_model()
        self.model.to(self.device)
        self._is_initialized = True

    def _parallelize_model(self) -> None:
        """
        并行化模型中的线性层

        遍历模型中的所有模块，找到符合条件的线性层并进行并行化。
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if "." not in name and (name == "fc2" or name == "fc"):
                    parallelized_layer = self._parallelize_linear(module)
                    setattr(self.model, name, parallelized_layer)
                    self.parallel_layers.append(name)

    def _parallelize_linear(self, linear_layer: nn.Linear) -> nn.Linear:
        """根据并行模式，并行化单个线性层"""
        if self.mode == "row":
            return self._parallelize_row(linear_layer)
        else:
            return self._parallelize_column(linear_layer)

    def _parallelize_row(self, linear_layer: nn.Linear) -> nn.Linear:
        """
        行并行：按输出维度分割权重

        每个 rank 持有 output_features // world_size 行的权重。
        前向传播时输出需要通过 all-gather 拼接。

        Args:
            linear_layer: 要并行化的线性层

        Returns:
            分割后的新线性层
        """
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        split_out_features = out_features // self.world_size

        if split_out_features * self.world_size != out_features:
            raise ValueError(
                f"Output features {out_features} must be divisible by world_size {self.world_size}"
            )

        new_linear = nn.Linear(
            in_features,
            split_out_features,
            bias=linear_layer.bias is not None
        )

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

        每个 rank 持有 input_features // world_size 列的权重。
        前向传播时输入需要先切分，输出需要通过 all-reduce 求和。

        Args:
            linear_layer: 要并行化的线性层

        Returns:
            分割后的新线性层
        """
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        split_in_features = in_features // self.world_size

        if split_in_features * self.world_size != in_features:
            raise ValueError(
                f"Input features {in_features} must be divisible by world_size {self.world_size}"
            )

        new_linear = nn.Linear(
            split_in_features,
            out_features,
            bias=linear_layer.bias is not None
        )

        with torch.no_grad():
            start_idx = split_in_features * self.rank
            end_idx = split_in_features * (self.rank + 1)
            new_linear.weight.copy_(linear_layer.weight[:, start_idx:end_idx])
            if linear_layer.bias is not None:
                new_linear.bias.copy_(linear_layer.bias)

        return new_linear

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        根据并行模式执行前向传播：
        - 行并行：输入完整，输出通过 all-gather 拼接
        - 列并行：输入先切分，输出通过 all-reduce 求和

        Args:
            inputs: 输入数据张量

        Returns:
            模型输出张量
        """
        x = self.to_device(inputs)

        if self.mode == "column":
            split_size = x.size(-1) // self.world_size
            x = x[:, split_size * self.rank : split_size * (self.rank + 1)]

        x = self.model(x)

        if self.mode == "row":
            x = _AllGatherWithGradient.apply(x, self.world_size, self.rank)
        elif self.mode == "column":
            x = _AllReduceWithGradient.apply(x, self.world_size)

        return x

    def gather_gradients(self) -> None:
        """
        收集梯度

        张量并行的梯度收集根据模式不同而不同：
        - 行并行: 梯度已经是完整的（通过 AllGather 反向自动处理）
        - 列并行: 梯度是部分和，需要 all-reduce 求和

        注意: 由于使用自定义 autograd Function，梯度会在反向传播时
        自动正确处理，此方法仅用于确保同步完成。
        """
        if not self._is_initialized:
            return

        import torch.distributed as dist
        if not dist.is_initialized():
            return

        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size

    def validate_distributed_state(self) -> None:
        """
        验证张量并行的分布式训练状态

        Raises:
            RuntimeError: 当分布式状态不合法时抛出
        """
        super().validate_distributed_state()

        import torch.distributed as dist

        if self.world_size > dist.get_world_size():
            raise RuntimeError(
                f"TensorParallel requires world_size={self.world_size}, "
                f"but distributed world_size={dist.get_world_size()}"
            )

    def get_parallel_info(self) -> Dict[str, Any]:
        """获取张量并行配置信息"""
        info = super().get_parallel_info()
        info.update({
            "mode": self.mode,
            "parallel_layers": self.parallel_layers,
        })
        return info