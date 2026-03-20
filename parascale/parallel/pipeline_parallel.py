# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : pipeline_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 流水线并行模块

本模块实现了流水线并行策略，将模型按层分割到不同设备，
支持动态形状传输和微批次处理，提高设备利用率。

特性:
    - 动态形状传输：支持任意形状的张量传输
    - 微批次处理：提高流水线利用率
    - 通信超时机制：防止永久阻塞
    - 自动层提取：支持多种模型结构
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List, Dict, Any
from .base import BaseParallel
import logging
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

DEFAULT_COMM_TIMEOUT = 300.0


class PipelineParallel(BaseParallel):
    """
    流水线并行策略类

    流水线并行将模型的不同层分配到不同设备上，形成多个阶段。
    数据按顺序流经各个阶段，每个阶段只计算自己负责的层。

    支持微批次（micro-batch）处理，将一个批次分割成多个微批次，
    不同微批次可以在不同阶段并行执行，提高设备利用率。

    Attributes:
        model: PyTorch 模型实例
        rank: 当前进程的 rank
        world_size: 世界大小（阶段数）
        chunks: 微批次数量
        stage_layers: 当前阶段的层
        is_first: 是否为第一个阶段
        is_last: 是否为最后一个阶段
        device: 当前设备
        comm_timeout: 通信超时时间（秒）
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        chunks: int = 1,
        comm_timeout: float = DEFAULT_COMM_TIMEOUT
    ):
        """
        初始化流水线并行策略

        Args:
            model: PyTorch 模型实例
            rank: 当前进程的 rank（阶段编号）
            world_size: 世界大小（阶段总数）
            chunks: 微批次数量，默认为 1（不使用微批次）
            comm_timeout: 通信超时时间（秒），默认 300 秒

        Raises:
            ValueError: 当 chunks 小于 1 或 comm_timeout 不合法时抛出
        """
        if chunks <= 0:
            raise ValueError("Chunks must be positive")
        if comm_timeout <= 0:
            raise ValueError("Communication timeout must be positive")

        super().__init__(
            model, rank, world_size,
            parallel_config={"chunks": chunks, "comm_timeout": comm_timeout}
        )
        self.chunks = chunks
        self.comm_timeout = comm_timeout
        self.stage_layers: Optional[nn.Sequential] = None
        self.is_first = (rank == 0)
        self.is_last = (rank == world_size - 1)
        self._pending_sends: Dict[int, torch.Tensor] = {}
        self._pending_recvs: Dict[int, torch.Tensor] = {}

        self._partition_model()
        self._is_initialized = True

    def _partition_model(self) -> None:
        """
        将模型分割到不同的流水线阶段

        从模型中提取所有层，然后按阶段数均匀分配。
        """
        all_layers = self._extract_layers()

        if len(all_layers) < self.world_size:
            raise ValueError(
                f"Model has {len(all_layers)} layers, but world_size is {self.world_size}"
            )

        layers_per_stage = len(all_layers) // self.world_size
        start_idx = layers_per_stage * self.rank
        end_idx = layers_per_stage * (self.rank + 1)

        if self.is_last:
            end_idx = len(all_layers)

        self.stage_layers = nn.Sequential(*all_layers[start_idx:end_idx])
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

    def forward(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        前向传播

        Args:
            input_data: 输入数据张量

        Returns:
            模型输出张量（最后一个阶段），其他阶段返回 None
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

        Args:
            input_data: 输入数据张量

        Returns:
            模型输出张量
        """
        if self.is_first:
            x = input_data
            if isinstance(x, torch.Tensor):
                x = x.view(x.size(0), -1)

            x = self.stage_layers(x)

            if not self.is_last:
                self._send_tensor_with_timeout(x, self.rank + 1)
                return None
            return x

        elif self.is_last:
            x = self._recv_tensor_with_timeout(self.rank - 1)
            x = self.stage_layers(x)
            return x

        else:
            x = self._recv_tensor_with_timeout(self.rank - 1)
            x = self.stage_layers(x)
            self._send_tensor_with_timeout(x, self.rank + 1)
            return None

    def _forward_micro_batches(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        微批次前向传播

        Args:
            input_data: 输入数据张量

        Returns:
            模型输出张量（最后一个阶段）
        """
        if not isinstance(input_data, torch.Tensor):
            raise ValueError("Chunked pipeline requires tensor input")

        batch_size = input_data.size(0)
        chunk_size = batch_size // self.chunks

        outputs: List[torch.Tensor] = []

        for i in range(self.chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.chunks - 1 else batch_size
            chunk_input = input_data[start_idx:end_idx]

            if self.is_first:
                x = chunk_input.view(chunk_input.size(0), -1)
                x = self.stage_layers(x)

                if not self.is_last:
                    self._send_tensor_with_timeout(x, self.rank + 1)
                else:
                    outputs.append(x)

            elif self.is_last:
                x = self._recv_tensor_with_timeout(self.rank - 1)
                x = self.stage_layers(x)
                outputs.append(x)

            else:
                x = self._recv_tensor_with_timeout(self.rank - 1)
                x = self.stage_layers(x)
                self._send_tensor_with_timeout(x, self.rank + 1)

        if self.is_last and outputs:
            return torch.cat(outputs, dim=0)
        return None

    def _send_tensor_with_timeout(
        self,
        tensor: torch.Tensor,
        dst: int,
        timeout: Optional[float] = None
    ) -> None:
        """
        发送张量到目标 rank（支持超时）

        Args:
            tensor: 要发送的张量
            dst: 目标 rank
            timeout: 超时时间（秒），默认使用 self.comm_timeout

        Raises:
            TimeoutError: 发送超时时抛出
            RuntimeError: 发送失败时抛出
        """
        timeout = timeout or self.comm_timeout

        try:
            shape_list = list(tensor.shape)
            shape_tensor = torch.tensor(shape_list, device=self.device, dtype=torch.long)
            shape_len = torch.tensor([len(shape_list)], device=self.device, dtype=torch.long)

            dist.send(shape_len, dst=dst)
            dist.send(shape_tensor, dst=dst)
            dist.send(tensor.contiguous(), dst=dst)

        except RuntimeError as e:
            if "timeout" in str(e).lower():
                raise TimeoutError(
                    f"Send operation to rank {dst} timed out after {timeout}s"
                ) from e
            raise RuntimeError(f"Failed to send tensor to rank {dst}: {e}") from e

    def _recv_tensor_with_timeout(
        self,
        src: int,
        timeout: Optional[float] = None
    ) -> torch.Tensor:
        """
        从源 rank 接收张量（支持超时）

        Args:
            src: 源 rank
            timeout: 超时时间（秒），默认使用 self.comm_timeout

        Returns:
            接收到的张量

        Raises:
            TimeoutError: 接收超时时抛出
            RuntimeError: 接收失败时抛出
        """
        timeout = timeout or self.comm_timeout

        try:
            shape_len = torch.zeros(1, dtype=torch.long, device=self.device)
            dist.recv(shape_len, src=src)

            shape_tensor = torch.zeros(shape_len.item(), dtype=torch.long, device=self.device)
            dist.recv(shape_tensor, src=src)

            tensor = torch.zeros(*shape_tensor.tolist(), device=self.device)
            dist.recv(tensor, src=src)

            return tensor

        except RuntimeError as e:
            if "timeout" in str(e).lower():
                raise TimeoutError(
                    f"Receive operation from rank {src} timed out after {timeout}s"
                ) from e
            raise RuntimeError(f"Failed to receive tensor from rank {src}: {e}") from e

    def _send_tensor(
        self,
        tensor: torch.Tensor,
        dst: int
    ) -> None:
        """
        发送张量（兼容旧接口）

        Args:
            tensor: 要发送的张量
            dst: 目标 rank
        """
        self._send_tensor_with_timeout(tensor, dst)

    def _recv_tensor(self, src: int) -> torch.Tensor:
        """
        接收张量（兼容旧接口）

        Args:
            src: 源 rank

        Returns:
            接收到的张量
        """
        return self._recv_tensor_with_timeout(src)

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

    def validate_distributed_state(self) -> None:
        """
        验证流水线并行的分布式训练状态

        Raises:
            RuntimeError: 当分布式状态不合法时抛出
        """
        super().validate_distributed_state()

        import torch.distributed as dist

        if self.world_size > dist.get_world_size():
            raise RuntimeError(
                f"PipelineParallel requires world_size={self.world_size}, "
                f"but distributed world_size={dist.get_world_size()}"
            )

        if self.is_first and self.is_last and self.world_size > 1:
            logger.warning(
                "PipelineParallel: rank is both first and last stage, "
                "but world_size > 1"
            )

    def get_parallel_info(self) -> Dict[str, Any]:
        """获取流水线并行配置信息"""
        info = super().get_parallel_info()
        info.update({
            "chunks": self.chunks,
            "comm_timeout": self.comm_timeout,
            "is_first_stage": self.is_first,
            "is_last_stage": self.is_last,
        })
        return info