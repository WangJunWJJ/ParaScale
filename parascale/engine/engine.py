# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : engine.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 核心引擎模块

本模块实现了 ParaScale 框架的核心引擎，负责协调各种并行策略和训练过程。
"""

import logging
import os
import sys
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from ..config import ParaScaleConfig
from ..optimizers import ZeroOptimizer
from ..parallel import (
    DataParallel,
    ModelParallel,
    PipelineParallel,
    TensorParallel,
)
from ..quantization import (
    QuantizationAwareTraining,
    QuantizationConfig,
    print_quantization_info,
)
from ..utils.distributed_utils import (
    cleanup_distributed,
    initialize_distributed,
    print_distributed_info,
)
from ..utils.utils import (
    ensure_directory,
    get_local_rank,
    get_node_rank,
    get_num_nodes,
    get_rank,
    get_world_size,
    print_rank_0,
    setup_logging,
)

# 获取日志记录器
logger = logging.getLogger(__name__)


class Engine:
    """
    ParaScale 核心引擎类

    负责协调各种并行策略和训练过程，提供统一的训练、评估和检查点管理接口。
    支持多节点分布式训练，自动检测并初始化 torchrun、SLURM、MPI 等环境。

    Attributes:
        model: PyTorch 模型实例
        optimizer: 优化器实例
        config: ParaScaleConfig 配置实例
        rank: 当前进程的 rank
        world_size: 世界大小（进程总数）
        local_rank: 本地 rank（当前节点内的 GPU 编号）
        node_rank: 节点编号
        num_nodes: 节点总数
        data_parallel: 数据并行实例
        model_parallel: 模型并行实例
        tensor_parallel: 张量并行实例
        pipeline_parallel: 流水线并行实例
        global_step: 全局训练步数

    Example:
        >>> # 自动检测环境并初始化
        >>> model = SimpleModel()
        >>> optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        >>> config = ParaScaleConfig(data_parallel_size=2)
        >>> engine = Engine(model, optimizer, config)
        >>> engine.train(dataloader, epochs=10)
        >>>
        >>> # 手动初始化分布式环境
        >>> initialize_distributed()
        >>> engine = Engine(model, optimizer, config)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: Optional[ParaScaleConfig] = None,
        auto_init_distributed: bool = True,
    ):
        """
        初始化 ParaScale 引擎

        Args:
            model: PyTorch 模型实例
            optimizer: PyTorch 优化器实例
            config: ParaScaleConfig 配置实例，如果为 None 则使用默认配置
            auto_init_distributed: 是否自动初始化分布式环境，默认为 True
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or ParaScaleConfig()

        # 自动初始化分布式环境
        if auto_init_distributed and not dist.is_initialized():
            initialize_distributed()

        # 获取分布式信息
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()
        self.node_rank = get_node_rank()
        self.num_nodes = get_num_nodes()

        # 打印分布式信息
        print_distributed_info()

        # 初始化并行策略实例
        self.data_parallel: Optional[DataParallel] = None
        self.model_parallel: Optional[ModelParallel] = None
        self.tensor_parallel: Optional[TensorParallel] = None
        self.pipeline_parallel: Optional[PipelineParallel] = None

        # 初始化量化相关
        self.qat_handler: Optional[QuantizationAwareTraining] = None
        self.is_quantized: bool = False

        # 初始化训练状态
        self.global_step = 0

        # 配置量化
        self._configure_quantization()

        # 配置并行策略和优化器
        self._configure_parallelism()
        self._configure_optimizer()

    def _configure_quantization(self) -> None:
        """
        根据配置初始化量化

        如果启用了量化感知训练，则准备模型进行 QAT。
        """
        if self.config.quantization.enabled:
            print_rank_0("启用量化感知训练 (QAT)")

            # 创建量化配置
            quant_config = QuantizationConfig(
                enabled=True,
                bits=self.config.quantization.bits,
                scheme=self.config.quantization.scheme,
                per_channel=self.config.quantization.per_channel,
                observer_type=self.config.quantization.observer_type,
                fuse_modules=self.config.quantization.fuse_modules,
                qat_epochs=self.config.quantization.qat_epochs,
            )

            # 准备 QAT 模型
            self.qat_handler = QuantizationAwareTraining(self.model, quant_config)
            self.model = self.qat_handler.prepare()
            self.is_quantized = True

            # 将量化后的模型转移到正确的设备
            if torch.cuda.is_available():
                self.model = self.model.to(f"cuda:{self.local_rank}")

            print_rank_0(
                f"量化配置: {quant_config.bits}位, {quant_config.scheme}量化"
            )
            print_quantization_info(self.model)

    def _configure_parallelism(self) -> None:
        """
        根据配置初始化并行策略

        根据配置中的并行大小参数，创建相应的并行策略实例。
        支持数据并行、模型并行、张量并行和流水线并行。
        """
        # 配置数据并行
        if self.config.data_parallel_size > 1:
            self.data_parallel = DataParallel(
                self.model,
                self.rank,
                self.config.data_parallel_size,
            )

        # 配置模型并行
        if self.config.model_parallel_size > 1:
            self.model_parallel = ModelParallel(
                self.model,
                self.rank,
                self.config.model_parallel_size,
            )
            self.model_parallel._split_model()

        # 配置张量并行
        if self.config.tensor_parallel_size > 1:
            self.tensor_parallel = TensorParallel(
                self.model,
                self.rank,
                self.config.tensor_parallel_size,
                self.config.tensor_parallel_mode,
            )

        # 配置流水线并行
        if self.config.pipeline_parallel_size > 1:
            self.pipeline_parallel = PipelineParallel(
                self.model,
                self.rank,
                self.config.pipeline_parallel_size,
                self.config.pipeline_parallel_chunks,
            )

    def _configure_optimizer(self) -> None:
        """
        根据配置初始化优化器

        如果启用了 ZeRO 优化，则使用 ZeroOptimizer 包装原始优化器。
        """
        if self.config.zero_optimization:
            self.optimizer = ZeroOptimizer(
                self.model,
                self.optimizer,
                self.config.zero_stage,
                self.config.zero_offload,
            )

    def train(self, dataloader: Any, epochs: int = 1) -> None:
        """
        训练模型

        执行指定轮数的模型训练，支持梯度累积和检查点保存。

        Args:
            dataloader: 数据加载器，提供训练数据
            epochs: 训练轮数，默认为 1
        """
        self.model.train()

        for epoch in range(epochs):
            print_rank_0(f"Epoch {epoch + 1}/{epochs}")

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # 将目标标签移动到当前设备
                if torch.cuda.is_available():
                    targets = targets.to(f"cuda:{self.local_rank}")

                # 执行前向传播
                outputs = self._forward(inputs)

                # 计算损失并反向传播
                if outputs is not None:
                    loss = nn.functional.cross_entropy(outputs, targets)
                    loss.backward()

                    # 梯度累积
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        # 收集梯度（用于数据并行和张量并行）
                        self._gather_gradients()

                        # 更新参数
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # 更新全局步数
                        self.global_step += 1

                        # 定期打印训练信息
                        if self.global_step % 100 == 0:
                            print_rank_0(
                                f"Step {self.global_step}, Loss: {loss.item():.4f}"
                            )

                        # 定期保存检查点
                        if self.global_step % self.config.checkpoint_save_interval == 0:
                            self.save_checkpoint()

    def _forward(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        执行前向传播

        根据配置的并行策略，选择合适的前向传播方式。
        优先级：流水线并行 > 数据并行 > 模型并行 > 张量并行 > 普通前向传播

        Args:
            inputs: 输入数据张量

        Returns:
            模型输出张量，对于非最后阶段的流水线并行可能返回 None
        """
        if self.pipeline_parallel:
            return self.pipeline_parallel.forward(inputs)
        elif self.data_parallel:
            return self.data_parallel.forward(inputs)
        elif self.model_parallel:
            return self.model_parallel.forward(inputs)
        elif self.tensor_parallel:
            return self.tensor_parallel.forward(inputs)
        else:
            # 无并行策略时的普通前向传播
            if torch.cuda.is_available():
                inputs = inputs.to(f"cuda:{self.local_rank}")
            return self.model(inputs)

    def _gather_gradients(self) -> None:
        """
        收集梯度

        在数据并行和张量并行模式下，需要在参数更新前收集所有进程的梯度。
        """
        if self.data_parallel:
            self.data_parallel.gather_gradients()
        elif self.tensor_parallel:
            self.tensor_parallel.gather_gradients()

    def evaluate(self, dataloader: Any) -> Tuple[float, float]:
        """
        评估模型

        在测试数据集上评估模型性能，计算平均损失和准确率。

        Args:
            dataloader: 数据加载器，提供测试数据

        Returns:
            元组 (平均损失, 准确率百分比)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # 判断是否为流水线并行的最后一个阶段
        is_pipeline_last = (
            self.pipeline_parallel is not None and self.pipeline_parallel.is_last
        )
        is_pipeline_worker = self.pipeline_parallel is not None

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # 将目标标签移动到当前设备（使用 local_rank 而不是 rank）
                if torch.cuda.is_available():
                    targets = targets.to(f"cuda:{self.local_rank}")

                # 执行前向传播
                outputs = self._forward(inputs)

                # 计算损失和准确率
                # 对于流水线并行，只有最后一个阶段计算损失
                if is_pipeline_last:
                    if outputs is not None:
                        loss = nn.functional.cross_entropy(
                            outputs, targets, reduction="sum"
                        )
                        total_loss += loss.item()

                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                elif not is_pipeline_worker:
                    # 非流水线并行模式
                    if outputs is not None:
                        loss = nn.functional.cross_entropy(
                            outputs, targets, reduction="sum"
                        )
                        total_loss += loss.item()

                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

        # 计算并返回评估结果
        if total > 0:
            accuracy = 100.0 * correct / total
            avg_loss = total_loss / total

            # 只在最后一个阶段打印评估结果
            if is_pipeline_last or not is_pipeline_worker:
                logger.info(
                    f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                )
            return avg_loss, accuracy
        else:
            if is_pipeline_last or not is_pipeline_worker:
                logger.info("No data evaluated")
            return 0.0, 0.0

    def save_checkpoint(self, save_path: Optional[str] = None) -> None:
        """
        保存检查点

        将模型状态、优化器状态、训练步数和配置保存到文件。

        Args:
            save_path: 检查点保存路径，如果为 None 则使用配置中的路径
        """
        save_path = save_path or self.config.checkpoint_save_path
        ensure_directory(save_path)

        checkpoint_path = os.path.join(
            save_path, f"checkpoint_{self.global_step}.pt"
        )

        # 构建检查点字典
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config.to_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
        print_rank_0(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点

        从文件恢复模型状态、优化器状态、训练步数和配置。

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=f"cuda:{self.local_rank}"
        )

        # 恢复模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # 恢复优化器状态
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 恢复全局步数
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]

        # 恢复配置
        if "config" in checkpoint:
            self.config.update(checkpoint["config"])

        print_rank_0(f"Checkpoint loaded from {checkpoint_path}")

    def freeze_quantization_observer(self) -> None:
        """
        冻结量化观察器

        停止收集统计信息，固定量化参数。
        通常在 QAT 训练的最后几个 epoch 调用。
        """
        if self.qat_handler is not None:
            self.qat_handler.freeze_observer()
            print_rank_0("量化观察器已冻结")

    def unfreeze_quantization_observer(self) -> None:
        """
        解冻量化观察器

        恢复收集统计信息。
        """
        if self.qat_handler is not None:
            self.qat_handler.unfreeze_observer()
            print_rank_0("量化观察器已解冻")

    def export_quantized_model(self, save_path: str) -> None:
        """
        导出量化模型

        将 QAT 训练好的模型导出为可以在推理时使用的量化模型。

        Args:
            save_path: 保存路径
        """
        if not self.is_quantized:
            print_rank_0("警告: 模型未启用量化，无法导出量化模型")
            return

        # 冻结观察器
        self.freeze_quantization_observer()

        # 转换模型
        quantized_model = self.qat_handler.convert()

        # 保存模型
        ensure_directory(os.path.dirname(save_path))
        torch.save(
            {
                "model_state_dict": quantized_model.state_dict(),
                "quantization_params": self.qat_handler.get_quantization_params(),
                "config": self.config.to_dict(),
            },
            save_path,
        )

        print_rank_0(f"量化模型已导出到: {save_path}")

    def get_quantization_info(self) -> dict:
        """
        获取量化信息

        Returns:
            量化信息字典
        """
        if not self.is_quantized:
            return {"enabled": False}

        return {
            "enabled": True,
            "bits": self.config.quantization.bits,
            "scheme": self.config.quantization.scheme,
            "per_channel": self.config.quantization.per_channel,
            "observer_type": self.config.quantization.observer_type,
            "quantization_params": (
                self.qat_handler.get_quantization_params()
                if self.qat_handler
                else {}
            ),
        }
