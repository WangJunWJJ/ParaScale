# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : config.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 配置管理模块

本模块使用 dataclass 实现 ParaScale 框架的配置管理，
支持自动参数验证和序列化/反序列化功能。
"""

from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional, List, Tuple
import warnings


class ConfigValidationError(ValueError):
    """配置验证错误
    
    当配置参数不一致或不合法时抛出此异常。
    """
    pass


@dataclass
class QuantizationConfig:
    """
    量化配置类
    
    用于配置量化感知训练练（QAT）和训练后量化（PTQ）的参数。
    
    Attributes:
        enabled: 是否启用量化
        mode: 量化模式，"qat"（量化感知训练）或 "ptq"（训练后量化）
        bits: 量化位数，支持 8 或 4
        scheme: 量化方案，"symmetric"（对称）或 "asymmetric"（非对称）
        per_channel: 是否逐通道量化
        observer_type: 观察器类型，"minmax" 或 "moving_average"
        moving_average_ratio: 移动平均观察器的比率（仅用于 moving_average 观察器）
        fuse_modules: 是否融合模块（Conv+BN+ReLU）
        qat_epochs: QAT 训练轮数
        calib_batches: PTQ 校准批次数量
    """
    enabled: bool = False
    mode: Literal["qat", "ptq"] = "qat"
    bits: int = 8
    scheme: Literal["symmetric", "asymmetric"] = "symmetric"
    per_channel: bool = True
    observer_type: Literal["minmax", "moving_average"] = "minmax"
    moving_average_ratio: float = 0.9
    fuse_modules: bool = True
    qat_epochs: int = 10
    calib_batches: int = 100
    
    def __post_init__(self):
        """验证配置参数"""
        self._validate()
    
    def _validate(self) -> None:
        """
        验证配置参数的合法性
        
        Raises:
            ConfigValidationError: 当配置参数不合法时抛出
        """
        if self.bits not in [4, 8]:
            raise ConfigValidationError(f"只支持 4 位或 8 位量化，当前: {self.bits}")
        if self.qat_epochs < 1:
            raise ConfigValidationError("QAT 训练轮数必须大于 0")
        if self.calib_batches < 1:
            raise ConfigValidationError("校准批次数量必须大于 0")
        if self.moving_average_ratio <= 0 or self.moving_average_ratio >= 1:
            raise ConfigValidationError("moving_average_ratio 必须在 (0, 1) 之间")
        
        # 跨参数一致性检查
        self._validate_cross_params()
    
    def _validate_cross_params(self) -> None:
        """
        跨参数一致性检查
        
        检查量化配置中相关参数之间的逻辑一致性。
        """
        # 检查 observer_type 与 moving_average_ratio 的一致性
        if self.observer_type == "minmax" and self.moving_average_ratio != 0.9:
            warnings.warn(
                "observer_type is 'minmax' but moving_average_ratio is set. "
                "moving_average_ratio is only used when observer_type='moving_average'. "
                "Consider setting observer_type='moving_average' or removing moving_average_ratio.",
                UserWarning
            )
        
        # 检查 mode 与 qat_epochs/calib_batches 的一致性
        if self.mode == "ptq" and self.qat_epochs != 10:
            warnings.warn(
                f"mode is 'ptq' but qat_epochs={self.qat_epochs}. "
                "qat_epochs is only used in QAT mode. "
                "Consider removing qat_epochs or setting mode='qat'.",
                UserWarning
            )
        
        if self.mode == "qat" and self.calib_batches != 100:
            warnings.warn(
                f"mode is 'qat' but calib_batches={self.calib_batches}. "
                "calib_batches is only used in PTQ mode. "
                "Consider removing calib_batches or setting mode='ptq'.",
                UserWarning
            )
        
        # 检查 bits 与 scheme 的兼容性
        if self.bits == 4 and self.scheme == "asymmetric":
            warnings.warn(
                "4-bit quantization with asymmetric scheme may have precision issues. "
                "Consider using symmetric scheme for 4-bit quantization.",
                UserWarning
            )

    def get_qmin_qmax(self) -> tuple:
        """
        获取量化的最小值和最大值

        Returns:
            元组 (qmin, qmax)
        """
        if self.bits == 8:
            return -128, 127  # INT8 范围
        elif self.bits == 4:
            return -8, 7      # INT4 范围
        else:
            raise ConfigValidationError(f"不支持的量化位数: {self.bits}")

    def to_dict(self) -> dict:
        """将配置转换为字典"""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "bits": self.bits,
            "scheme": self.scheme,
            "per_channel": self.per_channel,
            "observer_type": self.observer_type,
            "moving_average_ratio": self.moving_average_ratio,
            "fuse_modules": self.fuse_modules,
            "qat_epochs": self.qat_epochs,
            "calib_batches": self.calib_batches,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "QuantizationConfig":
        """从字典创建配置"""
        return cls(
            enabled=config_dict.get("enabled", False),
            mode=config_dict.get("mode", "qat"),
            bits=config_dict.get("bits", 8),
            scheme=config_dict.get("scheme", "symmetric"),
            per_channel=config_dict.get("per_channel", True),
            observer_type=config_dict.get("observer_type", "minmax"),
            moving_average_ratio=config_dict.get("moving_average_ratio", 0.9),
            fuse_modules=config_dict.get("fuse_modules", True),
            qat_epochs=config_dict.get("qat_epochs", 10),
            calib_batches=config_dict.get("calib_batches", 100),
        )


@dataclass
class ParaScaleConfig:
    """
    ParaScale 框架配置类
    
    使用 @dataclass 装饰器实现，支持自动参数验证。
    所有配置项都有默认值，可以通过构造函数或 update 方法修改。
    
    Attributes:
        data_parallel_size: 数据并行大小，表示使用多少个 GPU 进行数据并行
        model_parallel_size: 模型并行大小，表示模型分割到多少个设备
        tensor_parallel_size: 张量并行大小，表示张量分割到多少个 GPU
        tensor_parallel_mode: 张量并行模式，支持 "row"（行并行）或 "column"（列并行）
        pipeline_parallel_size: 流水线并行大小，表示流水线分割到多少个阶段
        pipeline_parallel_chunks: 流水线并行分块数，用于微批次处理
        zero_optimization: 是否启用 ZeRO 优化器
        zero_stage: ZeRO 优化器阶段（0, 1, 2, 3）
        zero_offload: 是否启用 ZeRO offload（将数据卸载到 CPU）
        batch_size: 训练批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        checkpoint_save_path: 检查点保存路径
        checkpoint_save_interval: 检查点保存间隔（步数）
    
    Example:
        >>> config = ParaScaleConfig(data_parallel_size=2, batch_size=64)
        >>> config.update({'learning_rate': 1e-4})
        >>> config.to_dict()
    """
    
    # 并行策略配置
    data_parallel_size: int = 1
    model_parallel_size: int = 1
    tensor_parallel_size: int = 1
    tensor_parallel_mode: Literal["row", "column"] = "row"
    pipeline_parallel_size: int = 1
    pipeline_parallel_chunks: int = 1
    
    # ZeRO 优化器配置
    zero_optimization: bool = False
    zero_stage: int = 0
    zero_offload: bool = False
    
    # 训练配置
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    
    # 检查点配置
    checkpoint_save_path: str = "./checkpoints"
    checkpoint_save_interval: int = 1000
    
    # 量化配置
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    def __post_init__(self) -> None:
        """
        dataclass 初始化后自动调用的验证方法
        
        在对象创建后自动执行参数验证，确保所有配置值合法。
        """
        self._validate()
    
    def _validate(self) -> None:
        """
        验证配置参数的合法性
        
        Raises:
            ConfigValidationError: 当任何配置参数不合法时抛出
        """
        # 验证并行大小参数
        if self.data_parallel_size < 1:
            raise ConfigValidationError("data_parallel_size must be >= 1")
        if self.model_parallel_size < 1:
            raise ConfigValidationError("model_parallel_size must be >= 1")
        if self.tensor_parallel_size < 1:
            raise ConfigValidationError("tensor_parallel_size must be >= 1")
        if self.pipeline_parallel_size < 1:
            raise ConfigValidationError("pipeline_parallel_size must be >= 1")
        if self.pipeline_parallel_chunks < 1:
            raise ConfigValidationError("pipeline_parallel_chunks must be >= 1")
        
        # 验证 ZeRO 阶段
        if self.zero_stage not in [0, 1, 2, 3]:
            raise ConfigValidationError("zero_stage must be 0, 1, 2, or 3")
        
        # 验证训练参数
        if self.batch_size < 1:
            raise ConfigValidationError("batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ConfigValidationError("gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigValidationError("learning_rate must be positive")
        if self.checkpoint_save_interval < 1:
            raise ConfigValidationError("checkpoint_save_interval must be >= 1")
        
        # 跨参数一致性检查
        self._validate_cross_params()
    
    def _validate_cross_params(self) -> None:
        """
        跨参数一致性检查
        
        检查相关参数之间的逻辑一致性，确保配置组合合理。
        
        Raises:
            ConfigValidationError: 当参数组合不一致时抛出
        """
        # 1. 检查并行策略一致性
        self._validate_parallel_strategy()
        
        # 2. 检查 ZeRO 配置一致性
        self._validate_zero_config()
        
        # 3. 检查量化配置一致性
        self._validate_quantization_config()
        
        # 4. 检查训练配置一致性
        self._validate_training_config()
    
    def _validate_parallel_strategy(self) -> None:
        """
        验证并行策略配置的一致性
        """
        # 计算总并行度
        total_parallel_size = (
            self.data_parallel_size * 
            self.tensor_parallel_size * 
            self.pipeline_parallel_size
        )
        
        # 检查是否存在多丮并行策略同时使用
        active_strategies = sum([
            1 if self.data_parallel_size > 1 else 0,
            1 if self.tensor_parallel_size > 1 else 0,
            1 if self.pipeline_parallel_size > 1 else 0,
            1 if self.model_parallel_size > 1 else 0
        ])
        
        if active_strategies > 2:
            warnings.warn(
                f"Using {active_strategies} parallel strategies simultaneously may cause "
                f"performance degradation. Consider using at most 2 strategies for optimal performance.",
                UserWarning
            )
        
        # 检查流水线并行与微批次的一致性
        if self.pipeline_parallel_size > 1 and self.pipeline_parallel_chunks < 2:
            raise ConfigValidationError(
                "When using pipeline parallelism, pipeline_parallel_chunks must be >= 2 "
                "to enable micro-batch processing and improve pipeline efficiency."
            )
        
        # 检查张量并行模式
        if self.tensor_parallel_size > 1 and self.tensor_parallel_mode not in ["row", "column"]:
            raise ConfigValidationError(
                f"tensor_parallel_mode must be 'row' or 'column', got '{self.tensor_parallel_mode}'"
            )
    
    def _validate_zero_config(self) -> None:
        """
        验证 ZeRO 配置的一致性
        """
        # 检查 ZeRO 优化与阶段的一致性
        if self.zero_stage > 0 and not self.zero_optimization:
            warnings.warn(
                f"zero_stage={self.zero_stage} but zero_optimization=False. "
                "ZeRO optimization will not be enabled. "
                "Consider setting zero_optimization=True.",
                UserWarning
            )
        
        if self.zero_optimization and self.zero_stage == 0:
            warnings.warn(
                "zero_optimization=True but zero_stage=0. "
                "ZeRO optimization is enabled but set to Stage 0 (disabled). "
                "Consider setting zero_stage to 1, 2, or 3.",
                UserWarning
            )
        
        # 检查 ZeRO offload 与阶段的一致性
        if self.zero_offload and self.zero_stage < 1:
            raise ConfigValidationError(
                "zero_offload=True requires zero_stage >= 1. "
                "CPU offload is only available when ZeRO is enabled."
            )
        
        # 检查 ZeRO 与数据并行的一致性
        if self.zero_stage >= 2 and self.data_parallel_size == 1:
            warnings.warn(
                f"Using ZeRO Stage {self.zero_stage} with data_parallel_size=1. "
                "ZeRO is designed for data parallelism. "
                "Consider increasing data_parallel_size for better memory efficiency.",
                UserWarning
            )
    
    def _validate_quantization_config(self) -> None:
        """
        验证量化配置的一致性
        """
        if not self.quantization.enabled:
            return
        
        # 检查量化与张量并行的兼容性
        if self.tensor_parallel_size > 1 and self.quantization.enabled:
            warnings.warn(
                "Quantization with tensor parallelism may have precision issues. "
                "Consider disabling quantization or using data parallelism only.",
                UserWarning
            )
        
        # 检查 QAT 训练轮数与检查点保存间隔的一致性
        if (self.quantization.enabled and 
            self.quantization.mode == "qat" and 
            self.checkpoint_save_interval < self.quantization.qat_epochs):
            warnings.warn(
                f"checkpoint_save_interval ({self.checkpoint_save_interval}) is less than "
                f"qat_epochs ({self.quantization.qat_epochs}). "
                "Consider increasing checkpoint_save_interval to avoid frequent saves during QAT.",
                UserWarning
            )
    
    def _validate_training_config(self) -> None:
        """
        验证训练配置的一致性
        """
        # 检查批次大小与梯度累积的一致性
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        if effective_batch_size > 1024 and self.gradient_accumulation_steps == 1:
            warnings.warn(
                f"Large effective batch size ({effective_batch_size}) without gradient accumulation "
                "may cause memory issues. Consider using gradient_accumulation_steps > 1.",
                UserWarning
            )
        
        # 检查学习率与批次大小的一致性
        if self.learning_rate > 0.01 and self.batch_size >= 256:
            warnings.warn(
                f"High learning rate ({self.learning_rate}) with large batch size ({self.batch_size}) "
                "may cause training instability. Consider reducing learning_rate or using learning rate warmup.",
                UserWarning
            )
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        获取配置验证报告
        
        Returns:
            包含验证结果和建议的字典
        """
        report = {
            "valid": True,
            "warnings": [],
            "suggestions": [],
            "config_summary": {
                "parallel_strategy": self._get_parallel_strategy_summary(),
                "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
                "memory_optimization": self._get_memory_optimization_summary(),
            }
        }
        
        # 检查并行策略建议
        if self.data_parallel_size > 1 and self.tensor_parallel_size > 1:
            report["suggestions"].append(
                "Consider using 3D parallelism (DP+TP+PP) for large models instead of just DP+TP."
            )
        
        # 检查内存优化建议
        if not self.zero_optimization and self.data_parallel_size >= 4:
            report["suggestions"].append(
                "Consider enabling ZeRO optimization (zero_optimization=True) for better memory efficiency."
            )
        
        # 检查量化建议
        if not self.quantization.enabled and self.model_parallel_size > 1:
            report["suggestions"].append(
                "Consider enabling quantization for model parallelism to reduce communication overhead."
            )
        
        return report
    
    def _get_parallel_strategy_summary(self) -> str:
        """获取并行策略摘要"""
        strategies = []
        if self.data_parallel_size > 1:
            strategies.append(f"DP={self.data_parallel_size}")
        if self.tensor_parallel_size > 1:
            strategies.append(f"TP={self.tensor_parallel_size}")
        if self.pipeline_parallel_size > 1:
            strategies.append(f"PP={self.pipeline_parallel_size}")
        
        return " + ".join(strategies) if strategies else "Single GPU"
    
    def _get_memory_optimization_summary(self) -> str:
        """获取内存优化摘要"""
        optimizations = []
        if self.zero_optimization:
            optimizations.append(f"ZeRO Stage {self.zero_stage}")
        if self.zero_offload:
            optimizations.append("CPU Offload")
        if self.quantization.enabled:
            optimizations.append(f"Quantization ({self.quantization.bits}-bit)")
        
        return " + ".join(optimizations) if optimizations else "None"
    
    def update(self, config_dict: Dict[str, Any]) -> "ParaScaleConfig":
        """
        从字典更新配置
        
        Args:
            config_dict: 包含配置项的字典，键为配置名，值为配置值
        
        Returns:
            更新后的配置实例（支持链式调用）
        
        Example:
            >>> config = ParaScaleConfig()
            >>> config.update({'batch_size': 64, 'learning_rate': 1e-4})
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._validate()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典
        
        Returns:
            包含所有配置项的字典
        
        Example:
            >>> config = ParaScaleConfig()
            >>> d = config.to_dict()
            >>> print(d['batch_size'])  # 32
        """
        return {
            "data_parallel_size": self.data_parallel_size,
            "model_parallel_size": self.model_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "tensor_parallel_mode": self.tensor_parallel_mode,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "pipeline_parallel_chunks": self.pipeline_parallel_chunks,
            "zero_optimization": self.zero_optimization,
            "zero_stage": self.zero_stage,
            "zero_offload": self.zero_offload,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "checkpoint_save_path": self.checkpoint_save_path,
            "checkpoint_save_interval": self.checkpoint_save_interval,
            "quantization": self.quantization.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ParaScaleConfig":
        """
        从字典创建配置实例
        
        Args:
            config_dict: 包含配置项的字典
        
        Returns:
            新创建的配置实例
        
        Example:
            >>> config = ParaScaleConfig.from_dict({'batch_size': 64})
        """
        # 处理 quantization 配置
        if 'quantization' in config_dict:
            quant_config = QuantizationConfig(**config_dict['quantization'])
            config_dict = config_dict.copy()
            config_dict['quantization'] = quant_config
        
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """
        返回配置的字符串表示
        
        Returns:
            配置的简要字符串表示
        """
        return (
            f"ParaScaleConfig("
            f"data_parallel_size={self.data_parallel_size}, "
            f"model_parallel_size={self.model_parallel_size}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, "
            f"tensor_parallel_mode={self.tensor_parallel_mode}, "
            f"pipeline_parallel_size={self.pipeline_parallel_size}, "
            f"zero_optimization={self.zero_optimization}, "
            f"zero_stage={self.zero_stage})"
        )
