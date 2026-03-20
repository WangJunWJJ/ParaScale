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
from typing import Literal, Dict, Any, Optional


@dataclass
class QuantizationConfig:
    """
    量化配置类
    
    用于配置量化感知训练（QAT）和训练后量化（PTQ）的参数。
    
    Attributes:
        enabled: 是否启用量化
        mode: 量化模式，"qat"（量化感知训练）或 "ptq"（训练后量化）
        bits: 量化位数，支持 8 或 4
        scheme: 量化方案，"symmetric"（对称）或 "asymmetric"（非对称）
        per_channel: 是否逐通道量化
        observer_type: 观察器类型，"minmax" 或 "moving_average"
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
    fuse_modules: bool = True
    qat_epochs: int = 10
    calib_batches: int = 100
    
    def __post_init__(self):
        """验证配置参数"""
        if self.bits not in [4, 8]:
            raise ValueError(f"只支持 4 位或 8 位量化，当前: {self.bits}")
        if self.qat_epochs < 1:
            raise ValueError("QAT 训练轮数必须大于 0")

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
            raise ValueError(f"不支持的量化位数: {self.bits}")

    def to_dict(self) -> dict:
        """将配置转换为字典"""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "bits": self.bits,
            "scheme": self.scheme,
            "per_channel": self.per_channel,
            "observer_type": self.observer_type,
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
            ValueError: 当任何配置参数不合法时抛出
        """
        # 验证并行大小参数
        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
        if self.model_parallel_size < 1:
            raise ValueError("model_parallel_size must be >= 1")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        if self.pipeline_parallel_chunks < 1:
            raise ValueError("pipeline_parallel_chunks must be >= 1")
        
        # 验证 ZeRO 阶段
        if self.zero_stage not in [0, 1, 2, 3]:
            raise ValueError("zero_stage must be 0, 1, 2, or 3")
        
        # 验证训练参数
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.checkpoint_save_interval < 1:
            raise ValueError("checkpoint_save_interval must be >= 1")
    
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
