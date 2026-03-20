# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : base.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 量化配置基类模块

本模块定义了量化训练的基础配置类，
支持多种量化方案和配置选项。
"""

from dataclasses import dataclass
from typing import Literal, Optional, List


@dataclass
class QuantizationConfig:
    """
    量化配置类
    
    用于配置量化感知训练（QAT）和训练后量化（PTQ）的参数。
    
    Attributes:
        enabled: 是否启用量化
        mode: 量化模式，支持 "qat"（量化感知训练）、"ptq"（训练后量化）
        bits: 量化位数，支持 8 或 4
        scheme: 量化方案，"symmetric"（对称）或 "asymmetric"（非对称）
        per_channel: 是否逐通道量化
        observer_type: 观察器类型，"minmax" 或 "moving_average"
        moving_average_ratio: 移动平均比例，仅用于 moving_average 观察器
        fuse_modules: 是否融合模块（Conv+BN+ReLU）
        qat_epochs: QAT 训练轮数
        calib_batches: PTQ 校准批次数量
        backend: 量化后端，"fbgemm"（x86）或 "qnnpack"（ARM）
    
    Example:
        >>> config = QuantizationConfig(
        ...     enabled=True,
        ...     mode="qat",
        ...     bits=8,
        ...     scheme="symmetric"
        ... )
    """
    
    # 基础配置
    enabled: bool = False
    mode: Literal["qat", "ptq"] = "qat"
    bits: int = 8
    
    # 量化方案
    scheme: Literal["symmetric", "asymmetric"] = "symmetric"
    per_channel: bool = True
    
    # 观察器配置
    observer_type: Literal["minmax", "moving_average"] = "minmax"
    moving_average_ratio: float = 0.9
    
    # 模块融合
    fuse_modules: bool = True
    
    # 训练配置
    qat_epochs: int = 10
    calib_batches: int = 100
    
    # 后端配置
    backend: Literal["fbgemm", "qnnpack"] = "fbgemm"
    
    # 需要量化的层类型
    quantizable_layers: Optional[List[str]] = None
    
    def __post_init__(self):
        """验证配置参数"""
        if self.bits not in [4, 8]:
            raise ValueError(f"只支持 4 位或 8 位量化，当前: {self.bits}")
        
        if self.moving_average_ratio < 0 or self.moving_average_ratio > 1:
            raise ValueError("移动平均比例必须在 0 到 1 之间")
        
        if self.qat_epochs < 1:
            raise ValueError("QAT 训练轮数必须大于 0")
        
        if self.calib_batches < 1:
            raise ValueError("校准批次数量必须大于 0")
        
        # 默认量化的层类型
        if self.quantizable_layers is None:
            self.quantizable_layers = [
                "Conv2d",
                "Linear",
                "ConvTranspose2d",
            ]
    
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
            "moving_average_ratio": self.moving_average_ratio,
            "fuse_modules": self.fuse_modules,
            "qat_epochs": self.qat_epochs,
            "calib_batches": self.calib_batches,
            "backend": self.backend,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "QuantizationConfig":
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
        
        Returns:
            QuantizationConfig 实例
        """
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
            backend=config_dict.get("backend", "fbgemm"),
        )
