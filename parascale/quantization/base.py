# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Quantization configuration primitives."""

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class QuantizationConfig:
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
    backend: Literal["fbgemm", "qnnpack"] = "fbgemm"
    quantizable_layers: Optional[List[str]] = None

    def __post_init__(self):
        if self.bits not in [4, 8]:
            raise ValueError(f"invalid quantization configuration{self.bits}")
        if self.moving_average_ratio < 0 or self.moving_average_ratio > 1:
            raise ValueError("invalid quantization configuration")
        if self.qat_epochs < 1:
            raise ValueError("invalid quantization configuration")
        if self.calib_batches < 1:
            raise ValueError("invalid quantization configuration")
        if self.quantizable_layers is None:
            self.quantizable_layers = ["Conv2d", "Linear", "ConvTranspose2d"]

    def get_qmin_qmax(self) -> tuple:
        if self.bits == 8:
            return (-128, 127)
        elif self.bits == 4:
            return (-8, 7)
        else:
            raise ValueError(f"invalid quantization configuration{self.bits}")

    def to_dict(self) -> dict:
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
