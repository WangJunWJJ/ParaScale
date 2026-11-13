# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Torch-free, canonical quantization configuration."""

from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional


@dataclass
class QuantizationConfig:
    """Configuration for float-compute reference PTQ/QAT."""

    enabled: bool = False
    mode: Literal["qat", "ptq"] = "qat"
    bits: int = 8
    scheme: Literal["symmetric", "asymmetric"] = "symmetric"
    per_channel: bool = True
    observer_type: Literal["minmax", "moving_average"] = "minmax"
    moving_average_ratio: float = 0.9
    fuse_modules: bool = False
    qat_epochs: int = 10
    calib_batches: int = 100
    backend: str = "reference"
    quantizable_layers: Optional[List[str]] = None
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    quantize_activations: bool = False

    def __post_init__(self):
        for name in ("enabled", "per_channel", "quantize_activations", "fuse_modules"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.fuse_modules:
            raise ValueError(
                "automatic fusion is unsupported by reference quantization"
            )
        if self.backend != "reference":
            raise ValueError("only the reference quantization backend is supported")
        if self.bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if self.mode not in ("qat", "ptq"):
            raise ValueError("mode must be qat or ptq")
        if self.scheme not in ("symmetric", "asymmetric"):
            raise ValueError("unsupported quantization scheme")
        if self.observer_type not in ("minmax", "moving_average"):
            raise ValueError("unsupported observer type")
        if not 0 <= self.moving_average_ratio <= 1:
            raise ValueError("moving_average_ratio must be in [0, 1]")
        if self.qat_epochs < 1 or self.calib_batches < 1:
            raise ValueError("epoch and calibration counts must be positive")
        if self.quantizable_layers is None:
            self.quantizable_layers = ["Linear", "Conv2d"]
        if not set(self.quantizable_layers) <= {"Linear", "Conv2d"}:
            raise ValueError("only Linear and Conv2d selections are supported")
        if not isinstance(self.exclude_modules, list) or not all(
            isinstance(name, str) and name for name in self.exclude_modules
        ):
            raise ValueError("exclude_modules must be a list of nonempty names")

    def get_qmin_qmax(self) -> tuple:
        return -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "QuantizationConfig":
        return cls(**config_dict)
