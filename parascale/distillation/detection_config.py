# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Torch-free settings for same-family detection feature distillation."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real


@dataclass
class DetectionDistillationConfig:
    """Runtime routing flag and nonnegative, additive neck-feature loss weight.

    ``enabled`` controls runtime routing, not an explicitly constructed objective.
    """

    enabled: bool = False
    task: str = "detection"
    feature_weight: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        if not isinstance(self.task, str):
            raise TypeError("task must be a string")
        if self.task != "detection":
            raise ValueError("task must be detection")
        if isinstance(self.feature_weight, bool) or not isinstance(
            self.feature_weight, Real
        ):
            raise TypeError("feature_weight must be a finite number")
        if not math.isfinite(self.feature_weight) or self.feature_weight < 0:
            raise ValueError("feature_weight must be finite and nonnegative")

    def to_dict(self):
        self.validate()
        return {
            "enabled": self.enabled,
            "task": self.task,
            "feature_weight": float(self.feature_weight),
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise TypeError("detection distillation config must be a mapping")
        return cls(**dict(data))
