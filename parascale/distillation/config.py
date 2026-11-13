# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Torch-independent distillation settings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any


@dataclass
class DistillationConfig:
    """Settings with temperature squared restricted to normal finite FP32.

    Args:
        enabled: Runtime routing flag; does not disable the loss callable.
        task: Classification or shifted causal language modeling.
        temperature: Positive softmax temperature with FP32-safe square.
        alpha: Teacher KL weight in [0, 1].
        ignore_index: Label value excluded from both loss terms.
        input_keys: Unique teacher input names, excluding labels.

    Raises:
        TypeError: A setting has an invalid type.
        ValueError: A setting is outside its supported domain.
    """

    enabled: bool = False
    task: str = "classification"
    temperature: float = 2.0
    alpha: float = 0.5
    ignore_index: int = -100
    input_keys: tuple[str, ...] | list[str] = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "position_ids",
        "token_type_ids",
    )

    def __post_init__(self) -> None:
        self.validate()
        self.input_keys = tuple(self.input_keys)

    def validate(self) -> None:
        """Validate current settings without importing torch.

        Returns:
            None.
        Raises:
            TypeError: A setting has an invalid type.
            ValueError: A setting is outside its supported domain.
        """
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        if self.task not in ("classification", "causal_lm"):
            raise ValueError("task must be classification or causal_lm")
        for name in ("temperature", "alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        # IEEE-754 limits keep this validation independent of torch/numpy.
        if (
            not math.sqrt(2.0**-126)
            <= self.temperature
            <= math.sqrt((2.0 - 2.0**-23) * 2.0**127)
        ):
            raise ValueError("temperature squared must be normal and finite in FP32")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if type(self.ignore_index) is not int:
            raise TypeError("ignore_index must be int")
        if not isinstance(self.input_keys, (tuple, list)):
            raise TypeError("input_keys must be a tuple or list")
        if not self.input_keys or any(
            not isinstance(k, str) or not k.strip() or k == "labels"
            for k in self.input_keys
        ):
            raise ValueError(
                "input_keys must contain nonempty input names, excluding labels"
            )
        if len(set(self.input_keys)) != len(self.input_keys):
            raise ValueError("input_keys must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible settings mapping.

        Returns:
            Settings with input_keys serialized as a list.
        Raises:
            TypeError: A setting has an invalid type.
            ValueError: Current settings fail validation.
        """
        self.validate()
        return {
            "enabled": self.enabled,
            "task": self.task,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "ignore_index": self.ignore_index,
            "input_keys": list(self.input_keys),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DistillationConfig:
        """Construct validated settings from a mapping.

        Args:
            data: Distillation settings, without runtime teacher configuration.
        Returns:
            A new independent configuration.
        Raises:
            TypeError: Invalid mapping, setting type, or unknown field.
            ValueError: A setting is outside its supported domain.
        """
        if not isinstance(data, Mapping):
            raise TypeError("distillation config must be a mapping")
        return cls(**dict(data))
