# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit optimizer-update schedules with serializable configuration identity."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SchedulerSpec:
    """Warm up from zero, then decay to a floor over a fixed update horizon."""

    type: str
    total_steps: int
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.type not in ("linear", "cosine"):
            raise ValueError("training.scheduler.type must be linear or cosine")
        for name, minimum in (("total_steps", 1), ("warmup_steps", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"training.scheduler.{name} must be an integer >= {minimum}"
                )
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                "training.scheduler.warmup_steps must be less than total_steps"
            )
        if (
            isinstance(self.min_lr_ratio, bool)
            or not isinstance(self.min_lr_ratio, (int, float))
            or not math.isfinite(self.min_lr_ratio)
            or not 0 <= self.min_lr_ratio <= 1
        ):
            raise ValueError(
                "training.scheduler.min_lr_ratio must be finite and in [0, 1]"
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SchedulerSpec | None:
        training = config.get("training", {})
        settings = training.get("scheduler") if isinstance(training, dict) else None
        if settings is None:
            return None
        if not isinstance(settings, dict):
            raise ValueError("training.scheduler must be a mapping or null")
        allowed = {"type", "total_steps", "warmup_steps", "min_lr_ratio"}
        if set(settings) - allowed or not {"type", "total_steps"} <= set(settings):
            raise ValueError(
                "training.scheduler requires type/total_steps and rejects unknown fields"
            )
        return cls(**settings)

    def validate_backend(self, backend: str) -> None:
        if backend not in {"native", "native_ddp"}:
            raise ValueError(
                "training.scheduler currently requires native or native_ddp"
            )

    def to_metadata(self) -> dict[str, Any]:
        return {"version": 1, "unit": "optimizer_update", **asdict(self)}

    def factor(self, completed_updates: int) -> float:
        """Return the multiplier for the next update, clamped after the horizon."""
        if completed_updates < self.warmup_steps:
            return completed_updates / self.warmup_steps
        progress = min(
            1.0,
            max(
                0.0,
                (completed_updates - self.warmup_steps)
                / (self.total_steps - self.warmup_steps),
            ),
        )
        decay = (
            1 - progress
            if self.type == "linear"
            else 0.5 * (1 + math.cos(math.pi * progress))
        )
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * decay

    def build(self, optimizer: Any) -> Any:
        """Bind a PyTorch schedule to the final backend optimizer before restore."""
        from torch.optim.lr_scheduler import LambdaLR

        return LambdaLR(optimizer, lr_lambda=lambda step: self.factor(step))
