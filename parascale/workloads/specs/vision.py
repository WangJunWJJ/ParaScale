# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""VisionSyntheticSpec configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .common import _section


@dataclass
class VisionSyntheticSpec:
    image_size: int = 32
    channels: int = 3
    patch_size: int = 16
    hidden_dim: int = 32
    num_classes: int = 10
    num_samples: int = 16
    batch_size: int = 4
    num_batches: int = 2
    max_patch_tokens_per_batch: int | None = None
    lr: float = 0.01
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_config(cls, config_data: Dict[str, Any]) -> "VisionSyntheticSpec":
        parascale = _section(config_data, "parascale")
        training = _section(config_data, "training")
        model = _section(config_data, "model")
        data = _section(config_data, "data")
        optimizer = _section(config_data, "optimizer")
        runtime = _section(config_data, "runtime")
        image_size = int(
            data.get(
                "image_size", model.get("image_size", training.get("image_size", 32))
            )
        )
        batch_size = int(
            data.get(
                "batch_size", training.get("batch_size", parascale.get("batch_size", 4))
            )
        )
        optimizer_steps = int(
            training.get("max_steps", training.get("num_batches", 2)) or 2
        )
        accumulation_steps = int(
            training.get(
                "gradient_accumulation_steps",
                parascale.get("gradient_accumulation_steps", 1),
            )
            or 1
        )
        num_batches = optimizer_steps * max(1, accumulation_steps)
        return cls(
            image_size=image_size,
            channels=int(data.get("channels", model.get("channels", 3))),
            patch_size=int(model.get("patch_size", data.get("patch_size", 16))),
            hidden_dim=int(model.get("hidden_dim", 32)),
            num_classes=int(model.get("num_classes", data.get("num_classes", 10))),
            num_samples=int(
                data.get("num_samples", max(batch_size * num_batches, batch_size))
            ),
            batch_size=batch_size,
            num_batches=num_batches,
            max_patch_tokens_per_batch=(
                int(parascale.get("max_patch_tokens_per_batch"))
                if parascale.get("max_patch_tokens_per_batch") is not None
                else None
            ),
            lr=float(optimizer.get("lr", training.get("lr", 0.01))),
            seed=int(training.get("seed", data.get("seed", 42))),
            device=str(runtime.get("device", training.get("device", "auto"))),
        )


__all__ = ["VisionSyntheticSpec"]
