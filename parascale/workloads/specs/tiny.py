# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""TinyTorchWorkloadSpec configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .common import _section


@dataclass
class TinyTorchWorkloadSpec:
    input_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 2
    batch_size: int = 2
    num_batches: int = 2
    lr: float = 0.01
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_config(cls, config_data: Dict[str, Any]) -> "TinyTorchWorkloadSpec":
        training = _section(config_data, "training")
        model = _section(config_data, "model")
        data = _section(config_data, "data")
        optimizer = _section(config_data, "optimizer")
        runtime = _section(config_data, "runtime")
        return cls(
            input_dim=int(
                model.get(
                    "input_dim", data.get("input_dim", training.get("input_dim", 4))
                )
            ),
            hidden_dim=int(model.get("hidden_dim", training.get("hidden_dim", 8))),
            output_dim=int(
                model.get(
                    "output_dim", data.get("output_dim", training.get("output_dim", 2))
                )
            ),
            batch_size=int(data.get("batch_size", training.get("batch_size", 2))),
            num_batches=int(
                training.get("max_steps", training.get("num_batches", 2)) or 2
            ),
            lr=float(optimizer.get("lr", training.get("lr", 0.01))),
            seed=int(training.get("seed", data.get("seed", 42))),
            device=str(runtime.get("device", training.get("device", "auto"))),
        )


__all__ = ["TinyTorchWorkloadSpec"]
