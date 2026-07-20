# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""GroundDinoSpec configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .common import _section


@dataclass
class GroundDinoSpec:
    model_path: str = "IDEA-Research/grounding-dino-tiny"
    data_type: str = "objects365_yolo_cache"
    data_dir: str | None = None
    image_dir: str | None = None
    label_dir: str | None = None
    annotation_dir: str | None = None
    prompt: str = "object"
    loss_type: str = "proxy"
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False
    tensor_cache: bool = False
    tensor_cache_dir: str | None = None
    image_size: int = 800
    num_samples: int = 128
    batch_size: int = 2
    num_batches: int = 2
    lr: float = 0.00001
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_config(cls, config_data: Dict[str, Any]) -> "GroundDinoSpec":
        parascale = _section(config_data, "parascale")
        training = _section(config_data, "training")
        model = _section(config_data, "model")
        data = _section(config_data, "data")
        task = _section(config_data, "task")
        optimizer = _section(config_data, "optimizer")
        runtime = _section(config_data, "runtime")
        data_dir = data.get("data_dir", data.get("data_path"))
        image_dir = data.get("image_dir")
        label_dir = data.get("label_dir")
        if data_dir is None and image_dir is None:
            raise ValueError(
                "GroundDINO workload requires data.data_dir or data.image_dir."
            )
        batch_size = int(
            data.get(
                "batch_size", training.get("batch_size", parascale.get("batch_size", 2))
            )
        )
        num_batches = int(
            training.get("max_steps", training.get("num_batches", 2)) or 2
        )
        return cls(
            model_path=str(
                model.get(
                    "pretrained_model_name_or_path",
                    model.get("path", model.get("model_path", cls.model_path)),
                )
            ),
            data_type=str(data.get("type", "objects365_yolo_cache")),
            data_dir=str(data_dir) if data_dir is not None else None,
            image_dir=str(image_dir) if image_dir is not None else None,
            label_dir=str(label_dir) if label_dir is not None else None,
            annotation_dir=(
                str(data.get("annotation_dir"))
                if data.get("annotation_dir") is not None
                else None
            ),
            prompt=str(task.get("prompt", data.get("prompt", "object"))),
            loss_type=str(
                training.get(
                    "loss_type", data.get("loss_type", model.get("loss_type", "proxy"))
                )
            ).lower(),
            num_workers=int(
                data.get("num_workers", parascale.get("dataloader_num_workers", 0)) or 0
            ),
            pin_memory=bool(
                data.get("pin_memory", parascale.get("dataloader_pin_memory", True))
            ),
            prefetch_factor=int(
                data.get(
                    "prefetch_factor", parascale.get("dataloader_prefetch_factor", 2)
                )
                or 2
            ),
            persistent_workers=bool(
                data.get(
                    "persistent_workers",
                    parascale.get("dataloader_persistent_workers", False),
                )
            ),
            tensor_cache=bool(
                data.get(
                    "tensor_cache",
                    data.get(
                        "enable_tensor_cache", parascale.get("tensor_cache", False)
                    ),
                )
            ),
            tensor_cache_dir=(
                str(data.get("tensor_cache_dir", parascale.get("tensor_cache_dir")))
                if data.get("tensor_cache_dir", parascale.get("tensor_cache_dir"))
                is not None
                else None
            ),
            image_size=int(data.get("image_size", model.get("image_size", 800))),
            num_samples=int(
                data.get("num_samples", max(batch_size * num_batches, batch_size))
            ),
            batch_size=batch_size,
            num_batches=num_batches,
            lr=float(optimizer.get("lr", training.get("lr", 0.00001))),
            seed=int(training.get("seed", data.get("seed", 42))),
            device=str(runtime.get("device", training.get("device", "auto"))),
        )


__all__ = ["GroundDinoSpec"]
