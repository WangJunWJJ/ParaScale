# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""YoloWorldSpec configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .common import _section


@dataclass
class YoloWorldSpec:
    model_path: str
    data_zip: str | None = None
    data_dir: str | None = None
    image_dir: str | None = None
    label_dir: str | None = None
    loss_type: str = "proxy"
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False
    tensor_cache: bool = False
    tensor_cache_dir: str | None = None
    image_size: int = 640
    num_samples: int = 128
    batch_size: int = 2
    num_batches: int = 2
    lr: float = 0.0001
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_config(cls, config_data: Dict[str, Any]) -> "YoloWorldSpec":
        parascale = _section(config_data, "parascale")
        training = _section(config_data, "training")
        model = _section(config_data, "model")
        data = _section(config_data, "data")
        optimizer = _section(config_data, "optimizer")
        runtime = _section(config_data, "runtime")
        model_path = model.get("path", model.get("model_path"))
        data_zip = data.get("zip_path", data.get("data_zip"))
        data_dir = data.get("data_dir", data.get("data_path"))
        image_dir = data.get("image_dir")
        label_dir = data.get("label_dir")
        loss_type = str(
            training.get(
                "loss_type", data.get("loss_type", model.get("loss_type", "proxy"))
            )
        ).lower()
        if model_path is None:
            raise ValueError(
                "YOLO-World workload requires model.path or model.model_path."
            )
        if data_zip is None and image_dir is None and data_dir is None:
            raise ValueError(
                "YOLO-World workload requires data.zip_path, data.image_dir or data.data_dir."
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
            model_path=str(model_path),
            data_zip=str(data_zip) if data_zip is not None else None,
            data_dir=str(data_dir) if data_dir is not None else None,
            image_dir=str(image_dir) if image_dir is not None else None,
            label_dir=str(label_dir) if label_dir is not None else None,
            loss_type=loss_type,
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
                else (
                    str(
                        data.get(
                            "dataset_local_cache_dir",
                            parascale.get("dataset_local_cache_dir"),
                        )
                    )
                    if data.get(
                        "dataset_local_cache_dir",
                        parascale.get("dataset_local_cache_dir"),
                    )
                    is not None
                    else None
                )
            ),
            image_size=int(data.get("image_size", model.get("image_size", 640))),
            num_samples=int(
                data.get("num_samples", max(batch_size * num_batches, batch_size))
            ),
            batch_size=batch_size,
            num_batches=num_batches,
            lr=float(optimizer.get("lr", training.get("lr", 0.0001))),
            seed=int(training.get("seed", data.get("seed", 42))),
            device=str(runtime.get("device", training.get("device", "auto"))),
        )


__all__ = ["YoloWorldSpec"]
