# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""ClipContrastiveSpec configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .common import _section


@dataclass
class ClipContrastiveSpec:
    model_type: str = "tiny_clip"
    data_type: str = "synthetic_image_text"
    data_dir: str | None = None
    metadata_path: str | None = None
    streaming: bool = False
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    dataset_local_cache_dir: str | None = None
    wds_image_mode: str = "tensor"
    cuda_prefetch: bool = False
    image_size: int = 32
    channels: int = 3
    patch_size: int = 16
    vocab_size: int = 128
    text_length: int = 12
    embed_dim: int = 32
    vision_layers: int = 0
    text_layers: int = 0
    num_heads: int = 4
    mlp_ratio: float = 4.0
    activation_checkpointing: bool = False
    pretrained_model_name_or_path: str | None = None
    num_samples: int = 16
    batch_size: int = 4
    num_batches: int = 2
    temperature: float = 0.07
    lr: float = 0.01
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_config(cls, config_data: Dict[str, Any]) -> "ClipContrastiveSpec":
        parascale = _section(config_data, "parascale")
        task = _section(config_data, "task")
        training = _section(config_data, "training")
        model = _section(config_data, "model")
        data = _section(config_data, "data")
        optimizer = _section(config_data, "optimizer")
        runtime = _section(config_data, "runtime")
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
            model_type=str(model.get("type", "tiny_clip")),
            data_type=str(data.get("type", "synthetic_image_text")),
            data_dir=(
                str(data.get("data_dir", data.get("wds_dir")))
                if data.get("data_dir", data.get("wds_dir")) is not None
                else None
            ),
            metadata_path=(
                str(data.get("metadata_path"))
                if data.get("metadata_path") is not None
                else None
            ),
            streaming=bool(data.get("streaming", False)),
            num_workers=int(
                data.get("num_workers", parascale.get("dataloader_num_workers", 0)) or 0
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
            dataset_local_cache_dir=(
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
            ),
            wds_image_mode=str(data.get("wds_image_mode", "tensor")),
            cuda_prefetch=bool(parascale.get("cuda_prefetch", False)),
            image_size=int(data.get("image_size", model.get("image_size", 32))),
            channels=int(data.get("channels", model.get("channels", 3))),
            patch_size=int(model.get("patch_size", data.get("patch_size", 16))),
            vocab_size=int(model.get("vocab_size", data.get("vocab_size", 128))),
            text_length=int(data.get("text_length", model.get("text_length", 12))),
            embed_dim=int(model.get("embed_dim", 32)),
            vision_layers=int(model.get("vision_layers", model.get("num_layers", 0))),
            text_layers=int(model.get("text_layers", 0)),
            num_heads=int(model.get("num_heads", 4)),
            mlp_ratio=float(model.get("mlp_ratio", 4.0)),
            activation_checkpointing=bool(
                model.get(
                    "activation_checkpointing",
                    parascale.get("enable_activation_checkpointing", False),
                )
            )
            and str(parascale.get("training_backend", "")).lower() != "fsdp",
            pretrained_model_name_or_path=(
                str(model.get("pretrained_model_name_or_path"))
                if model.get("pretrained_model_name_or_path") is not None
                else None
            ),
            num_samples=int(
                data.get("num_samples", max(batch_size * num_batches, batch_size))
            ),
            batch_size=batch_size,
            num_batches=num_batches,
            temperature=float(
                task.get("temperature", training.get("temperature", 0.07))
            ),
            lr=float(optimizer.get("lr", training.get("lr", 0.01))),
            seed=int(training.get("seed", data.get("seed", 42))),
            device=str(runtime.get("device", training.get("device", "auto"))),
        )


__all__ = ["ClipContrastiveSpec"]
