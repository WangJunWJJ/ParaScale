# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""VlmLoraSpec configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .common import _section


@dataclass
class VlmLoraSpec:
    model_type: str = "tiny_vlm_lora"
    model_family: str = "tiny_vlm"
    data_type: str = "synthetic_image_text"
    data_dir: str | None = None
    metadata_path: str | None = None
    streaming: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False
    dataset_local_cache_dir: str | None = None
    preprocess_in_workers: bool = False
    pipeline_cache: bool = False
    pipeline_cache_dir: str | None = None
    pipeline_cache_max_entries: int = 4096
    pipeline_cache_max_bytes: int = 20_000_000_000
    pipeline_cache_ttl_seconds: float = 0.0
    prompt_template_cache: bool = False
    prompt_template_cache_dir: str | None = None
    cuda_prefetch: bool = False
    image_size: int = 64
    channels: int = 3
    patch_size: int = 16
    vocab_size: int = 1024
    text_length: int = 32
    embed_dim: int = 128
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    use_peft: bool = True
    conversation_template: str = "qwen2_vl"
    prompt_field: str = "text"
    response_template: str = "Describe the image briefly."
    train_lm_head: bool = True
    pretrained_model_name_or_path: str | None = None
    activation_checkpointing: bool = False
    num_samples: int = 32
    batch_size: int = 4
    num_batches: int = 2
    lr: float = 0.001
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_config(cls, config_data: Dict[str, Any]) -> "VlmLoraSpec":
        parascale = _section(config_data, "parascale")
        task = _section(config_data, "task")
        training = _section(config_data, "training")
        model = _section(config_data, "model")
        data = _section(config_data, "data")
        optimizer = _section(config_data, "optimizer")
        runtime = _section(config_data, "runtime")
        lora = _section(config_data, "lora")
        target_modules = lora.get("target_modules", task.get("lora_target_modules"))
        if target_modules is None:
            target_modules = ("q_proj", "v_proj")
        elif isinstance(target_modules, str):
            target_modules = tuple(
                item.strip() for item in target_modules.split(",") if item.strip()
            )
        else:
            target_modules = tuple(str(item) for item in target_modules)
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
            model_type=str(model.get("type", "tiny_vlm_lora")),
            model_family=str(model.get("family", task.get("model_family", "tiny_vlm"))),
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
            preprocess_in_workers=bool(
                data.get(
                    "preprocess_in_workers",
                    parascale.get("preprocess_in_workers", False),
                )
            ),
            pipeline_cache=bool(
                data.get("pipeline_cache", parascale.get("pipeline_cache", False))
            ),
            pipeline_cache_dir=(
                str(data.get("pipeline_cache_dir", parascale.get("pipeline_cache_dir")))
                if data.get("pipeline_cache_dir", parascale.get("pipeline_cache_dir"))
                is not None
                else None
            ),
            pipeline_cache_max_entries=int(
                data.get(
                    "pipeline_cache_max_entries",
                    parascale.get("pipeline_cache_max_entries", 4096),
                )
                or 4096
            ),
            pipeline_cache_max_bytes=int(
                data.get(
                    "pipeline_cache_max_bytes",
                    parascale.get("pipeline_cache_max_bytes", 20_000_000_000),
                )
                or 20_000_000_000
            ),
            pipeline_cache_ttl_seconds=float(
                data.get(
                    "pipeline_cache_ttl_seconds",
                    parascale.get("pipeline_cache_ttl_seconds", 0.0),
                )
                or 0.0
            ),
            prompt_template_cache=bool(
                data.get(
                    "prompt_template_cache",
                    parascale.get("prompt_template_cache", False),
                )
            ),
            prompt_template_cache_dir=(
                str(
                    data.get(
                        "prompt_template_cache_dir",
                        parascale.get("prompt_template_cache_dir"),
                    )
                )
                if data.get(
                    "prompt_template_cache_dir",
                    parascale.get("prompt_template_cache_dir"),
                )
                is not None
                else None
            ),
            cuda_prefetch=bool(parascale.get("cuda_prefetch", False)),
            image_size=int(data.get("image_size", model.get("image_size", 64))),
            channels=int(data.get("channels", model.get("channels", 3))),
            patch_size=int(model.get("patch_size", data.get("patch_size", 16))),
            vocab_size=int(model.get("vocab_size", data.get("vocab_size", 1024))),
            text_length=int(data.get("text_length", model.get("text_length", 32))),
            embed_dim=int(model.get("embed_dim", 128)),
            lora_rank=int(lora.get("rank", task.get("lora_rank", 8))),
            lora_alpha=float(lora.get("alpha", task.get("lora_alpha", 16.0))),
            lora_dropout=float(lora.get("dropout", task.get("lora_dropout", 0.0))),
            lora_target_modules=target_modules,
            use_peft=bool(lora.get("use_peft", task.get("use_peft", True))),
            conversation_template=str(
                task.get(
                    "conversation_template",
                    model.get("conversation_template", "qwen2_vl"),
                )
            ),
            prompt_field=str(task.get("prompt_field", "text")),
            response_template=str(
                task.get("response_template", "Describe the image briefly.")
            ),
            train_lm_head=bool(
                lora.get("train_lm_head", task.get("train_lm_head", True))
            ),
            pretrained_model_name_or_path=(
                str(model.get("pretrained_model_name_or_path"))
                if model.get("pretrained_model_name_or_path") is not None
                else None
            ),
            activation_checkpointing=bool(
                model.get(
                    "activation_checkpointing",
                    parascale.get("enable_activation_checkpointing", False),
                )
            )
            and str(parascale.get("training_backend", "")).lower() != "fsdp",
            num_samples=int(
                data.get("num_samples", max(batch_size * num_batches, batch_size))
            ),
            batch_size=batch_size,
            num_batches=num_batches,
            lr=float(optimizer.get("lr", training.get("lr", 0.001))),
            seed=int(training.get("seed", data.get("seed", 42))),
            device=str(runtime.get("device", training.get("device", "auto"))),
        )


__all__ = ["VlmLoraSpec"]
