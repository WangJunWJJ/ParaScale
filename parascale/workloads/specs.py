# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Typed workload specs for built-in runtime factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


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
