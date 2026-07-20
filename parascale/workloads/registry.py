# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:48
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Built-in workload registry and training component entrypoints."""

from __future__ import annotations

from typing import Any, Dict

from parascale.runtime.workloads import WorkloadRegistry
from parascale.workloads.specs.clip import ClipContrastiveSpec
from parascale.workloads.specs.ground_dino import GroundDinoSpec
from parascale.workloads.specs.tiny import TinyTorchWorkloadSpec
from parascale.workloads.specs.vision import VisionSyntheticSpec
from parascale.workloads.specs.vlm_lora import VlmLoraSpec
from parascale.workloads.specs.yolo import YoloWorldSpec

from .clip import build_clip_contrastive_components
from .common import _require_torch, _section
from .ground_dino import build_ground_dino_components
from .optimizer import (
    trainable_parameter_stats as trainable_parameter_stats,
)
from .tiny import build_tiny_torch_components
from .vision import build_vision_synthetic_components
from .vlm_lora import build_vlm_lora_components
from .yolo import build_yolo_world_components


def build_training_components(config_data: Dict[str, Any]):
    """Build model, optimizer, dataloader and loss for a configured workload."""
    training = _section(config_data, "training")
    workload = str(training.get("workload", "synthetic_regression"))
    return default_workload_registry().create(workload, config_data)


def default_workload_registry() -> WorkloadRegistry:
    registry = WorkloadRegistry()
    registry.register(
        "vision_synthetic",
        _build_vision_synthetic_from_config,
        aliases=("synthetic_vision", "tiny_vit"),
    )
    registry.register(
        "clip_contrastive",
        _build_clip_contrastive_from_config,
        aliases=("clip_style_contrastive", "tiny_clip"),
    )
    registry.register(
        "vlm_lora",
        _build_vlm_lora_from_config,
        aliases=("vlm_lora_finetune", "tiny_vlm_lora"),
    )
    registry.register(
        "yolo_world",
        _build_yolo_world_from_config,
        aliases=("yolo_world_detection", "yoloworld"),
    )
    registry.register(
        "ground_dino",
        _build_ground_dino_from_config,
        aliases=("grounding_dino", "groundingdino", "ground_dino_detection"),
    )
    registry.register(
        "torch_tiny",
        _build_tiny_torch_from_config,
        aliases=("synthetic_regression", "torch_tiny_mlp", "tiny_torch"),
    )
    return registry


def _build_tiny_torch_from_config(config_data: Dict[str, Any]):
    return build_tiny_torch_components(TinyTorchWorkloadSpec.from_config(config_data))


def _build_vision_synthetic_from_config(config_data: Dict[str, Any]):
    return build_vision_synthetic_components(
        VisionSyntheticSpec.from_config(config_data)
    )


def _build_clip_contrastive_from_config(config_data: Dict[str, Any]):
    return build_clip_contrastive_components(
        ClipContrastiveSpec.from_config(config_data)
    )


def _build_vlm_lora_from_config(config_data: Dict[str, Any]):
    return build_vlm_lora_components(VlmLoraSpec.from_config(config_data))


def _build_yolo_world_from_config(config_data: Dict[str, Any]):
    return build_yolo_world_components(YoloWorldSpec.from_config(config_data))


def _build_ground_dino_from_config(config_data: Dict[str, Any]):
    return build_ground_dino_components(GroundDinoSpec.from_config(config_data))


def build_optimizer_for_model(model: Any, config_data: Dict[str, Any]):
    _require_torch()
    from parascale.optimizers import build_optimizer

    return build_optimizer(model, config_data)
