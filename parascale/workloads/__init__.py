# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Built-in ParaScale workload builders."""

from __future__ import annotations

from .causal_lm import (
    CausalLmSpec,
    build_causal_lm_components,
    build_causal_lm_stage_components,
)
from .clip import ClipContrastiveSpec, build_clip_contrastive_components
from .datacomp import _iter_datacomp_tar_entries, _looks_like_supported_image
from .ground_dino import GroundDinoSpec, build_ground_dino_components
from .registry import (
    build_optimizer_for_model,
    build_training_components,
    default_workload_registry,
)
from .serving import build_serving_model_from_checkpoint
from .tiny import TinyTorchWorkloadSpec, build_tiny_torch_components
from .vision import VisionSyntheticSpec, build_vision_synthetic_components
from .vlm_lora import VlmLoraSpec, build_vlm_lora_components
from .yolo import (
    YoloDetectionSpec,
    YoloWorldSpec,
    build_yolo_detection_components,
    build_yolo_world_components,
)

__all__ = [
    "CausalLmSpec",
    "ClipContrastiveSpec",
    "GroundDinoSpec",
    "TinyTorchWorkloadSpec",
    "VisionSyntheticSpec",
    "VlmLoraSpec",
    "YoloWorldSpec",
    "YoloDetectionSpec",
    "_iter_datacomp_tar_entries",
    "_looks_like_supported_image",
    "build_clip_contrastive_components",
    "build_causal_lm_components",
    "build_causal_lm_stage_components",
    "build_ground_dino_components",
    "build_optimizer_for_model",
    "build_serving_model_from_checkpoint",
    "build_tiny_torch_components",
    "build_training_components",
    "build_vision_synthetic_components",
    "build_vlm_lora_components",
    "build_yolo_world_components",
    "build_yolo_detection_components",
    "default_workload_registry",
]
