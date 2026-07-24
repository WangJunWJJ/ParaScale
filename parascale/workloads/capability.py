# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午9:36
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Workload capability classification helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class WorkloadCapability:
    workload: str
    data_type: str
    capability_level: str
    flags: Dict[str, bool]

    def payload_flags(self) -> Dict[str, bool]:
        return dict(self.flags)


def describe_workload(config_data: Dict[str, Any]) -> WorkloadCapability:
    training = _section(config_data, "training")
    data = _section(config_data, "data")
    workload = str(training.get("workload", "synthetic_regression"))
    data_type = str(data.get("type", ""))
    flags = workload_flags(workload)
    base_level = _base_capability(flags, data_type)
    return WorkloadCapability(
        workload=workload,
        data_type=data_type,
        capability_level=capability_level_for_scope(base_level, config_data),
        flags=flags,
    )


def capability_level_for_training(config_data: Dict[str, Any]) -> str:
    return describe_workload(config_data).capability_level


def capability_level_for_scope(base_level: str, config_data: Dict[str, Any]) -> str:
    hardware = _section(config_data, "hardware_profile")
    launch = _section(config_data, "launch")
    env_world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    world_size = int(
        hardware.get("world_size", hardware.get("num_gpus", env_world_size)) or 1
    )
    gpus_per_node = int(
        hardware.get("gpus_per_node", launch.get("nproc_per_node", world_size))
        or world_size
    )
    nnodes = int(hardware.get("num_nodes", launch.get("nnodes", 1)) or 1)
    if world_size > 1 and (nnodes > 1 or gpus_per_node == 1):
        return "multi_node_smoke"
    return base_level


def workload_flags(workload: str) -> Dict[str, bool]:
    normalized = str(workload or "").strip().lower()
    return {
        "synthetic": normalized
        in {"synthetic_regression", "torch_tiny", "torch_tiny_mlp", "tiny_torch"},
        "vision_synthetic": normalized
        in {"vision_synthetic", "synthetic_vision", "tiny_vit"},
        "clip_contrastive": normalized
        in {"clip_contrastive", "clip_style_contrastive", "tiny_clip"},
        "vlm_lora": normalized
        in {"vlm_lora", "vlm_lora_finetune", "tiny_vlm_lora"},
        "yolo_world": normalized
        in {"yolo_world", "yolo_world_detection", "yoloworld"},
        "ground_dino": normalized
        in {"ground_dino", "grounding_dino", "groundingdino", "ground_dino_detection"},
    }


def _base_capability(flags: Dict[str, bool], data_type: str) -> str:
    if flags["synthetic"]:
        return "local_native_synthetic"
    if flags["vision_synthetic"]:
        return "local_native_vision_synthetic"
    if flags["clip_contrastive"]:
        if data_type in {"datacomp_wds", "webdataset", "wds"}:
            return "local_native_clip_contrastive_datacomp_wds"
        return "local_native_clip_contrastive_synthetic"
    if flags["vlm_lora"]:
        if data_type in {"datacomp_wds", "webdataset", "wds"}:
            return "local_native_vlm_lora_datacomp_wds"
        return "local_native_vlm_lora_synthetic"
    if flags["yolo_world"]:
        if data_type in {"objects365_cached", "objects365", "object365"}:
            return "local_native_yolo_world_objects365"
        if data_type in {"coco", "coco_cached"}:
            return "local_native_yolo_world_coco"
        return "local_native_yolo_world_detection"
    if flags["ground_dino"]:
        if data_type in {"phrase_grounding", "grounding_phrase", "ground_dino_json"}:
            return "local_native_ground_dino_phrase_grounding"
        return "local_native_ground_dino_detection"
    return "local_native_real_torch"


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


__all__ = [
    "WorkloadCapability",
    "capability_level_for_scope",
    "capability_level_for_training",
    "describe_workload",
    "workload_flags",
]
