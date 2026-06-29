# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午1:08
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Vision transform helpers that stay torch-free by default."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def estimate_patch_tokens(height: int, width: int, patch_size: int = 16) -> int:
    return max(1, int(height) // patch_size) * max(1, int(width) // patch_size)


def sample_resolution(sample: Any) -> Tuple[int, int]:
    if isinstance(sample, dict):
        if "height" in sample and "width" in sample:
            return int(sample["height"]), int(sample["width"])
        if "resolution" in sample:
            height, width = sample["resolution"]
            return int(height), int(width)
    return 224, 224


def normalize_vision_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(sample)
    height, width = sample_resolution(sample)
    normalized.setdefault("height", height)
    normalized.setdefault("width", width)
    if "image" in normalized and "pixel_values" not in normalized:
        normalized["pixel_values"] = normalized["image"]
    return normalized
