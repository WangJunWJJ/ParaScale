# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午1:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Vision throughput profiling primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .transforms import estimate_patch_tokens, sample_resolution


@dataclass
class VisionThroughputProfile:
    images_per_second: float = 0.0
    patch_tokens_per_second: float = 0.0
    decode_time: float = 0.0
    augment_time: float = 0.0
    device_transfer_time: float = 0.0
    peak_memory: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class VisionThroughputProfiler:
    def profile_batch(
        self, batch: Dict[str, Any], step_time_seconds: float = 0.0
    ) -> VisionThroughputProfile:
        images = len(batch.get("pixel_values", batch.get("images", [])))
        patch_tokens = sum(
            estimate_patch_tokens(*sample_resolution(sample))
            for sample in batch.get("metadata", [])
            if isinstance(sample, dict)
        )
        if step_time_seconds <= 0:
            return VisionThroughputProfile()
        return VisionThroughputProfile(
            images_per_second=images / step_time_seconds,
            patch_tokens_per_second=patch_tokens / step_time_seconds,
        )
