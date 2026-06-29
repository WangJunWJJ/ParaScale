# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Token and patch-token estimators for dynamic batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass
class BatchEstimate:
    text_tokens: int = 0
    image_tokens: int = 0
    video_tokens: int = 0
    audio_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.text_tokens + self.image_tokens + self.video_tokens + self.audio_tokens
        )


def shape_numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def estimate_sample_tokens(
    sample: Any,
    image_patch_size: int = 14,
    video_patch_size: int = 14,
    audio_frame_tokens: int = 1,
) -> BatchEstimate:
    """Estimate token pressure for common multimodal sample dictionaries."""
    if not isinstance(sample, Mapping):
        return BatchEstimate(text_tokens=1)

    estimate = BatchEstimate()
    input_ids = sample.get("input_ids")
    if hasattr(input_ids, "numel"):
        estimate.text_tokens = int(input_ids.numel())
    elif isinstance(input_ids, Sequence):
        estimate.text_tokens = len(input_ids)

    pixel_values = _first_present(sample, "pixel_values", "images")
    if hasattr(pixel_values, "shape") and len(pixel_values.shape) >= 3:
        shape = list(pixel_values.shape)
        height, width = shape[-2], shape[-1]
        num_images = shape_numel(shape[:-3]) if len(shape) > 3 else 1
        estimate.image_tokens = int(
            num_images
            * max(1, height // image_patch_size)
            * max(1, width // image_patch_size)
        )

    video = _first_present(sample, "video", "video_values")
    if hasattr(video, "shape") and len(video.shape) >= 4:
        shape = list(video.shape)
        frames, height, width = shape[-4], shape[-2], shape[-1]
        estimate.video_tokens = int(
            frames
            * max(1, height // video_patch_size)
            * max(1, width // video_patch_size)
        )

    audio = _first_present(sample, "audio", "input_features")
    if hasattr(audio, "shape"):
        estimate.audio_tokens = int(audio.shape[-1]) * audio_frame_tokens

    return estimate


def _first_present(sample: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in sample and sample[key] is not None:
            return sample[key]
    return None
