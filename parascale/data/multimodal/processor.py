# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:57
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Multimodal sample normalization and processor pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from parascale.data.schema import MultiModalBatchSchema

from .profiler import estimate_multimodal_token_cost


def normalize_multimodal_sample(
    sample: Mapping[str, Any],
    schema: Optional[MultiModalBatchSchema] = None,
) -> Dict[str, Any]:
    """Normalize common aliases into the canonical multimodal schema."""
    schema = schema or MultiModalBatchSchema()
    normalized = dict(sample)

    aliases = {
        schema.pixel_values: ["images", "image", "pixel_values"],
        schema.video_values: ["video", "videos", "video_values"],
        schema.audio_features: ["audio", "input_features", "audio_features"],
    }
    for canonical_key, alias_keys in aliases.items():
        if canonical_key in normalized:
            continue
        for alias in alias_keys:
            if alias in normalized:
                normalized[canonical_key] = normalized[alias]
                break

    if schema.modality_mask not in normalized:
        normalized[schema.modality_mask] = {
            "text": schema.input_ids in normalized,
            "image": schema.pixel_values in normalized,
            "video": schema.video_values in normalized,
            "audio": schema.audio_features in normalized,
        }
    return normalized


@dataclass
class MultiModalDataPipeline:
    tokenizer: Optional[Callable[[Any], Any]] = None
    image_processor: Optional[Callable[[Any], Any]] = None
    video_processor: Optional[Callable[[Any], Any]] = None
    audio_processor: Optional[Callable[[Any], Any]] = None
    schema: MultiModalBatchSchema = field(default_factory=MultiModalBatchSchema)
    batching: str = "token_budget"
    cache: Dict[str, Any] = field(default_factory=dict)

    def process(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        processed = dict(sample)
        if (
            self.tokenizer is not None
            and "text" in processed
            and self.schema.input_ids not in processed
        ):
            processed[self.schema.input_ids] = self.tokenizer(processed["text"])
        if self.image_processor is not None and "images" in processed:
            processed[self.schema.pixel_values] = self.image_processor(
                processed["images"]
            )
        if self.video_processor is not None and "video" in processed:
            processed[self.schema.video_values] = self.video_processor(
                processed["video"]
            )
        if self.audio_processor is not None and "audio" in processed:
            processed[self.schema.audio_features] = self.audio_processor(
                processed["audio"]
            )
        return normalize_multimodal_sample(processed, self.schema)

    def process_cached(
        self, sample_id: str, sample: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if sample_id in self.cache:
            return dict(self.cache[sample_id])
        processed = self.process(sample)
        self.cache[sample_id] = dict(processed)
        return processed

    def profile_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        processed = normalize_multimodal_sample(sample, self.schema)
        input_ids = processed.get(self.schema.input_ids, [])
        token_count = len(input_ids) if hasattr(input_ids, "__len__") else 0
        modalities = processed.get(self.schema.modality_mask, {})
        token_cost = estimate_multimodal_token_cost(processed)
        return {
            "tokens": token_count,
            "total_tokens": token_cost.total_tokens,
            "image_tokens": token_cost.image_tokens,
            "token_cost": token_cost.to_dict(),
            "has_text": bool(modalities.get("text")),
            "has_image": bool(modalities.get("image")),
            "has_video": bool(modalities.get("video")),
            "has_audio": bool(modalities.get("audio")),
        }


__all__ = ["MultiModalDataPipeline", "normalize_multimodal_sample"]
