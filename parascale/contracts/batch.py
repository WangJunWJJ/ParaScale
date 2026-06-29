# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:54
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Batch-level runtime contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class BatchMetadata:
    num_samples: int = 0
    num_images: int = 0
    num_pairs: int = 0
    tokens: int = 0
    patch_tokens: int = 0
    padding_ratio: float = 0.0

    @classmethod
    def from_batch(cls, batch: Mapping[str, Any]) -> "BatchMetadata":
        return cls(
            num_samples=int(batch.get("batch_size", batch.get("num_samples", 0)) or 0),
            num_images=int(batch.get("num_images", batch.get("images", 0)) or 0),
            num_pairs=int(
                batch.get("num_pairs", batch.get("image_text_pairs", 0)) or 0
            ),
            tokens=int(batch.get("tokens", batch.get("token_count", 0)) or 0),
            patch_tokens=int(batch.get("patch_tokens", 0) or 0),
            padding_ratio=float(batch.get("padding_ratio", 0.0) or 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_images": self.num_images,
            "num_pairs": self.num_pairs,
            "tokens": self.tokens,
            "patch_tokens": self.patch_tokens,
            "padding_ratio": self.padding_ratio,
        }


@dataclass(frozen=True)
class BatchContract:
    """Document the fields a model-ready batch may expose to TrainEngine."""

    tensor_fields: tuple[str, ...] = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "labels",
        "img",
        "cls",
        "bboxes",
        "batch_idx",
    )
    metric_fields: tuple[str, ...] = (
        "num_images",
        "num_pairs",
        "tokens",
        "patch_tokens",
        "padding_ratio",
    )
    profile_field: str = "pipeline_profile"
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate_lightweight(self, batch: Mapping[str, Any]) -> list[str]:
        warnings: list[str] = []
        if self.profile_field in batch and not isinstance(
            batch[self.profile_field], dict
        ):
            warnings.append("pipeline_profile must be a dictionary when present.")
        if not any(field in batch for field in self.tensor_fields):
            warnings.append("batch does not expose a known tensor payload field.")
        return warnings
