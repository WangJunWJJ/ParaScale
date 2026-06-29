# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:57
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Multimodal token-cost profile helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from parascale.data.estimators import BatchEstimate, estimate_sample_tokens


@dataclass(frozen=True)
class TokenCostEstimate:
    text_tokens: int = 0
    image_tokens: int = 0
    video_tokens: int = 0
    audio_tokens: int = 0
    samples: int = 1

    @property
    def total_tokens(self) -> int:
        return (
            self.text_tokens + self.image_tokens + self.video_tokens + self.audio_tokens
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_tokens": self.text_tokens,
            "image_tokens": self.image_tokens,
            "video_tokens": self.video_tokens,
            "audio_tokens": self.audio_tokens,
            "total_tokens": self.total_tokens,
            "samples": self.samples,
        }


def estimate_multimodal_token_cost(
    sample: Mapping[str, Any], image_patch_size: int = 14
) -> TokenCostEstimate:
    estimate: BatchEstimate = estimate_sample_tokens(
        sample, image_patch_size=image_patch_size
    )
    return TokenCostEstimate(
        text_tokens=estimate.text_tokens,
        image_tokens=estimate.image_tokens,
        video_tokens=estimate.video_tokens,
        audio_tokens=estimate.audio_tokens,
        samples=1,
    )


__all__ = ["TokenCostEstimate", "estimate_multimodal_token_cost"]
