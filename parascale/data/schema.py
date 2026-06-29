# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Data schemas shared by text, vision and multimodal pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MultiModalBatchSchema:
    """Canonical DeepSpeed-friendly multimodal batch keys."""

    input_ids: str = "input_ids"
    attention_mask: str = "attention_mask"
    labels: str = "labels"
    pixel_values: str = "pixel_values"
    image_grid_thw: str = "image_grid_thw"
    video_values: str = "video_values"
    video_grid_thw: str = "video_grid_thw"
    audio_features: str = "audio_features"
    modality_mask: str = "modality_mask"
    metadata: str = "metadata"

    @property
    def model_input_keys(self) -> List[str]:
        return [
            self.input_ids,
            self.attention_mask,
            self.pixel_values,
            self.image_grid_thw,
            self.video_values,
            self.video_grid_thw,
            self.audio_features,
            self.modality_mask,
        ]

    @property
    def label_keys(self) -> List[str]:
        return [self.labels]

    def to_dict(self) -> Dict[str, str]:
        return dict(self.__dict__)
