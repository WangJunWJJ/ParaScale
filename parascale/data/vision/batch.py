# -*- coding: utf-8 -*-
# @Time : 2026/6/22 上午10:49
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Batch assembly primitives for preprocessed vision samples."""

from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence

from .preprocessor import ProcessedVisionSample, VisionPreprocessor, VisionSample


class VisionBatchAdapter(Protocol):
    def collate(self, samples: Sequence[ProcessedVisionSample]) -> Dict[str, Any]: ...


class VisionBatchCollator:
    """Apply a generic preprocessor and a model-specific batch adapter."""

    def __init__(
        self,
        *,
        preprocessor: VisionPreprocessor,
        batch_adapter: VisionBatchAdapter,
    ) -> None:
        self.preprocessor = preprocessor
        self.batch_adapter = batch_adapter

    def __call__(self, samples: Sequence[VisionSample]) -> Dict[str, Any]:
        processed = [self.preprocessor.process(sample) for sample in samples]
        batch = self.batch_adapter.collate(processed)
        batch["pipeline_profile"] = self._merge_profiles(processed)
        return batch

    @staticmethod
    def _merge_profiles(samples: Sequence[ProcessedVisionSample]) -> Dict[str, float]:
        profile: Dict[str, float] = {}
        for sample in samples:
            for key, value in sample.profile.items():
                profile[key] = profile.get(key, 0.0) + float(value or 0.0)
        hit_count = profile.get("tensor_cache_hit_count", 0.0)
        sample_count = max(profile.get("tensor_cache_sample_count", 0.0), 1.0)
        profile["tensor_cache_hit_ratio"] = hit_count / sample_count
        cache_hit_count = profile.get("cache_hit_count", hit_count)
        cache_sample_count = max(profile.get("cache_sample_count", sample_count), 1.0)
        profile["cache_hit"] = cache_hit_count / cache_sample_count
        return profile
