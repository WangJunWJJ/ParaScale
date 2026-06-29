# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午1:08
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Vision-aware batch samplers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .transforms import estimate_patch_tokens, sample_resolution


class PatchTokenBatchSampler:
    def __init__(
        self,
        dataset: Sequence[Any],
        max_patch_tokens: int,
        patch_size: int = 16,
        max_samples: int | None = None,
    ):
        if max_patch_tokens < 1:
            raise ValueError("max_patch_tokens must be >= 1")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be >= 1 when set")
        self.dataset = dataset
        self.max_patch_tokens = max_patch_tokens
        self.patch_size = patch_size
        self.max_samples = max_samples

    def __iter__(self):
        batch: List[int] = []
        used = 0
        for idx, sample in enumerate(self.dataset):
            height, width = sample_resolution(sample)
            tokens = estimate_patch_tokens(height, width, self.patch_size)
            sample_limit_reached = (
                self.max_samples is not None and len(batch) >= self.max_samples
            )
            if batch and (
                used + tokens > self.max_patch_tokens or sample_limit_reached
            ):
                yield batch
                batch, used = [], 0
            batch.append(idx)
            used += tokens
        if batch:
            yield batch


class ResolutionBucketSampler:
    def __init__(
        self,
        dataset: Sequence[Any],
        buckets: Iterable[Tuple[int, int]],
        batch_size: int,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.buckets = list(buckets)
        self.batch_size = batch_size

    def __iter__(self):
        grouped: Dict[Tuple[int, int], List[int]] = {
            bucket: [] for bucket in self.buckets
        }
        for idx, sample in enumerate(self.dataset):
            resolution = nearest_bucket(sample_resolution(sample), self.buckets)
            grouped.setdefault(resolution, []).append(idx)
        for indices in grouped.values():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if batch:
                    yield batch


def nearest_bucket(
    resolution: Tuple[int, int], buckets: Sequence[Tuple[int, int]]
) -> Tuple[int, int]:
    if not buckets:
        return resolution
    height, width = resolution
    return min(buckets, key=lambda bucket: abs(bucket[0] * bucket[1] - height * width))
