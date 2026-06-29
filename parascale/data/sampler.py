# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Length, token-budget and distributed batch samplers."""

from __future__ import annotations

import random
from typing import Any, Callable, Iterator, List, Optional, Sequence

from .estimators import estimate_sample_tokens


class LengthBucketSampler:
    """Simple length-aware sampler to reduce padding waste and stragglers."""

    def __init__(
        self,
        dataset: Sequence[Any],
        batch_size: int,
        length_fn: Optional[Callable[[Any], int]] = None,
        shuffle: bool = True,
        bucket_size: int = 1024,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if bucket_size < batch_size:
            raise ValueError("bucket_size must be >= batch_size")
        self.dataset = dataset
        self.batch_size = batch_size
        self.length_fn = length_fn or (
            lambda sample: estimate_sample_tokens(sample).total_tokens
        )
        self.shuffle = shuffle
        self.bucket_size = bucket_size

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batches: List[List[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.length_fn(self.dataset[idx]))
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if batch:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class TokenBudgetBatchSampler:
    """Batch samples by an approximate max token budget instead of sample count."""

    def __init__(
        self,
        dataset: Sequence[Any],
        max_tokens: int,
        length_fn: Optional[Callable[[Any], int]] = None,
        shuffle: bool = True,
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.length_fn = length_fn or (
            lambda sample: estimate_sample_tokens(sample).total_tokens
        )
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        indices.sort(key=lambda idx: self.length_fn(self.dataset[idx]))

        batch: List[int] = []
        batch_tokens = 0
        for idx in indices:
            sample_tokens = max(1, self.length_fn(self.dataset[idx]))
            if batch and batch_tokens + sample_tokens > self.max_tokens:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(idx)
            batch_tokens += sample_tokens
        if batch:
            yield batch

    def __len__(self) -> int:
        total = sum(max(1, self.length_fn(sample)) for sample in self.dataset)
        return max(1, (total + self.max_tokens - 1) // self.max_tokens)


class DistributedTokenBudgetBatchSampler:
    """Token-budget sampler with deterministic rank sharding."""

    def __init__(
        self,
        dataset: Sequence[Any],
        max_tokens: int,
        rank: int = 0,
        world_size: int = 1,
        length_fn: Optional[Callable[[Any], int]] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.rank = rank
        self.world_size = world_size
        self.length_fn = length_fn or (
            lambda sample: estimate_sample_tokens(sample).total_tokens
        )
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        sampler = TokenBudgetBatchSampler(
            self.dataset,
            self.max_tokens,
            length_fn=self.length_fn,
            shuffle=self.shuffle,
        )
        batches = list(sampler)
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(batches)

        if not self.drop_last:
            remainder = len(batches) % self.world_size
            if remainder:
                batches.extend(batches[: self.world_size - remainder])

        for batch_idx, batch in enumerate(batches):
            if batch_idx % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        base_len = len(
            TokenBudgetBatchSampler(
                self.dataset, self.max_tokens, self.length_fn, self.shuffle
            )
        )
        if self.drop_last:
            return base_len // self.world_size
        return (base_len + self.world_size - 1) // self.world_size
