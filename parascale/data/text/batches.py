# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Disjoint DP token blocks with direct batch-position restoration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator

from .artifact import TokenCorpus


class TokenBatchCursor:
    """Pack full fixed-length blocks; drop equal-rank tails, never pad with duplicates."""

    def __init__(
        self,
        corpus: TokenCorpus,
        *,
        batch_size: int,
        sequence_length: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle: bool = True,
        repeat: bool = True,
        pin_memory: bool = False,
        shuffle_window_blocks: int = 65536,
    ) -> None:
        for name, value, minimum in (
            ("batch_size", batch_size, 1),
            ("sequence_length", sequence_length, 2),
            ("world_size", world_size, 1),
            ("rank", rank, 0),
            ("seed", seed, 0),
            ("shuffle_window_blocks", shuffle_window_blocks, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid {name}")
        if rank >= world_size:
            raise ValueError("rank must be less than world_size")
        for name, value in (
            ("shuffle", shuffle),
            ("repeat", repeat),
            ("pin_memory", pin_memory),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        self.global_batch = batch_size * world_size
        self.window_blocks = (
            shuffle_window_blocks // self.global_batch * self.global_batch
        )
        if not self.window_blocks:
            raise ValueError("shuffle_window_blocks must cover a global batch")
        self.corpus = corpus
        self.batch_size, self.sequence_length = batch_size, sequence_length
        self.block_count = len(corpus.tokens) // sequence_length
        self.batches_per_epoch = self.block_count // world_size // batch_size
        if not self.batches_per_epoch:
            raise ValueError("Token corpus cannot provide a full batch to every rank")
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.shuffle, self.repeat, self.pin_memory = shuffle, repeat, pin_memory
        self.position = 0
        self.identity = {
            "corpus": corpus.identity,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "rank": rank,
            "world_size": world_size,
            "seed": seed,
            "shuffle": shuffle,
            "repeat": repeat,
            "shuffle_window_blocks": shuffle_window_blocks,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"identity": deepcopy(self.identity), "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.validate_state_dict(state)
        self.position = state["position"]

    def validate_state_dict(self, state: dict[str, Any]) -> None:
        """Validate identity/position without changing data or RNG state."""
        if state.get("identity") != self.identity:
            raise ValueError("Token data resume identity mismatch")
        position = state.get("position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ValueError("Invalid token batch position")
        if not self.repeat and position > self.batches_per_epoch:
            raise ValueError("Token batch position exceeds the finite corpus")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import numpy as np
        import torch
        from torch.utils.data.distributed import DistributedSampler

        usable = self.batches_per_epoch * self.global_batch
        windows = (usable + self.window_blocks - 1) // self.window_blocks
        cached_window, indices = None, []
        while self.repeat or self.position < self.batches_per_epoch:
            epoch, batch = divmod(self.position, self.batches_per_epoch)
            window, offset = divmod(batch * self.global_batch, self.window_blocks)
            start = window * self.window_blocks
            if (epoch, window) != cached_window:
                length = min(self.window_blocks, usable - start)
                if self.shuffle:
                    sampler = DistributedSampler(
                        range(length),
                        num_replicas=self.world_size,
                        rank=self.rank,
                        shuffle=True,
                        seed=self.seed,
                        drop_last=True,
                    )
                    sampler.set_epoch(epoch * windows + window)
                    indices = list(sampler)
                else:
                    indices = range(self.rank, length, self.world_size)
                cached_window = epoch, window
            local = offset // self.global_batch * self.batch_size
            blocks = [
                start + index for index in indices[local : local + self.batch_size]
            ]
            tokens = np.stack(
                [
                    self.corpus.tokens[
                        index * self.sequence_length : (index + 1)
                        * self.sequence_length
                    ]
                    for index in blocks
                ]
            ).astype(np.int64)
            input_ids = torch.from_numpy(tokens)
            if self.pin_memory:
                input_ids = input_ids.pin_memory()
            self.position += 1
            yield {
                "input_ids": input_ids,
                "labels": input_ids,
                "batch_size": self.batch_size,
                "num_tokens": self.batch_size * (self.sequence_length - 1),
                "block_ids": blocks,
            }
