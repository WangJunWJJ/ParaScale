# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午5:36
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CUDA stream prefetch controller."""

from __future__ import annotations

import time
from typing import Any, Iterator

from parascale.runtime.backends.devices import move_batch_to_device


class CudaStreamPrefetchIterator:
    def __init__(self, torch: Any, source: Iterator[Any], device: Any) -> None:
        self.torch = torch
        self.source = source
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch: Any = None
        self.next_ready = False
        self.next_h2d_ms = 0.0
        self._preload()

    def __iter__(self) -> "CudaStreamPrefetchIterator":
        return self

    def __next__(self) -> Any:
        if not self.next_ready:
            raise StopIteration
        wait_start = time.perf_counter()
        self.torch.cuda.current_stream(self.device).wait_stream(self.stream)
        wait_ms = (time.perf_counter() - wait_start) * 1000.0
        batch = self.next_batch
        h2d_ms = self.next_h2d_ms
        self._record_prefetch_profile(batch, h2d_ms=h2d_ms, wait_ms=wait_ms)
        self._preload()
        return batch

    def _preload(self) -> None:
        try:
            batch = next(self.source)
        except StopIteration:
            self.next_batch = None
            self.next_ready = False
            self.next_h2d_ms = 0.0
            return
        start = time.perf_counter()
        with self.torch.cuda.stream(self.stream):
            self.next_batch = self._move_to_device(batch)
        self.next_h2d_ms = (time.perf_counter() - start) * 1000.0
        self.next_ready = True

    def _move_to_device(self, value: Any) -> Any:
        return move_batch_to_device(value, str(self.device))

    @staticmethod
    def _record_prefetch_profile(batch: Any, *, h2d_ms: float, wait_ms: float) -> None:
        if not isinstance(batch, dict):
            return
        profile = batch.get("pipeline_profile")
        if not isinstance(profile, dict):
            profile = {}
            batch["pipeline_profile"] = profile
        profile["cuda_prefetch_h2d_ms"] = max(0.0, float(h2d_ms))
        profile["cuda_prefetch_wait_ms"] = max(0.0, float(wait_ms))


def maybe_cuda_prefetch_iterator(
    iterator: Iterator[Any],
    *,
    config: Any,
    local_rank: int,
) -> Iterator[Any]:
    return maybe_device_prefetch_iterator(
        iterator,
        config=config,
        local_rank=local_rank,
    )


def maybe_device_prefetch_iterator(
    iterator: Iterator[Any],
    *,
    config: Any,
    local_rank: int,
) -> Iterator[Any]:
    enabled = bool(
        getattr(config, "device_prefetch", False)
        or getattr(config, "cuda_prefetch", False)
    )
    if not enabled:
        return iterator
    try:
        import torch
    except Exception:
        return iterator
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return iterator
    configured = getattr(config, "prefetch_device", None) or getattr(
        config, "cuda_prefetch_device", None
    )
    if configured:
        device = torch.device(str(configured))
    elif local_rank >= 0:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")
    return CudaStreamPrefetchIterator(torch, iterator, device)


__all__ = [
    "CudaStreamPrefetchIterator",
    "maybe_cuda_prefetch_iterator",
    "maybe_device_prefetch_iterator",
]
