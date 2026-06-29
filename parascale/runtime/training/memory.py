# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:51
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Runtime device synchronization and memory metric collection."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from parascale.runtime.backends.devices import current_accelerator


class RuntimeMemoryTracker:
    """Collect accelerator memory metrics without making torch a hard dependency."""

    def __init__(
        self,
        cuda_getter: Callable[[], Any] | None = None,
        accelerator_getter: Callable[[], Any] | None = None,
    ) -> None:
        self._cuda_getter = cuda_getter
        self._accelerator_getter = accelerator_getter

    def accelerator(self) -> Any:
        if self._accelerator_getter is not None:
            return self._accelerator_getter()
        if self._cuda_getter is not None:
            return self._cuda_getter()
        try:
            import torch

            accelerator = current_accelerator(torch)
            if accelerator == "cuda":
                return torch.cuda
            if accelerator == "npu":
                return torch.npu
        except Exception:
            return None
        return None

    def torch_cuda(self) -> Any:
        return self.accelerator()

    def reset_peak_memory_stats(self) -> None:
        accelerator = self.accelerator()
        if accelerator is None:
            return
        try:
            accelerator.reset_peak_memory_stats()
        except Exception:
            return

    def add_peak_memory_metrics(self, metrics: Dict[str, Any]) -> None:
        accelerator = self.accelerator()
        if accelerator is None:
            metrics.setdefault("peak_memory_bytes", 0)
            metrics.setdefault("allocated_memory_bytes", 0)
            return
        try:
            metrics["peak_memory_bytes"] = int(accelerator.max_memory_allocated())
            metrics["allocated_memory_bytes"] = int(accelerator.memory_allocated())
        except Exception:
            metrics.setdefault("peak_memory_bytes", 0)
            metrics.setdefault("allocated_memory_bytes", 0)

    def synchronize_device(self) -> None:
        accelerator = self.accelerator()
        if accelerator is None:
            return
        try:
            accelerator.synchronize()
        except Exception:
            return

    def elapsed_since(self, start: float, *, synchronized: bool = False) -> float:
        if not synchronized:
            self.synchronize_device()
        return time.perf_counter() - start


__all__ = ["RuntimeMemoryTracker"]
