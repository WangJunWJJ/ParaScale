# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Nvidia CUDA device backend."""

from __future__ import annotations

import importlib.util

from .base import DeviceBackend


class CudaDeviceBackend(DeviceBackend):
    def __init__(self) -> None:
        torch_available = importlib.util.find_spec("torch") is not None
        cuda_available = False
        if torch_available:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except Exception:
                cuda_available = False
        super().__init__(
            name="nvidia",
            accelerator="cuda",
            communication="nccl",
            available=cuda_available,
        )

    def set_device(self, local_rank: int = 0) -> None:
        if not self.available:
            return None
        import torch

        torch.cuda.set_device(local_rank)
        return None

    def synchronize(self) -> None:
        if self.available:
            import torch

            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        if self.available:
            import torch

            torch.cuda.empty_cache()

    def memory_allocated(self) -> int:
        if not self.available:
            return 0
        import torch

        return int(torch.cuda.memory_allocated())

    def max_memory_allocated(self) -> int:
        if not self.available:
            return 0
        import torch

        return int(torch.cuda.max_memory_allocated())

    def reset_peak_memory_stats(self) -> None:
        if self.available:
            import torch

            torch.cuda.reset_peak_memory_stats()

    def supports_bf16(self) -> bool:
        if not self.available:
            return False
        try:
            import torch

            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False

    def supports_flash_attention(self) -> bool:
        if not self.available:
            return False
        try:
            import torch

            major, _minor = torch.cuda.get_device_capability()
            return int(major) >= 8
        except Exception:
            return False


NvidiaDeviceBackend = CudaDeviceBackend

__all__ = ["CudaDeviceBackend", "NvidiaDeviceBackend"]
