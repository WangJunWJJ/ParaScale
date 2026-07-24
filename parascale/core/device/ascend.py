# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Huawei Ascend NPU device backend."""

from __future__ import annotations

import importlib.util

from .base import DeviceBackend


class AscendDeviceBackend(DeviceBackend):
    def __init__(self) -> None:
        torch_npu_available = importlib.util.find_spec("torch_npu") is not None
        available, device_count = self._inspect_npu_runtime(torch_npu_available)
        self.device_count = device_count
        super().__init__(
            name="ascend",
            accelerator="npu",
            communication="hccl",
            available=available,
        )

    def set_device(self, local_rank: int = 0) -> None:
        if not self.available:
            return None
        import torch

        torch.npu.set_device(local_rank)
        return None

    def synchronize(self) -> None:
        if self.available:
            import torch

            torch.npu.synchronize()

    def empty_cache(self) -> None:
        if self.available:
            import torch

            torch.npu.empty_cache()

    def memory_allocated(self) -> int:
        if not self.available:
            return 0
        import torch

        return int(torch.npu.memory_allocated())

    def max_memory_allocated(self) -> int:
        if not self.available:
            return 0
        import torch

        return int(torch.npu.max_memory_allocated())

    def reset_peak_memory_stats(self) -> None:
        if self.available:
            import torch

            torch.npu.reset_peak_memory_stats()

    def supports_bf16(self) -> bool:
        return self.available

    def capability(self):
        payload = super().capability()
        payload["device_count"] = int(self.device_count)
        return payload

    @staticmethod
    def _inspect_npu_runtime(torch_npu_available: bool) -> tuple[bool, int]:
        if not torch_npu_available:
            return False, 0
        try:
            import torch_npu  # noqa: F401
            import torch

            npu = getattr(torch, "npu", None)
            if npu is None:
                return False, 0
            is_available = (
                bool(npu.is_available()) if hasattr(npu, "is_available") else False
            )
            device_count = (
                int(npu.device_count()) if hasattr(npu, "device_count") else 0
            )
            return bool(is_available and device_count > 0), device_count
        except Exception:
            return False, 0


__all__ = ["AscendDeviceBackend"]
