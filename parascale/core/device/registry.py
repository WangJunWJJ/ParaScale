# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Device backend registry."""

from __future__ import annotations

from .ascend import AscendDeviceBackend
from .cpu import CpuDeviceBackend
from .cuda import CudaDeviceBackend, NvidiaDeviceBackend


def create_device_backend(kind: str = "auto"):
    normalized = (kind or "auto").lower()
    if normalized in {"cpu", "mock"}:
        return CpuDeviceBackend()
    if normalized in {"cuda", "gpu", "nvidia"}:
        return CudaDeviceBackend()
    if normalized in {"ascend", "npu"}:
        return AscendDeviceBackend()
    if normalized == "auto":
        cuda = CudaDeviceBackend()
        if cuda.is_available():
            return cuda
        ascend = AscendDeviceBackend()
        if ascend.is_available():
            return ascend
        return CpuDeviceBackend()
    raise ValueError(f"unknown device backend: {kind}")


__all__ = ["create_device_backend", "CudaDeviceBackend", "NvidiaDeviceBackend"]
