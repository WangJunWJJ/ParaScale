# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Hardware device backend namespace."""

from .ascend import AscendDeviceBackend
from .base import DeviceBackend
from .cpu import CpuDeviceBackend
from .cuda import CudaDeviceBackend, NvidiaDeviceBackend
from .registry import create_device_backend

__all__ = [
    "AscendDeviceBackend",
    "CpuDeviceBackend",
    "CudaDeviceBackend",
    "DeviceBackend",
    "NvidiaDeviceBackend",
    "create_device_backend",
]
