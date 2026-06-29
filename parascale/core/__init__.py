# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Core runtime abstractions shared by ParaScale training and inference."""

from .cluster import ClusterTopology, DeviceSpec, NodeSpec
from .device import (
    AscendDeviceBackend,
    CpuDeviceBackend,
    DeviceBackend,
    NvidiaDeviceBackend,
)
from .distributed import (
    CollectiveBackend,
    MockCollectiveBackend,
    TorchDistributedCollectiveBackend,
)

__all__ = [
    "DeviceBackend",
    "CpuDeviceBackend",
    "NvidiaDeviceBackend",
    "AscendDeviceBackend",
    "CollectiveBackend",
    "MockCollectiveBackend",
    "TorchDistributedCollectiveBackend",
    "DeviceSpec",
    "NodeSpec",
    "ClusterTopology",
]
