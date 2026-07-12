# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Compatibility exports for workload specs.

The canonical workload spec definitions live in :mod:`parascale.workloads.specs`.
This module keeps the previous public import path stable for downstream users.
"""

from parascale.workloads.specs import (
    ClipContrastiveSpec,
    GroundDinoSpec,
    TinyTorchWorkloadSpec,
    VisionSyntheticSpec,
    VlmLoraSpec,
    YoloWorldSpec,
)

__all__ = [
    "ClipContrastiveSpec",
    "GroundDinoSpec",
    "TinyTorchWorkloadSpec",
    "VisionSyntheticSpec",
    "VlmLoraSpec",
    "YoloWorldSpec",
]
