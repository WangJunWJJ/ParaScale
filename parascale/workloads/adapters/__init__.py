# -*- coding: utf-8 -*-
# @Time : 2026/6/22 上午10:49
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Model-specific workload adapters."""

from parascale.contracts import WorkloadAdapter

from .registry import WorkloadAdapterRegistry
from .yolo import YoloDetectionTargetAdapter, YoloOfficialBatchAdapter

__all__ = [
    "WorkloadAdapter",
    "WorkloadAdapterRegistry",
    "YoloDetectionTargetAdapter",
    "YoloOfficialBatchAdapter",
]
