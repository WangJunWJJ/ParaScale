# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:58
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Training backend adapter namespace.

The concrete classes still live behind the legacy facade during the slimming
transition. New code should import from this package so backend modules can be
physically split without changing user-facing APIs.
"""

from .ascend_native import AscendNativeTrainingBackend
from .base import TrainingBackend
from .deepspeed import DeepSpeedTrainingBackend
from .fsdp import FSDPTrainingBackend
from .native import NativeDdpTrainingBackend, NativeTrainingBackend
from .registry import (
    TrainingBackendRegistry,
    create_runtime_training_backend,
    default_training_backend_registry,
)

__all__ = [
    "AscendNativeTrainingBackend",
    "DeepSpeedTrainingBackend",
    "FSDPTrainingBackend",
    "NativeDdpTrainingBackend",
    "NativeTrainingBackend",
    "TrainingBackend",
    "TrainingBackendRegistry",
    "create_runtime_training_backend",
    "default_training_backend_registry",
]
