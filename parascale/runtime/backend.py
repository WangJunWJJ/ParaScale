# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Legacy training backend import facade.

New code should import from :mod:`parascale.runtime.backends`.
"""

from __future__ import annotations

from parascale.runtime.backends import (
    DeepSpeedTrainingBackend,
    FSDPTrainingBackend,
    NativeDdpTrainingBackend,
    NativeTrainingBackend,
    TrainingBackend,
    TrainingBackendRegistry,
    create_runtime_training_backend,
    default_training_backend_registry,
)

__all__ = [
    "DeepSpeedTrainingBackend",
    "FSDPTrainingBackend",
    "NativeDdpTrainingBackend",
    "NativeTrainingBackend",
    "TrainingBackend",
    "TrainingBackendRegistry",
    "create_runtime_training_backend",
    "default_training_backend_registry",
]
