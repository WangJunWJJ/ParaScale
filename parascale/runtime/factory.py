# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:48
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Runtime entrypoints for built-in workload factories."""

from __future__ import annotations

from parascale.workloads.registry import (
    build_optimizer_for_model,
    build_training_components,
    default_workload_registry,
)
from parascale.workloads.serving import (
    ServingModelRegistry,
    TinyTorchServingAdapter,
    build_serving_model_from_checkpoint,
    default_serving_model_registry,
    load_tiny_torch_serving_model,
)

__all__ = [
    "ServingModelRegistry",
    "TinyTorchServingAdapter",
    "build_optimizer_for_model",
    "build_serving_model_from_checkpoint",
    "build_training_components",
    "default_serving_model_registry",
    "default_workload_registry",
    "load_tiny_torch_serving_model",
]
