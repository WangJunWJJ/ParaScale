# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:26
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inference task adapters."""

from .base import InferenceTaskAdapter
from .multimodal import MultimodalInferenceTaskAdapter
from .registry import InferenceTaskRegistry, default_inference_task_registry
from .text import TextInferenceTaskAdapter
from .vision import VisionInferenceTaskAdapter

__all__ = [
    "InferenceTaskAdapter",
    "InferenceTaskRegistry",
    "MultimodalInferenceTaskAdapter",
    "TextInferenceTaskAdapter",
    "VisionInferenceTaskAdapter",
    "default_inference_task_registry",
]
