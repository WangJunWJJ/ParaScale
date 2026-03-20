# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午 14:30
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 量化训练模块

本模块提供量化感知训练（QAT）和训练后量化（PTQ）功能，
支持 INT8/INT4 量化，可与数据并行、模型并行、张量并行、流水线并行和 3D 混合并行同时使用。
"""

from .base import QuantizationConfig
from .observers import MinMaxObserver, MovingAverageObserver
from .fake_quantize import FakeQuantize, FakeQuantizedLinear, FakeQuantizedConv2d
from .qat import QuantizationAwareTraining, prepare_qat_model, convert_qat_model
from .ptq import (
    PostTrainingQuantization,
    ptq_quantize,
    load_quantized_model,
)
from .utils import (
    quantize_tensor,
    dequantize_tensor,
    calculate_scale_zero_point,
    fuse_modules,
    get_quantizable_layers,
    print_quantization_info,
)

__all__ = [
    # 配置
    "QuantizationConfig",
    # 观察器
    "MinMaxObserver",
    "MovingAverageObserver",
    # 伪量化
    "FakeQuantize",
    "FakeQuantizedLinear",
    "FakeQuantizedConv2d",
    # QAT
    "QuantizationAwareTraining",
    "prepare_qat_model",
    "convert_qat_model",
    # PTQ
    "PostTrainingQuantization",
    "ptq_quantize",
    "load_quantized_model",
    # 工具函数
    "quantize_tensor",
    "dequantize_tensor",
    "calculate_scale_zero_point",
    "fuse_modules",
    "get_quantizable_layers",
    "print_quantization_info",
]
