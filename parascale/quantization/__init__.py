# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Lazy exports keep the configuration available without PyTorch."""

from importlib import import_module

from .base import QuantizationConfig

_EXPORTS = {
    "observers": ["MinMaxObserver", "MovingAverageObserver"],
    "fake_quantize": ["FakeQuantize", "FakeQuantizedLinear", "FakeQuantizedConv2d"],
    "quantized_layers": [
        "QuantizedLinear",
        "QuantizedConv2d",
        "DynamicQuantizedLinear",
        "StaticQuantizedLinear",
    ],
    "qat": ["QuantizationAwareTraining", "prepare_qat_model", "convert_qat_model"],
    "ptq": ["PostTrainingQuantization", "ptq_quantize", "load_quantized_model"],
    "utils": [
        "quantize_tensor",
        "dequantize_tensor",
        "calculate_scale_zero_point",
        "fuse_modules",
        "get_quantizable_layers",
        "print_quantization_info",
    ],
}
__all__ = [
    "QuantizationConfig",
    *[name for names in _EXPORTS.values() for name in names],
]


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(name)
