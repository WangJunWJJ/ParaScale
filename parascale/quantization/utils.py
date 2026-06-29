# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Quantization utility functions."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import QuantizationConfig


def calculate_scale_zero_point(
    min_val: torch.Tensor, max_val: torch.Tensor, config: QuantizationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    qmin, qmax = config.get_qmin_qmax()
    if config.scheme == "symmetric":
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = max_abs / qmax
        zero_point = torch.zeros_like(scale)
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
    scale = torch.clamp(scale, min=1e-08)
    return (scale, zero_point)


def quantize_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    config: QuantizationConfig,
) -> torch.Tensor:
    qmin, qmax = config.get_qmin_qmax()
    x_quant = torch.round(x / scale + zero_point)
    x_quant = torch.clamp(x_quant, qmin, qmax)
    return x_quant


def dequantize_tensor(
    x_quant: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    return (x_quant - zero_point) * scale


def fuse_modules(model: nn.Module, modules_to_fuse: Optional[list] = None) -> nn.Module:
    if modules_to_fuse is None:
        modules_to_fuse = []
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                layers = list(module.children())
                if len(layers) >= 3:
                    if (
                        isinstance(layers[0], (nn.Conv2d, nn.Conv1d))
                        and isinstance(layers[1], nn.BatchNorm2d)
                        and isinstance(layers[2], nn.ReLU)
                    ):
                        modules_to_fuse.append([f"{name}.0", f"{name}.1", f"{name}.2"])
    if modules_to_fuse:
        try:
            torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
        except Exception as e:
            print(f"quantization utility message{e}")
    return model


def get_quantizable_layers(model: nn.Module) -> list:
    quantizable_types = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)
    quantizable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, quantizable_types):
            quantizable_layers.append(name)
    return quantizable_layers


def copy_model_weights(src_model: nn.Module, dst_model: nn.Module) -> None:
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    for name, param in src_state.items():
        if name in dst_state:
            if dst_state[name].shape == param.shape:
                dst_state[name].copy_(param)


def print_quantization_info(model: nn.Module) -> None:
    print("=" * 60)
    print("quantization utility message")
    print("=" * 60)
    total_params = 0
    quantized_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_params = sum((p.numel() for p in module.parameters()))
            total_params += num_params
            if hasattr(module, "weight_fake_quant"):
                quantized_params += num_params
                print(
                    f"{name}quantization utility message{num_params:,}quantization utility message"
                )
            else:
                print(
                    f"{name}quantization utility message{num_params:,}quantization utility message"
                )
    print("-" * 60)
    print(f"quantization utility message{total_params:,}")
    print(f"quantization utility message{quantized_params:,}")
    print(f"quantization utility message{quantized_params / total_params * 100:.2f}%")
    print("=" * 60)
