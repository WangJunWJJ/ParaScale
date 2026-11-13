# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Numerical and module utilities for reference quantization."""

import torch
from torch import nn


def calculate_scale_zero_point(min_val, max_val, config):
    """Include zero in affine ranges and produce integral, valid zero points."""
    if not torch.isfinite(min_val).all() or not torch.isfinite(max_val).all():
        raise ValueError("quantization ranges must be finite")
    if (min_val > max_val).any():
        raise ValueError("minimum exceeds maximum")
    low = torch.minimum(min_val.float(), torch.zeros_like(min_val).float())
    high = torch.maximum(max_val.float(), torch.zeros_like(max_val).float())
    qmin, qmax = config.get_qmin_qmax()
    if config.scheme == "symmetric":
        scale = torch.maximum(low.abs(), high.abs()) / qmax
        zp = torch.zeros_like(scale, dtype=torch.int32)
    else:
        scale = (high - low) / (qmax - qmin)
        scale = scale.clamp_min(torch.finfo(torch.float32).eps)
        zp = (qmin - torch.round(low / scale)).clamp(qmin, qmax).int()
    scale = scale.clamp_min(torch.finfo(torch.float32).eps)
    return scale, zp


def quantize_tensor(x, scale, zero_point, config):
    """Return signed integer codes using the native affine rounding convention."""
    if scale.numel() > 1 and scale.ndim == 1 and x.ndim > 1:
        shape = (-1,) + (1,) * (x.ndim - 1)
        scale, zero_point = scale.reshape(shape), zero_point.reshape(shape)
    qmin, qmax = config.get_qmin_qmax()
    # Native fake-quant kernels multiply by reciprocal scale before rounding.
    return (
        (torch.round(x.float() * scale.reciprocal()) + zero_point)
        .clamp(qmin, qmax)
        .to(torch.int8)
    )


def dequantize_tensor(x_quant, scale, zero_point):
    if scale.numel() > 1 and scale.ndim == 1 and x_quant.ndim > 1:
        shape = (-1,) + (1,) * (x_quant.ndim - 1)
        scale, zero_point = scale.reshape(shape), zero_point.reshape(shape)
    return (x_quant.float() - zero_point) * scale


def fuse_modules(model, modules_to_fuse=None):
    """Explicit fusion helper; reference preparation never changes model topology."""
    if modules_to_fuse:
        torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    return model


def get_quantizable_layers(model):
    return [
        name
        for name, module in model.named_modules()
        if type(module) in (nn.Linear, nn.Conv2d)
    ]


def copy_model_weights(src_model, dst_model):
    with torch.no_grad():
        target = dst_model.state_dict()
        for name, value in src_model.state_dict().items():
            if name in target and value.shape == target[name].shape:
                target[name].copy_(value)


def print_quantization_info(model):
    print("Reference float-compute quantization:", get_quantizable_layers(model))
