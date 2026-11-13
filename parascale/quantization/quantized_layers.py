# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Integer storage with explicit floating-point reference computation."""

import copy
import math

import torch
from torch import nn

from .base import QuantizationConfig
from .observers import TensorStateModule


class _QuantizedWeight(TensorStateModule):
    """Packed signed int4 or signed int8 weights, never an integer compute claim."""

    def _initialize(self, shape, bias, config):
        self.quant_config = copy.deepcopy(config or QuantizationConfig())
        self.weight_shape = tuple(shape)
        size = math.prod(shape)
        dtype = torch.uint8 if self.quant_config.bits == 4 else torch.int8
        self.register_buffer(
            "packed_weight",
            torch.zeros((size + 1) // 2 if dtype == torch.uint8 else size, dtype=dtype),
        )
        channels = shape[0] if self.quant_config.per_channel else 1
        self.register_buffer("weight_scale", torch.ones(channels))
        self.register_buffer(
            "weight_zero_point", torch.zeros(channels, dtype=torch.int32)
        )
        self.register_buffer("bias", torch.zeros(shape[0]) if bias else None)
        self.register_buffer("_weight_dtype", torch.empty(0))
        self.activation_fake_quant = nn.Identity()

    @property
    def weight(self):
        """Read-only floating view for callers inspecting Linear/Conv2d weights."""
        return self.dequantized_weight(self._weight_dtype.dtype)

    def _apply(self, fn, recurse=True):
        dtype_marker = self._weight_dtype
        super()._apply(fn, recurse)
        # Scales retain float32 precision; the public weight follows model casts.
        self._weight_dtype = fn(dtype_marker)
        return self

    def set_quantized_weight(self, weight, scale, zero_point):
        if tuple(weight.shape) != self.weight_shape:
            raise ValueError("quantized weight shape mismatch")
        qmin, qmax = self.quant_config.get_qmin_qmax()
        if (
            not torch.isfinite(weight).all()
            or (weight < qmin).any()
            or (weight > qmax).any()
            or not torch.equal(weight, weight.round())
        ):
            raise ValueError("weight must contain in-range integers")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("scale must be finite and positive")
        flat = weight.reshape(-1).to(torch.int8)
        if self.quant_config.bits == 4:
            values = (flat.to(torch.int16) - qmin).to(torch.uint8)
            if values.numel() % 2:
                values = torch.cat((values, values.new_zeros(1)))
            self.packed_weight = values[::2] | (values[1::2] << 4)
        else:
            self.packed_weight = flat.clone()
        self.weight_scale = scale.detach().float().reshape(-1).clone()
        self.weight_zero_point = zero_point.detach().int().reshape(-1).clone()

    def dequantized_weight(self, dtype=torch.float32):
        if self.quant_config.bits == 4:
            values = torch.stack(
                (self.packed_weight & 15, self.packed_weight >> 4), -1
            ).flatten()
            values = values[: math.prod(self.weight_shape)].to(torch.int16) - 8
        else:
            values = self.packed_weight
        values = values.reshape(self.weight_shape).float()
        shape = (-1,) + (1,) * (len(self.weight_shape) - 1)
        scale = self.weight_scale.reshape(shape)
        zp = self.weight_zero_point.reshape(shape)
        return ((values - zp) * scale).to(dtype)


class QuantizedLinear(_QuantizedWeight):
    """Reference Linear with packed weights and input-dtype float compute."""

    def __init__(self, in_features, out_features, bias=True, quant_config=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self._initialize((out_features, in_features), bias, quant_config)

    def forward(self, x):
        x = self.activation_fake_quant(x)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.linear(x, self.dequantized_weight(x.dtype), bias)


class QuantizedConv2d(_QuantizedWeight):
    """Reference Conv2d including grouped and nonzero padding modes."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        quant_config=None,
    ):
        super().__init__()
        template = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        for name in (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "padding_mode",
            "_reversed_padding_repeated_twice",
        ):
            setattr(self, name, getattr(template, name))
        self._initialize(template.weight.shape, bias, quant_config)

    def forward(self, x):
        x = self.activation_fake_quant(x)
        padding = self.padding
        if self.padding_mode != "zeros":
            x = nn.functional.pad(
                x, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding = (0, 0)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.conv2d(
            x,
            self.dequantized_weight(x.dtype),
            bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


class DynamicQuantizedLinear(QuantizedLinear):
    """Legacy name for weight-only reference Linear (no dynamic acceleration)."""


class StaticQuantizedLinear(QuantizedLinear):
    """Legacy name for reference Linear."""
