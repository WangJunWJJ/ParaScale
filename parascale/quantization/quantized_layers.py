# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Quantized layer implementations."""

from typing import Optional

import torch
import torch.nn as nn

from .base import QuantizationConfig


class QuantizedLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_config = quant_config
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.weight_scale: Optional[torch.Tensor] = None
        self.weight_zero_point: Optional[torch.Tensor] = None
        self.bits = getattr(quant_config, "bits", 8) if quant_config else 8
        self.qmin = -128 if self.bits == 8 else -8
        self.qmax = 127 if self.bits == 8 else 7

    def set_quantized_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
    ) -> None:
        self.weight.data = weight.to(dtype=torch.int8)
        self.weight_scale = scale
        self.weight_zero_point = (
            zero_point if zero_point is not None else torch.zeros_like(scale)
        )

    def _dequantize_weight(self) -> torch.Tensor:
        if self.weight_scale is None:
            return self.weight.float()
        weight_int = self.weight.float()
        scale = self.weight_scale.float()
        zp = (
            self.weight_zero_point.float()
            if self.weight_zero_point is not None
            else torch.zeros_like(scale)
        )
        return (weight_int - zp) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp32 = self._dequantize_weight()
        if self.bias is not None:
            return torch.nn.functional.linear(x, weight_fp32, self.bias)
        else:
            return torch.nn.functional.linear(x, weight_fp32)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class QuantizedConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        kernel_size_tuple = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size_tuple
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.quant_config = quant_config
        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels // groups, *kernel_size_tuple),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.weight_scale: Optional[torch.Tensor] = None
        self.weight_zero_point: Optional[torch.Tensor] = None
        self.bits = getattr(quant_config, "bits", 8) if quant_config else 8

    def set_quantized_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
    ) -> None:
        self.weight.data = weight.to(dtype=torch.int8)
        self.weight_scale = scale
        self.weight_zero_point = (
            zero_point if zero_point is not None else torch.zeros_like(scale)
        )

    def _dequantize_weight(self) -> torch.Tensor:
        if self.weight_scale is None:
            return self.weight.float()
        weight_int = self.weight.float()
        scale = self.weight_scale.float()
        zp = (
            self.weight_zero_point.float()
            if self.weight_zero_point is not None
            else torch.zeros_like(scale)
        )
        return (weight_int - zp) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp32 = self._dequantize_weight()
        if self.padding_mode != "zeros":
            return torch.nn.functional.conv2d(
                x,
                weight_fp32,
                self.bias,
                self.stride,
                self._reversed_padding_repeated_twice,
                self.dilation,
                self.groups,
            )
        return torch.nn.functional.conv2d(
            x,
            weight_fp32,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def _reversed_padding_repeated_twice(self) -> tuple:
        if self.padding == 0:
            return (0, 0, 0, 0)
        return (self.padding, self.padding, self.padding, self.padding)

    def extra_repr(self) -> str:
        s = f"in_channels={self.in_channels}, out_channels={self.out_channels}"
        s += f", kernel_size={self.kernel_size}, stride={self.stride}"
        s += f", padding={self.padding}, dilation={self.dilation}, groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        return s


class DynamicQuantizedLinear(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, dtype: torch.dtype = torch.qint8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self._packed_weight = None
        self.weight_scale: Optional[torch.Tensor] = None

    def set_quantized_weight(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self._packed_weight = torch.ops.quantized.linear_prepack(weight, scale)
        self.weight_scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._packed_weight is None:
            raise RuntimeError("quantized weights have not been set")
        return torch.ops.quantized.linear_dynamic(
            x, self._packed_weight, reduce_range=True
        )

    def _weight_bias(self):
        if self._packed_weight is None:
            return (None, None)
        (weight, bias), _ = torch.ops.quantized.linear_unpack(self._packed_weight)
        return (weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class StaticQuantizedLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.qint8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self._quant: Optional[torch.quantization.QuantStub] = None
        self._dequant: Optional[torch.quantization.DeQuantStub] = None
        self._fc: Optional[nn.Linear] = None
        if bias:
            self._fc = nn.Linear(in_features, out_features, bias=True)
        else:
            self._fc = nn.Linear(in_features, out_features, bias=False)
        self._quant = torch.quantization.QuantStub()
        self._dequant = torch.quantization.DeQuantStub()

    def set_quantized_weight(
        self, weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> None:
        self._fc.weight.data = weight.to(dtype=torch.float)
        if self._fc.bias is not None:
            self._fc.bias.data = torch.zeros(self.out_features)
        self._fc.scale = scale
        self._fc.zero_point = zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._quant(x)
        x = self._fc(x)
        x = self._dequant(x)
        return x

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
