# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Fake quantization layers for QAT and PTQ flows."""

from typing import Optional

import torch
import torch.nn as nn

from .base import QuantizationConfig
from .observers import MinMaxObserver, MovingAverageObserver


class FakeQuantize(nn.Module):

    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        if config.observer_type == "minmax":
            self.observer = MinMaxObserver(config)
        elif config.observer_type == "moving_average":
            self.observer = MovingAverageObserver(config)
        else:
            raise ValueError(f"unsupported observer type{config.observer_type}")
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.fake_quant_enabled = True
        self.observer_enabled = True
        self._cache_interval = 100
        self._global_step = 0
        self._scale_cache = None
        self._last_update_step = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            self.observer.update(x.detach())
        if not self.fake_quant_enabled:
            return x
        self._global_step += 1
        if self.training:
            if (
                self._global_step % self._cache_interval == 0
                or self._last_update_step < 0
            ):
                scale, zero_point = self.observer.calculate_qparams()
                self.scale = scale
                self.zero_point = zero_point
                self._scale_cache = (scale, zero_point)
                self._last_update_step = self._global_step
        return self._fake_quantize(x, self.scale, self.zero_point)

    def _fake_quantize(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        qmin, qmax = self.config.get_qmin_qmax()
        scale_safe = torch.where(scale > 1e-08, scale, torch.ones_like(scale))
        x_quant = torch.round(x / scale_safe + zero_point)
        x_quant = torch.clamp(x_quant, qmin, qmax)
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant

    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled = enabled

    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled = enabled

    def calculate_qparams(self) -> tuple:
        return self.observer.calculate_qparams()

    def reset_observer(self) -> None:
        self.observer.reset()


class FakeQuantizedLinear(nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(in_features, out_features, bias)
        if config is None:
            config = QuantizationConfig()
        self.activation_fake_quant = FakeQuantize(config)
        self.weight_fake_quant = FakeQuantize(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        return nn.functional.linear(x, w, self.bias)


class FakeQuantizedConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(
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
        if config is None:
            config = QuantizationConfig()
        self.activation_fake_quant = FakeQuantize(config)
        self.weight_fake_quant = FakeQuantize(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        return self._conv_forward(x, w, self.bias)
