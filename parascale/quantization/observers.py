# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Observers used to estimate quantization ranges."""

from typing import Optional, Tuple

import torch

from .base import QuantizationConfig


class BaseObserver:

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.min_val: Optional[torch.Tensor] = None
        self.max_val: Optional[torch.Tensor] = None

    def update(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_val is None or self.max_val is None:
            raise ValueError("observer has not seen any data")
        qmin, qmax = self.config.get_qmin_qmax()
        if self.config.scheme == "symmetric":
            max_abs = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
            scale = max_abs / qmax
            zero_point = torch.zeros_like(scale)
        else:
            scale = (self.max_val - self.min_val) / (qmax - qmin)
            zero_point = qmin - self.min_val / scale
        scale = torch.clamp(scale, min=1e-08)
        return (scale, zero_point)

    def reset(self) -> None:
        self.min_val = None
        self.max_val = None


class MinMaxObserver(BaseObserver):

    def __init__(self, config: QuantizationConfig, window_size: int = 0):
        super().__init__(config)
        self.window_size = window_size
        self.history_min: list = []
        self.history_max: list = []

    def update(self, x: torch.Tensor) -> None:
        if self.config.per_channel and x.dim() > 1:
            dims = list(range(1, x.dim()))
            min_val = torch.min(x, dim=dims[0], keepdim=True)[0]
            max_val = torch.max(x, dim=dims[0], keepdim=True)[0]
            for dim in dims[1:]:
                min_val = torch.min(min_val, dim=dim, keepdim=True)[0]
                max_val = torch.max(max_val, dim=dim, keepdim=True)[0]
        else:
            min_val = torch.min(x)
            max_val = torch.max(x)
        if self.window_size > 0:
            self.history_min.append(min_val.detach().clone())
            self.history_max.append(max_val.detach().clone())
            if len(self.history_min) > self.window_size:
                self.history_min.pop(0)
                self.history_max.pop(0)
            self.min_val = torch.stack(self.history_min).amin(dim=0)
            self.max_val = torch.stack(self.history_max).amax(dim=0)
        elif self.min_val is None:
            self.min_val = min_val
            self.max_val = max_val
        else:
            self.min_val = torch.min(self.min_val, min_val)
            self.max_val = torch.max(self.max_val, max_val)

    def reset(self) -> None:
        super().reset()
        self.history_min = []
        self.history_max = []

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, zero_point = super().calculate_qparams()
        if self.window_size > 0 and scale.numel() > 1:
            scale = scale.max().reshape(())
            zero_point = torch.zeros_like(scale)
        return (scale, zero_point)


class MovingAverageObserver(BaseObserver):

    def __init__(self, config: QuantizationConfig):
        super().__init__(config)
        self.ratio = config.moving_average_ratio

    def update(self, x: torch.Tensor) -> None:
        if self.config.per_channel and x.dim() > 1:
            dims = list(range(1, x.dim()))
            min_val = torch.min(x, dim=dims[0], keepdim=True)[0]
            max_val = torch.max(x, dim=dims[0], keepdim=True)[0]
            for dim in dims[1:]:
                min_val = torch.min(min_val, dim=dim, keepdim=True)[0]
                max_val = torch.max(max_val, dim=dim, keepdim=True)[0]
        else:
            min_val = torch.min(x)
            max_val = torch.max(x)
        if self.min_val is None:
            self.min_val = min_val
            self.max_val = max_val
        else:
            self.min_val = self.ratio * min_val + (1 - self.ratio) * self.min_val
            self.max_val = self.ratio * max_val + (1 - self.ratio) * self.max_val
