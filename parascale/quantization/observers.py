# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Serializable min/max observers for reference quantization."""

import copy

import torch
from torch import nn


class TensorStateModule(nn.Module):
    """Allow observer buffers to acquire their channel shapes on strict reload."""

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name, value in self._buffers.items():
            incoming = state_dict.get(prefix + name)
            if (
                value is not None
                and incoming is not None
                and value.shape != incoming.shape
            ):
                self._buffers[name] = torch.empty(
                    incoming.shape, dtype=value.dtype, device=value.device
                )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _apply(self, fn, recurse=True):
        precise = {
            name: value
            for name, value in self._buffers.items()
            if value is not None and value.dtype == torch.float32
        }
        super()._apply(fn, recurse)
        for name, value in precise.items():
            self._buffers[name] = value.to(device=self._buffers[name].device)
        return self


class BaseObserver(TensorStateModule):
    """Track output-channel weight ranges or one activation range."""

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.register_buffer("min_val", torch.empty(0))
        self.register_buffer("max_val", torch.empty(0))

    def calculate_qparams(self):
        if not self.min_val.numel():
            raise ValueError("observer has not seen any data")
        from .utils import calculate_scale_zero_point

        return calculate_scale_zero_point(self.min_val, self.max_val, self.config)

    def reset(self):
        self.min_val = self.min_val.new_empty(0)
        self.max_val = self.max_val.new_empty(0)

    def _range(self, x):
        if not x.numel() or not torch.isfinite(x).all():
            raise ValueError("observer input must be nonempty and finite")
        x = x.detach().float()
        if self.config.per_channel and x.ndim > 1:
            dims = tuple(range(1, x.ndim))
            return x.amin(dims), x.amax(dims)
        return x.amin().reshape(1), x.amax().reshape(1)


class MinMaxObserver(BaseObserver):
    """Cumulative ranges, or a serialized finite observation window."""

    def __init__(self, config, window_size=0):
        super().__init__(config)
        self.window_size = window_size
        self.register_buffer("_history_min", torch.empty(0))
        self.register_buffer("_history_max", torch.empty(0))

    @property
    def history_min(self):
        return list(self._history_min.unbind()) if self._history_min.numel() else []

    @property
    def history_max(self):
        return list(self._history_max.unbind()) if self._history_max.numel() else []

    def update(self, x):
        low, high = self._range(x)
        if self.window_size:
            self._history_min = torch.stack(
                (self.history_min + [low])[-self.window_size :]
            )
            self._history_max = torch.stack(
                (self.history_max + [high])[-self.window_size :]
            )
            self.min_val = self._history_min.amin(0)
            self.max_val = self._history_max.amax(0)
        elif not self.min_val.numel():
            self.min_val, self.max_val = low, high
        else:
            self.min_val = torch.minimum(self.min_val, low)
            self.max_val = torch.maximum(self.max_val, high)

    def reset(self):
        super().reset()
        self._history_min = self._history_min.new_empty(0)
        self._history_max = self._history_max.new_empty(0)


class MovingAverageObserver(BaseObserver):
    """Exponential moving ranges using ratio as new-observation weight."""

    def update(self, x):
        low, high = self._range(x)
        ratio = self.config.moving_average_ratio
        if not self.min_val.numel():
            self.min_val, self.max_val = low, high
        else:
            self.min_val = self.min_val * (1 - ratio) + low * ratio
            self.max_val = self.max_val * (1 - ratio) + high * ratio
