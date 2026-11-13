# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Native fake-quantization STE and trainable reference layers."""

import copy
import hashlib
import json

import torch
from torch import nn

from .base import QuantizationConfig
from .observers import MinMaxObserver, MovingAverageObserver, TensorStateModule


class FakeQuantize(TensorStateModule):
    """Fake quantization with tensor-only observer and enable state."""

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        observer = (
            MinMaxObserver
            if config.observer_type == "minmax"
            else MovingAverageObserver
        )
        self.observer = observer(self.config)
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("fake_quant_enabled", torch.tensor(True))
        self.register_buffer("observer_enabled", torch.tensor(True))
        numerical_config = {
            name: getattr(config, name)
            for name in (
                "bits",
                "scheme",
                "per_channel",
                "quantize_activations",
                "observer_type",
                "moving_average_ratio",
            )
        }
        fingerprint = hashlib.sha256(
            json.dumps(numerical_config, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).digest()
        self.register_buffer(
            "config_fingerprint", torch.tensor(list(fingerprint), dtype=torch.uint8)
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        incoming = state_dict.get(prefix + "config_fingerprint")
        if (
            not isinstance(incoming, torch.Tensor)
            or incoming.dtype != self.config_fingerprint.dtype
            or incoming.shape != self.config_fingerprint.shape
            or not torch.equal(incoming.cpu(), self.config_fingerprint.cpu())
        ):
            raise RuntimeError(
                f"quantization numerical configuration mismatch: {prefix}"
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        if self.observer_enabled:
            self.observer.update(x)
            scale, zp = self.observer.calculate_qparams()
            self.scale = scale
            self.zero_point = zp
        if not self.fake_quant_enabled:
            return x
        if not self.observer.min_val.numel():
            raise ValueError("fake quantizer has not been calibrated")
        return self._fake_quantize(x, self.scale, self.zero_point)

    def _fake_quantize(self, x, scale, zero_point):
        qmin, qmax = self.config.get_qmin_qmax()
        work = x.float()
        if self.config.per_channel and x.ndim > 1:
            result = torch.fake_quantize_per_channel_affine(
                work,
                scale.float().reshape(-1),
                zero_point.int().reshape(-1),
                0,
                qmin,
                qmax,
            )
        else:
            result = torch.fake_quantize_per_tensor_affine(
                work,
                scale.float().reshape(()),
                zero_point.int().reshape(()),
                qmin,
                qmax,
            )
        return result.to(x.dtype)

    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled.fill_(enabled)

    def enable_observer(self, enabled=True):
        self.observer_enabled.fill_(enabled)

    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    def reset_observer(self):
        self.observer.reset()


def attach_quantizers(layer, config):
    layer.weight_fake_quant = FakeQuantize(config).to(layer.weight.device)
    activation_config = copy.deepcopy(config)
    activation_config.per_channel = False
    layer.activation_fake_quant = FakeQuantize(activation_config).to(
        layer.weight.device
    )
    layer.quantize_activations = config.quantize_activations


class FakeQuantizedLinear(nn.Linear):
    """Linear with native STE for weights and optional input activations."""

    def __init__(self, in_features, out_features, bias=True, config=None):
        super().__init__(in_features, out_features, bias)
        attach_quantizers(self, config or QuantizationConfig())

    def forward(self, x):
        if self.quantize_activations:
            x = self.activation_fake_quant(x)
        weight = self.weight_fake_quant(self.weight).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.linear(x, weight, bias)


class FakeQuantizedConv2d(nn.Conv2d):
    """Conv2d retaining padding semantics with native STE."""

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
        config=None,
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
        attach_quantizers(self, config or QuantizationConfig())

    def forward(self, x):
        if self.quantize_activations:
            x = self.activation_fake_quant(x)
        weight = self.weight_fake_quant(self.weight).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return self._conv_forward(x, weight, bias)
