# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Quantization-aware training helpers."""

from typing import Any, Dict, Optional

import torch.nn as nn

from .base import QuantizationConfig
from .fake_quantize import FakeQuantize, FakeQuantizedConv2d, FakeQuantizedLinear
from .utils import copy_model_weights, fuse_modules, get_quantizable_layers


class QuantizationAwareTraining:

    def __init__(self, model: nn.Module, config: QuantizationConfig):
        self.model = model
        self.config = config
        self.prepared_model: Optional[nn.Module] = None

    def prepare(self) -> nn.Module:
        self.prepared_model = self._copy_model(self.model)
        if self.config.fuse_modules:
            fuse_modules(self.prepared_model)
        self._insert_fake_quantize(self.prepared_model)
        copy_model_weights(self.model, self.prepared_model)
        return self.prepared_model

    def _copy_model(self, model: nn.Module) -> nn.Module:
        import copy

        return copy.deepcopy(model)

    def _insert_fake_quantize(self, model: nn.Module) -> None:
        quantizable_layers = get_quantizable_layers(model)
        for name in quantizable_layers:
            *parent_path, layer_name = name.split(".")
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            original_layer = getattr(parent, layer_name)
            if isinstance(original_layer, nn.Linear):
                quantized_layer = FakeQuantizedLinear(
                    original_layer.in_features,
                    original_layer.out_features,
                    bias=original_layer.bias is not None,
                    config=self.config,
                )
                quantized_layer.weight.data = original_layer.weight.data.clone()
                if original_layer.bias is not None:
                    quantized_layer.bias.data = original_layer.bias.data.clone()
                setattr(parent, layer_name, quantized_layer)
            elif isinstance(original_layer, nn.Conv2d):
                quantized_layer = FakeQuantizedConv2d(
                    original_layer.in_channels,
                    original_layer.out_channels,
                    original_layer.kernel_size,
                    stride=original_layer.stride,
                    padding=original_layer.padding,
                    dilation=original_layer.dilation,
                    groups=original_layer.groups,
                    bias=original_layer.bias is not None,
                    padding_mode=original_layer.padding_mode,
                    config=self.config,
                )
                quantized_layer.weight.data = original_layer.weight.data.clone()
                if original_layer.bias is not None:
                    quantized_layer.bias.data = original_layer.bias.data.clone()
                setattr(parent, layer_name, quantized_layer)

    def freeze_observer(self) -> None:
        if self.prepared_model is None:
            raise RuntimeError("QAT operation cannot continue")
        for module in self.prepared_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(False)

    def unfreeze_observer(self) -> None:
        if self.prepared_model is None:
            raise RuntimeError("QAT operation cannot continue")
        for module in self.prepared_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(True)

    def enable_fake_quant(self, enabled: bool = True) -> None:
        if self.prepared_model is None:
            raise RuntimeError("QAT operation cannot continue")
        for module in self.prepared_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_fake_quant(enabled)

    def convert(self) -> nn.Module:
        if self.prepared_model is None:
            raise RuntimeError("QAT operation cannot continue")
        return self.prepared_model

    def get_quantization_params(self) -> Dict[str, Any]:
        if self.prepared_model is None:
            raise RuntimeError("QAT operation cannot continue")
        params = {}
        for name, module in self.prepared_model.named_modules():
            if isinstance(module, FakeQuantize):
                scale = module.scale
                zero_point = module.zero_point
                params[name] = {
                    "scale": scale.cpu().numpy().tolist(),
                    "zero_point": zero_point.cpu().numpy().tolist(),
                }
        return params


def prepare_qat_model(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    qat = QuantizationAwareTraining(model, config)
    return qat.prepare()


def convert_qat_model(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            module.enable_observer(False)
    return model
