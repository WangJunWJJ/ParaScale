# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Quantization-aware training with a shared reference inference conversion."""

import copy

from .core import (
    convert_model,
    export_checkpoint,
    prepare_model,
    quantization_params,
    signature,
)
from .fake_quantize import FakeQuantize


class QuantizationAwareTraining:
    """Own a copied model/config; train the model returned by prepare."""

    def __init__(self, model, config):
        self.model = model
        self.config = copy.deepcopy(config)
        self.prepared_model = None
        self.quantized_model = None
        self._signature = signature(model)

    def prepare(self):
        self.prepared_model = prepare_model(self.model, self.config)
        self.quantized_model = None
        return self.prepared_model

    def _require_prepared(self):
        if self.prepared_model is None:
            raise RuntimeError("prepare must be called first")
        return self.prepared_model

    def freeze_observer(self):
        for module in self._require_prepared().modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(False)

    def unfreeze_observer(self):
        for module in self._require_prepared().modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(True)

    def enable_fake_quant(self, enabled=True):
        for module in self._require_prepared().modules():
            if isinstance(module, FakeQuantize):
                module.enable_fake_quant(enabled)

    def convert(self):
        self.quantized_model = convert_model(self._require_prepared(), self.config)
        return self.quantized_model

    def get_quantization_params(self):
        return quantization_params(self._require_prepared())

    def export(self, path, format="pytorch"):
        if format != "pytorch":
            raise ValueError("only pytorch checkpoint export is supported")
        model = self.convert()
        export_checkpoint(
            path,
            model,
            self.config,
            self._signature,
            self.get_quantization_params(),
            format,
        )


def prepare_qat_model(model, config):
    return QuantizationAwareTraining(model, config).prepare()


def convert_qat_model(model, config):
    """Convert observed QAT layers; reject raw float or already converted models."""
    return convert_model(model, copy.deepcopy(config))
