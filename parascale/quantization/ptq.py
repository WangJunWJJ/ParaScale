# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Post-training calibration and reference inference checkpoints."""

import copy
from itertools import islice

import torch

from .core import (
    convert_model,
    export_checkpoint,
    load_checkpoint,
    prepare_model,
    quantization_params,
    signature,
    validate_observed,
)
from .fake_quantize import FakeQuantize


def _validate_batch(value):
    if isinstance(value, torch.Tensor):
        if not value.numel() or not torch.isfinite(value).all():
            raise ValueError("calibration tensors must be nonempty and finite")
    elif isinstance(value, dict):
        if not value:
            raise ValueError("empty calibration batch")
        for item in value.values():
            _validate_batch(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_batch(item)


class PostTrainingQuantization:
    """Calibrate a private copy without gradients, then encode weights once."""

    def __init__(self, model, config):
        self.model = model
        self.config = copy.deepcopy(config)
        self.calibrated_model = None
        self.quantized_model = None
        self.is_calibrated = False
        self.is_converted = False
        self._signature = signature(model)

    def prepare(self):
        self.calibrated_model = prepare_model(self.model, self.config)
        for module in self.calibrated_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_fake_quant(False)
        self.is_calibrated = self.is_converted = False
        self.quantized_model = None
        return self.calibrated_model

    def calibrate(self, calib_loader, progress_callback=None):
        if self.calibrated_model is None:
            raise RuntimeError("prepare must be called before calibration")
        self.is_calibrated = self.is_converted = False
        self.quantized_model = None
        states = [
            (module, module.training) for module in self.calibrated_model.modules()
        ]
        quantizers = [
            m for m in self.calibrated_model.modules() if isinstance(m, FakeQuantize)
        ]
        for module in quantizers:
            module.reset_observer()
            module.enable_observer(True)
            module.enable_fake_quant(False)
        count = 0
        self.calibrated_model.eval()
        try:
            with torch.no_grad():
                for batch in islice(calib_loader, self.config.calib_batches):
                    _validate_batch(batch)
                    inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                    if isinstance(inputs, dict):
                        self.calibrated_model(**inputs)
                    elif isinstance(inputs, torch.Tensor):
                        self.calibrated_model(inputs)
                    else:
                        raise ValueError(
                            "expected tensor, (inputs, labels), or keyword dictionary"
                        )
                    count += 1
                    if progress_callback:
                        progress_callback(count, self.config.calib_batches)
            if not count:
                raise ValueError("calibration iterable is empty")
            self.freeze_observer()
            for module in quantizers:
                module.enable_fake_quant(True)
            # Validate every selected layer was exercised before marking success.
            validate_observed(self.calibrated_model, self.config)
            self.is_calibrated = True
        finally:
            for module, training in states:
                module.training = training

    def freeze_observer(self):
        if self.calibrated_model is None:
            raise RuntimeError("prepare must be called first")
        for module in self.calibrated_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(False)

    def quantize_weights(self):
        """Compatibility alias: return integer-storage conversion without mutating weights."""
        return self.convert()

    def convert(self):
        if not self.is_calibrated:
            raise RuntimeError("successful calibration is required")
        self.quantized_model = convert_model(self.calibrated_model, self.config)
        self.is_converted = True
        return self.quantized_model

    def get_quantization_params(self):
        if not self.is_calibrated:
            raise RuntimeError("successful calibration is required")
        return quantization_params(self.calibrated_model)

    def export(self, save_path, format="pytorch"):
        if not self.is_converted:
            raise RuntimeError("convert must be called before export")
        export_checkpoint(
            save_path,
            self.quantized_model,
            self.config,
            self._signature,
            self.get_quantization_params(),
            format,
        )

    def evaluate(self, test_loader, criterion, device=None):
        if not self.is_converted:
            raise RuntimeError("convert must be called before evaluation")
        model = self.quantized_model
        states = [(module, module.training) for module in model.modules()]
        model.eval()
        loss_sum = correct = count = 0
        try:
            with torch.no_grad():
                for inputs, targets in test_loader:
                    if device is not None:
                        inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    loss_sum += criterion(output, targets).item() * targets.shape[0]
                    correct += (output.argmax(-1) == targets).sum().item()
                    count += targets.shape[0]
        finally:
            for module, training in states:
                module.training = training
        if not count:
            raise ValueError("evaluation iterable is empty")
        return loss_sum / count, 100 * correct / count

    def get_quantization_info(self):
        total = sum(parameter.numel() for parameter in self.model.parameters())
        selected = (
            sum(
                module.weight.numel()
                for module in self.calibrated_model.modules()
                if hasattr(module, "weight_fake_quant")
            )
            if self.calibrated_model
            else 0
        )
        return {
            "total_params": total,
            "quantized_params": selected,
            "quantization_ratio": selected / total if total else 0,
            "bits": self.config.bits,
            "scheme": self.config.scheme,
            "per_channel": self.config.per_channel,
            "compute": "floating_point_reference",
        }

    def print_quantization_info(self):
        print(self.get_quantization_info())


def ptq_quantize(model, config, calib_loader, progress_callback=None):
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader, progress_callback)
    return ptq.convert()


def load_quantized_model(checkpoint_path, model=None):
    """Return (model, config, quantization_params), without recalibration."""
    return load_checkpoint(checkpoint_path, model)
