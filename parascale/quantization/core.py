# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Shared preparation, conversion, and strict atomic checkpoint format."""

import copy
import os
import tempfile

import torch
from torch import nn

from .base import QuantizationConfig
from .fake_quantize import (
    FakeQuantize,
    FakeQuantizedConv2d,
    FakeQuantizedLinear,
    attach_quantizers,
)
from .quantized_layers import QuantizedConv2d, QuantizedLinear
from .utils import quantize_tensor

_FAKE = (FakeQuantizedLinear, FakeQuantizedConv2d)
_QUANT = (QuantizedLinear, QuantizedConv2d)

_VOLATILE_CONFIG_FIELDS = {
    "_name_or_path",
    "name_or_path",
    "_commit_hash",
    "transformers_version",
    "torch_dtype",
    "dtype",
    "device",
    "device_map",
    "_attn_implementation",
    "_attn_implementation_internal",
    "_attn_implementation_autoset",
    "_from_model_config",
}


def _normalize_semantics(value):
    """Produce checkpoint-safe config values, excluding deployment metadata."""
    if isinstance(value, dict):
        return {
            str(key): _normalize_semantics(item)
            for key, item in value.items()
            if key not in _VOLATILE_CONFIG_FIELDS
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_semantics(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    raise ValueError(
        f"unsupported inference configuration value: {type(value).__name__}"
    )


def signature(model):
    """Describe topology, shapes, PyTorch constants, and normalized HF configs.

    Arbitrary custom forward behavior cannot be inferred from a state dictionary.
    Models exposing a config.to_dict() include that semantic contract as well.
    """
    modules = []
    for name, module in model.named_modules():
        entry = {
            "name": name,
            "type": type(module).__module__ + "." + type(module).__qualname__,
        }
        attributes = set(getattr(module, "__constants__", ()))
        attributes.update(
            attr
            for attr in (
                "eps",
                "variance_epsilon",
                "stride",
                "padding",
                "dilation",
                "groups",
                "padding_mode",
                "kernel_size",
                "ceil_mode",
                "count_include_pad",
                "divisor_override",
                "output_size",
                "normalized_shape",
                "num_groups",
            )
            if hasattr(module, attr)
        )
        entry["semantics"] = {
            attr: _normalize_semantics(getattr(module, attr))
            for attr in sorted(attributes)
            if hasattr(module, attr)
            and not isinstance(getattr(module, attr), (torch.Tensor, nn.Module))
        }
        config = getattr(module, "config", None)
        if callable(getattr(config, "to_dict", None)):
            entry["config"] = _normalize_semantics(config.to_dict())
        modules.append(entry)
    return {
        "modules": modules,
        "state": {
            name: list(value.shape) for name, value in model.state_dict().items()
        },
    }


def replace(model, name, replacement):
    if not name:
        return replacement
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, replacement)
    return model


def selected_layers(model, config):
    selected = []
    for name, module in model.named_modules(remove_duplicate=False):
        excluded = any(
            name == item or name.startswith(item + ".") or name.split(".")[-1] == item
            for item in config.exclude_modules
        )
        if excluded:
            continue
        if type(module).__name__ in config.quantizable_layers:
            if type(module) not in (nn.Linear, nn.Conv2d):
                raise ValueError(f"unsupported selected layer: {name}")
            selected.append((name, module))
        elif isinstance(module, (nn.Linear, nn.Conv2d)) and any(
            isinstance(module, cls) and cls.__name__ in config.quantizable_layers
            for cls in (nn.Linear, nn.Conv2d)
        ):
            raise ValueError(f"custom Linear/Conv2d subclass unsupported: {name}")
    if not selected:
        raise ValueError("no supported layers selected")
    all_params = list(model.named_parameters(remove_duplicate=False))
    for name, module in selected:
        for _, parameter in module.named_parameters(recurse=False):
            if parameter.device.type == "meta":
                raise ValueError("meta parameters are unsupported")
            aliases = [
                path
                for path, candidate in all_params
                if candidate.device == parameter.device
                and candidate.untyped_storage().data_ptr()
                == parameter.untyped_storage().data_ptr()
            ]
            if len(aliases) > 1:
                raise ValueError(
                    f"tied selected parameters are unsupported: {name}: {aliases}"
                )
    return selected


def prepare_model(model, config):
    selected = selected_layers(model, config)
    result = copy.deepcopy(model)
    for name, _ in selected:
        layer = result.get_submodule(name) if name else result
        layer.__class__ = (
            FakeQuantizedLinear if type(layer) is nn.Linear else FakeQuantizedConv2d
        )
        attach_quantizers(layer, config)
    return result


def make_quantized(layer, config, calibrated=True):
    if isinstance(layer, (nn.Linear, FakeQuantizedLinear)):
        result = QuantizedLinear(
            layer.in_features, layer.out_features, layer.bias is not None, config
        )
    else:
        result = QuantizedConv2d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
            layer.padding_mode,
            config,
        )
    result = result.to(device=layer.weight.device, dtype=layer.weight.dtype)
    if layer.bias is not None:
        result.bias = layer.bias.detach().clone()
    if calibrated:
        fq = layer.weight_fake_quant
        if not fq.observer.min_val.numel() or not fq.fake_quant_enabled:
            raise ValueError(
                "conversion requires observed, enabled weight quantization"
            )
        result.set_quantized_weight(
            quantize_tensor(layer.weight, fq.scale, fq.zero_point, config),
            fq.scale,
            fq.zero_point,
        )
    if config.quantize_activations:
        if calibrated:
            activation = copy.deepcopy(layer.activation_fake_quant)
            if (
                not activation.observer.min_val.numel()
                or not activation.fake_quant_enabled
            ):
                raise ValueError(
                    "conversion requires observed, enabled activation quantization"
                )
            activation.enable_observer(False)
        else:
            activation_config = copy.deepcopy(config)
            activation_config.per_channel = False
            activation = FakeQuantize(activation_config).to(layer.weight.device)
            activation(torch.zeros(1, device=layer.weight.device))
            activation.enable_observer(False)
        result.activation_fake_quant = activation
    return result


def convert_model(model, config):
    validate_observed(model, config)
    result = copy.deepcopy(model)
    for name, layer in list(result.named_modules()):
        if isinstance(layer, _FAKE):
            result = replace(result, name, make_quantized(layer, config))
    return result.eval()


def validate_observed(model, config):
    """Check readiness without allocating or encoding another model."""
    found_prepared = False
    for name, layer in model.named_modules():
        if isinstance(layer, _QUANT):
            raise ValueError(
                "conversion requires prepared layers, not already quantized layers"
            )
        if not isinstance(layer, _FAKE):
            continue
        found_prepared = True
        if layer.weight_fake_quant.config.to_dict() != config.to_dict():
            raise ValueError(
                f"conversion configuration must match the prepared quantizers: {name}"
            )
        if not torch.isfinite(layer.weight).all() or (
            layer.bias is not None and not torch.isfinite(layer.bias).all()
        ):
            raise ValueError(f"selected layer parameters must be finite: {name}")
        quantizers = [layer.weight_fake_quant]
        if config.quantize_activations:
            quantizers.append(layer.activation_fake_quant)
        for quantizer in quantizers:
            if (
                not quantizer.observer.min_val.numel()
                or not quantizer.fake_quant_enabled
            ):
                raise ValueError(
                    f"selected layer requires observed, enabled quantization: {name}"
                )
            if (
                not torch.isfinite(quantizer.scale).all()
                or (quantizer.scale <= 0).any()
            ):
                raise ValueError(f"invalid quantization scale: {name}")
    if not found_prepared:
        raise ValueError("conversion requires at least one prepared quantization layer")


def quantization_params(model):
    return {
        name: {
            "scale": module.scale.detach().cpu().tolist(),
            "zero_point": module.zero_point.detach().cpu().tolist(),
        }
        for name, module in model.named_modules()
        if isinstance(module, FakeQuantize) and module.observer.min_val.numel()
    }


def export_checkpoint(
    path, model, config, template_signature, params, format="pytorch"
):
    """Atomically replace a versioned checkpoint; never serialize Python model objects."""
    if format != "pytorch":
        raise ValueError("only pytorch checkpoint export is supported")
    checkpoint = {
        "schema_version": 2,
        "format": "parascale.reference_quantization",
        "config": config.to_dict(),
        "manifest": template_signature,
        "quantization_params": params,
        "model_state_dict": model.state_dict(),
        "selected_modules": [
            name for name, module in model.named_modules() if isinstance(module, _QUANT)
        ],
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".quant-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as output:
            torch.save(checkpoint, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model):
    if model is None:
        raise ValueError("a float model template is required")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "format",
        "config",
        "manifest",
        "quantization_params",
        "model_state_dict",
        "selected_modules",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != required:
        raise ValueError("invalid checkpoint schema")
    if (
        checkpoint["schema_version"] != 2
        or checkpoint["format"] != "parascale.reference_quantization"
    ):
        raise ValueError("unsupported checkpoint format or schema")
    config = QuantizationConfig.from_dict(checkpoint["config"])
    if checkpoint["manifest"] != signature(model):
        raise ValueError("checkpoint template manifest mismatch")
    selected = selected_layers(model, config)
    if checkpoint["selected_modules"] != [name for name, _ in selected]:
        raise ValueError("checkpoint selection manifest mismatch")
    result = copy.deepcopy(model)
    for name, _ in selected:
        layer = result.get_submodule(name) if name else result
        result = replace(result, name, make_quantized(layer, config, calibrated=False))
    expected = result.state_dict()
    quantization_keys = {}
    for module_name, module in result.named_modules():
        prefix = module_name + "." if module_name else ""
        if isinstance(module, _QUANT):
            for attr in ("weight_scale", "weight_zero_point"):
                quantization_keys[prefix + attr] = attr
        if isinstance(module, FakeQuantize):
            for attr in (
                "scale",
                "zero_point",
                "observer_enabled",
                "fake_quant_enabled",
            ):
                quantization_keys[prefix + attr] = attr
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("checkpoint state keys mismatch")
    for name, value in state.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape:
            raise ValueError(f"checkpoint tensor shape mismatch: {name}")
        if value.dtype != target.dtype:
            raise ValueError(f"checkpoint tensor dtype mismatch: {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite checkpoint tensor: {name}")
        kind = quantization_keys.get(name, "")
        if kind in ("scale", "weight_scale") and (value <= 0).any():
            raise ValueError(f"invalid quantization scale: {name}")
        if kind in ("zero_point", "weight_zero_point"):
            low, high = config.get_qmin_qmax()
            if (value < low).any() or (value > high).any():
                raise ValueError(f"invalid zero point: {name}")
        if kind == "observer_enabled" and value.item():
            raise ValueError("inference checkpoint cannot enable observers")
        if kind == "fake_quant_enabled" and not value.item():
            raise ValueError("inference checkpoint cannot disable quantization")
    result.load_state_dict(state, strict=True)
    return result.eval(), config, checkpoint["quantization_params"]
