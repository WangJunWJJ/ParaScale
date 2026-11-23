# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午7:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Fuse canonical reference INT8 artifacts without changing stored quantization."""

import copy

import torch
from torch import nn

from .execution_contracts import validate_artifact
from .fake_quantize import FakeQuantizedConv2d, FakeQuantizedLinear
from .inference_gemlite import GemliteInt8Linear, _load_gemlite
from .quantized_layers import QuantizedConv2d, QuantizedLinear


def _validate_layer(layer, name):
    config = layer.quant_config
    if config.bits != 8:
        raise ValueError(f"{name}: fused reference execution requires INT8 weights")
    if not config.per_channel:
        raise ValueError(
            f"{name}: fused reference execution requires per-channel scales"
        )
    if (
        config.quantize_activations
        or type(layer.activation_fake_quant) is not nn.Identity
    ):
        raise ValueError(f"{name}: fused reference execution is weight-only")
    if (
        type(layer.in_features) is not int
        or type(layer.out_features) is not int
        or layer.in_features <= 0
        or layer.out_features <= 0
        or layer.in_features % 32
        or layer.out_features % 8
    ):
        raise ValueError(
            f"{name}: fused reference dimensions must be aligned to K=32,N=8"
        )
    codes, scales, zeros = (
        layer.packed_weight,
        layer.weight_scale,
        layer.weight_zero_point,
    )
    if any(t.device.type == "meta" for t in (codes, scales, zeros)):
        raise ValueError(f"{name}: reference tensors must be materialized, not meta")
    if (
        codes.dtype != torch.int8
        or codes.shape != (layer.in_features * layer.out_features,)
        or layer.weight_shape != (layer.out_features, layer.in_features)
    ):
        raise ValueError(f"{name}: invalid reference INT8 codes")
    if (
        scales.dtype != torch.float32
        or scales.shape != (layer.out_features,)
        or not torch.isfinite(scales).all()
        or (scales <= 0).any()
    ):
        raise ValueError(f"{name}: invalid reference scale")
    if (
        zeros.dtype != torch.int32
        or zeros.shape != (layer.out_features,)
        or zeros.any()
    ):
        raise ValueError(f"{name}: only zero reference zero-points are supported")
    if layer.bias is not None and (
        layer.bias.device.type == "meta"
        or not layer.bias.is_floating_point()
        or layer.bias.shape != (layer.out_features,)
        or not torch.isfinite(layer.bias).all()
    ):
        raise ValueError(f"{name}: invalid reference bias")
    low, high = torch.aminmax(codes.reshape(layer.weight_shape), dim=1)
    magnitude = torch.maximum(low.float().abs(), high.float().abs()) * scales
    if not torch.isfinite(magnitude.bfloat16()).all() or (
        layer.bias is not None and not torch.isfinite(layer.bias.bfloat16()).all()
    ):
        raise ValueError(f"{name}: reference weights or bias overflow BF16")


def _select_reference_layers(model):
    functional_parents = {
        name
        for name, layer in model.named_modules()
        if isinstance(
            layer,
            (
                nn.MultiheadAttention,
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
            ),
        )
    }
    selected, identities = {}, set()
    for name, layer in model.named_modules(remove_duplicate=False):
        if isinstance(layer, (FakeQuantizedLinear, FakeQuantizedConv2d)):
            raise ValueError(f"{name}: QAT/PTQ layers must be converted before fusion")
        if not isinstance(layer, (QuantizedLinear, QuantizedConv2d)):
            continue
        if type(layer) is not QuantizedLinear:
            raise ValueError(
                f"{name}: fused reference execution supports exact QuantizedLinear only"
            )
        if id(layer) in identities:
            raise ValueError(f"{name}: shared reference modules are unsupported")
        identities.add(id(layer))
        if any(
            not parent or name.startswith(parent + ".") for parent in functional_parents
        ):
            raise ValueError(f"{name}: functional weight access is unsupported")
        _validate_layer(layer, name or "<root>")
        selected[name] = layer
    if not selected:
        raise ValueError("no converted reference INT8 layers found")
    return selected


def _pack_reference_layer(layer, processor):
    # GemLite may retain views of its inputs. Never alias the canonical artifact.
    codes = layer.packed_weight.reshape(layer.weight_shape).detach().clone()
    scales = layer.weight_scale.detach().clone()
    bias = (
        layer.bias.detach().to(dtype=torch.bfloat16, copy=True)
        if layer.bias is not None
        else None
    )
    kernel = processor(
        device=str(codes.device), dtype=torch.bfloat16, fp32_scale=True
    ).from_weights(weight=codes, scales=scales, bias=bias)
    if (
        kernel.W_q.dtype != torch.int8
        or kernel.scales.dtype != torch.float32
        or kernel.W_q.device != layer.packed_weight.device
        or kernel.scales.device != layer.weight_scale.device
        or not torch.equal(
            kernel.W_q.t(), layer.packed_weight.reshape(layer.weight_shape)
        )
        or not torch.equal(kernel.scales.reshape(-1), layer.weight_scale)
        or kernel.zeros.numel() != 0
        or kernel.W_group_mode != 2
        or kernel.channel_scale_mode != 0
        or (bias is None) != (kernel.bias is None)
        or (
            bias is not None
            and (
                kernel.bias.dtype != torch.bfloat16
                or kernel.bias.device != layer.bias.device
                or not torch.equal(kernel.bias, layer.bias.bfloat16())
            )
        )
    ):
        raise RuntimeError(
            "GemLite changed the stored reference quantization parameters"
        )
    return GemliteInt8Linear(kernel)


def prepare_quantized_inference_model(model, *, inplace=False):
    """Pack loaded PTQ/QAT INT8 codes for GemLite BF16 CUDA inference.

    Move the converted/reference-loaded model to CUDA BF16 and eval mode first.
    Every reference quantized layer must satisfy this backend's contract; no
    layer is silently skipped. Codes and FP32 scales are not requantized.
    BF16 kernel numerics still require quality validation. Use inference_mode
    when executing the result, and compile only after preparation.

    Persist the canonical reference checkpoint, not this runtime representation.
    With inplace=True a backend failure can leave a partially replaced model;
    reload the canonical checkpoint before retrying. Invalid source metadata is
    rejected before any replacement. Default copying preserves the source.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if type(inplace) is not bool:
        raise TypeError("inplace must be a boolean")
    if any(layer.training for layer in model.modules()):
        raise ValueError("Call model.eval() before artifact fusion")
    validate_artifact(model)
    selected = _select_reference_layers(model)
    devices = {t.device for t in (*model.parameters(), *model.buffers())}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("Artifact fusion requires one CUDA device")
    device = next(iter(devices))
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("Artifact fusion requires CUDA compute capability >= 8.0")
    if any(p.dtype != torch.bfloat16 for p in model.parameters()) or any(
        layer._weight_dtype.dtype != torch.bfloat16 for layer in selected.values()
    ):
        raise ValueError("Move the reference model to BF16 before artifact fusion")
    processor, gem_version = _load_gemlite()
    result = model if inplace else copy.deepcopy(model)
    modules = dict(result.named_modules())
    packed_bytes = 0
    with torch.no_grad(), torch.cuda.device(device):
        for name in selected:
            replacement = _pack_reference_layer(modules[name], processor)
            packed_bytes += sum(
                t.numel() * t.element_size()
                for t in (*replacement.parameters(), *replacement.buffers())
            )
            if name:
                result.set_submodule(name, replacement)
            else:
                result = replacement
    return result, {
        "backend": "gemlite",
        "execution": "int8_weight_only_fused_bf16",
        "source_format": "parascale.reference_quantization",
        "requantized": False,
        "stored_codes_and_scales_preserved": True,
        "bias_policy": "cast_to_bfloat16_like_reference_forward",
        "converted_layers": list(selected),
        "packed_weight_bytes": packed_bytes,
        "gemlite_version": gem_version,
        "torch_version": torch.__version__,
        "performance_validated": False,
    }
