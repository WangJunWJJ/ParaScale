# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit fused INT4 execution through TorchAO, with no float fallback."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import nn

from .inference_config import InferenceQuantizationConfig


class PackedInt4Linear(nn.Module):
    """Aligned inference Linear with direct fused-kernel dispatch.

    TorchAO chooses quantization parameters and packs once. Keeping the packed
    buffers here avoids tensor-subclass dispatch and zero-padding allocations on
    every forward. No full floating-point weight is retained or reconstructed.
    """

    def __init__(self, source, group_size):
        super().__init__()
        weight = source.weight
        if not hasattr(weight, "qdata") or list(weight.block_size) != [1, group_size]:
            raise ValueError("Expected matching tile-packed INT4 weights")
        self.in_features, self.out_features = source.in_features, source.out_features
        if self.in_features % 1024 or self.out_features % 8:
            raise ValueError("PackedInt4Linear requires aligned dimensions")
        self.group_size = group_size
        self.register_buffer("packed_weight", weight.qdata.detach())
        self.register_buffer("scale_and_zero", weight.scale_and_zero.detach())
        self.register_buffer(
            "bias", source.bias.detach() if source.bias is not None else None
        )
        self.eval()

    @property
    def weight(self):
        """Read-only packed view for parents using F.linear or inspecting dtype."""
        from torchao.quantization.quantize_.workflows import Int4TilePackedTo4dTensor

        return Int4TilePackedTo4dTensor(
            self.packed_weight,
            self.scale_and_zero,
            [1, self.group_size],
            torch.Size((self.out_features, self.in_features)),
        )

    def forward(self, inputs):
        if (
            inputs.dtype != torch.bfloat16
            or self.scale_and_zero.dtype != torch.bfloat16
        ):
            raise ValueError(
                "PackedInt4Linear requires bfloat16 activations and scales"
            )
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("PackedInt4Linear input feature dimension mismatch")
        output_shape = (*inputs.shape[:-1], self.out_features)
        if inputs.numel() == 0:
            return inputs.new_empty(output_shape)
        output = torch.ops.aten._weight_int4pack_mm(
            inputs.reshape(-1, self.in_features).contiguous(),
            self.packed_weight,
            self.group_size,
            self.scale_and_zero,
        ).reshape(output_shape)
        return output + self.bias if self.bias is not None else output


def _excluded(name, exclusions):
    parts = name.split(".")
    return any(
        suffix == pattern or suffix.startswith(pattern + ".")
        for index in range(len(parts))
        for suffix in [".".join(parts[index:])]
        for pattern in exclusions
    )


def _select_linear_layers(model, config):
    functional_parents = {
        name
        for name, module in model.named_modules()
        if isinstance(
            module,
            (
                nn.MultiheadAttention,
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
            ),
        )
    }
    # Include aliases: an excluded output head may share a weight or whole module.
    references = Counter(
        id(parameter) for _, parameter in model.named_parameters(remove_duplicate=False)
    )
    selected, skipped = {}, {}
    for name, layer in model.named_modules(remove_duplicate=False):
        if type(layer) is not nn.Linear:
            continue
        if _excluded(name, config.exclude_modules):
            skipped[name] = "excluded"
        elif config.backend == "gemlite_int8" and any(
            not parent or name.startswith(parent + ".") for parent in functional_parents
        ):
            skipped[name] = "functional_weight_access"
        elif references[id(layer.weight)] > 1:
            skipped[name] = "shared_parameter"
        elif layer._forward_pre_hooks or layer._forward_hooks:
            skipped[name] = "execution_hooks"
        elif (
            layer.in_features % (32 if config.backend == "gemlite_int8" else 1024)
            or layer.out_features % 8
        ):
            # TorchAO pads K to 1024 and N to 8. Avoid paying for padding each call.
            skipped[name] = "unaligned_shape"
        elif type(layer.weight) is not nn.Parameter:
            skipped[name] = "already_transformed_weight"
        else:
            selected[name] = layer
    return selected, skipped


def _load_torchao():
    try:
        ao_version = version("torchao")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Fused INT4 requires the optional parascale[inference-int4] dependencies "
            "in a separate CUDA environment (torch 2.9.1, torchao 0.15.0)."
        ) from exc
    if (
        torch.__version__.split("+")[0] != "2.9.1"
        or ao_version.split("+")[0] != "0.15.0"
    ):
        raise RuntimeError(
            f"Fused INT4 adapter targets torch 2.9.1 / torchao 0.15.0; "
            f"found torch {torch.__version__} / torchao {ao_version}. "
            "Use a separate matching environment; do not upgrade shared training jobs."
        )
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from torchao.quantization.quantize_.workflows import Int4TilePackedTo4dTensor

    return quantize_, Int4WeightOnlyConfig, Int4TilePackedTo4dTensor, ao_version


def quantize_inference_model(model, config=None, *, inplace=False):
    """Return ``(model, report)`` with fused INT4 Linear weights.

    Call ``model.eval().to(device='cuda', dtype=torch.bfloat16)`` first. Run the
    returned model under ``torch.inference_mode()`` or ParaScale InferenceEngine.
    Compile the returned model if desired; this function does not change global
    compiler settings. No performance claim is made by conversion alone.

    By default the source is copied. ``inplace=True`` avoids a second full model
    allocation, but a backend conversion failure may leave it partly converted.
    Re-load the original checkpoint after such a failure. This runtime format
    does not use the reference PTQ/QAT exporter; reapply to the float checkpoint.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    config = InferenceQuantizationConfig() if config is None else config
    if not isinstance(config, InferenceQuantizationConfig):
        raise TypeError("config must be an InferenceQuantizationConfig")
    if type(inplace) is not bool:
        raise TypeError("inplace must be a boolean")
    if any(layer.training for layer in model.modules()):
        raise ValueError("Call model.eval() before inference quantization")
    selected, skipped = _select_linear_layers(model, config)
    if not selected:
        raise ValueError(f"No eligible Linear layers for fused INT4: {skipped}")
    devices = {tensor.device for tensor in (*model.parameters(), *model.buffers())}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("Fused INT4 requires a model on one CUDA device")
    device = next(iter(devices))
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("Fused INT4 requires CUDA compute capability >= 8.0")
    for name, layer in selected.items():
        if layer.weight.dtype != torch.bfloat16:
            raise ValueError(
                f"{name}: fused INT4 requires bfloat16 weights/activations"
            )
        if not torch.isfinite(layer.weight).all():
            raise ValueError(f"{name}: weights must be finite")
    if config.backend == "gemlite_int8":
        from .inference_gemlite import convert_int8

        return convert_int8(model, config, selected, skipped, inplace=inplace)
    quantize_, config_type, packed_type, ao_version = _load_torchao()
    source_bytes = sum(
        layer.weight.numel() * layer.weight.element_size()
        for layer in selected.values()
    )
    result = model if inplace else copy.deepcopy(model)
    names = set(selected)
    recipe = config_type(
        group_size=config.group_size,
        version=2,
        int4_packing_format="tile_packed_to_4d",
        int4_choose_qparams_algorithm="hqq",
        set_inductor_config=False,
    )
    with torch.no_grad(), torch.cuda.device(device):
        quantize_(result, recipe, filter_fn=lambda module, name: name in names)
    converted = dict(result.named_modules())
    packed_bytes = 0
    for name in selected:
        weight = converted[name].weight
        if not isinstance(weight, packed_type):
            raise RuntimeError(
                f"{name}: fused INT4 conversion did not install packed weights"
            )
        packed_bytes += sum(
            t.numel() * t.element_size() for t in (weight.qdata, weight.scale_and_zero)
        )
        packed_layer = PackedInt4Linear(converted[name], config.group_size)
        if name:
            result.set_submodule(name, packed_layer)
        else:
            result = packed_layer
    return result, {
        "backend": "torchao",
        "execution": "int4_weight_only_fused_bf16",
        "kernel": "aten._weight_int4pack_mm",
        "dispatch": "packed_buffers_direct_aten",
        "torch_version": torch.__version__,
        "torchao_version": ao_version,
        "config": asdict(config),
        "converted_layers": list(selected),
        "skipped_layers": skipped,
        "source_weight_bytes": source_bytes,
        "packed_weight_bytes": packed_bytes,
        "performance_validated": False,
    }
