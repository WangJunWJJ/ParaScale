# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Optional channelwise INT8 weights with fused GemLite CUDA execution."""

import copy
from dataclasses import asdict
from importlib.metadata import version

import torch
from torch import nn


class GemliteInt8Linear(nn.Module):
    """Inference-only Linear.forward adapter; functional weight access unsupported."""

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel.requires_grad_(False).eval()
        self.in_features, self.out_features = kernel.in_features, kernel.out_features
        self.eval()

    @property
    def weight(self):
        raise RuntimeError(
            "INT8 requires Linear.forward; functional weight access is unsupported"
        )

    def forward(self, inputs):
        if inputs.dtype != torch.bfloat16 or self.kernel.scales.dtype != torch.float32:
            raise ValueError(
                "INT8 requires BF16 inputs and FP32 scales; set dtype before conversion"
            )
        if inputs.device != self.kernel.W_q.device:
            raise ValueError("INT8 inputs and weights must share a device")
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("INT8 input feature dimension mismatch")
        if inputs.numel() == 0:
            return inputs.new_empty((*inputs.shape[:-1], self.out_features))
        return self.kernel(inputs)


def convert_int8(model, config, selected, skipped, *, inplace):
    gem_version = version("gemlite")
    if (
        torch.__version__.split("+")[0] != "2.9.1"
        or gem_version != "0.6.0.post2"
        or version("triton") != "3.5.1"
    ):
        raise RuntimeError(
            "INT8 requires torch 2.9.1, triton 3.5.1 and gemlite 0.6.0.post2"
        )
    from gemlite.helper import A16W8_INT8

    source_bytes = sum(
        layer.weight.numel() * layer.weight.element_size()
        for layer in selected.values()
    )
    result = model if inplace else copy.deepcopy(model)
    modules = dict(result.named_modules())
    packed_bytes = 0
    with torch.no_grad():
        for name in selected:
            source = modules[name]
            kernel = A16W8_INT8(
                device=str(source.weight.device), dtype=torch.bfloat16
            ).from_linear(source)
            if kernel.W_q.dtype != torch.int8 or kernel.scales.dtype != torch.float32:
                raise RuntimeError(
                    "GemLite did not produce INT8 weights and FP32 scales"
                )
            packed_bytes += sum(
                t.numel() * t.element_size()
                for t in (kernel.W_q, kernel.scales, kernel.zeros)
            )
            layer = GemliteInt8Linear(kernel)
            if name:
                result.set_submodule(name, layer)
            else:
                result = layer
    return result, {
        "backend": "gemlite",
        "execution": "int8_weight_only_fused_bf16",
        "weight_scaling": "per_output_channel",
        "group_size": None,
        "kernel": "gemlite::forward_functional",
        "gemlite_version": gem_version,
        "torch_version": torch.__version__,
        "config": asdict(config),
        "converted_layers": list(selected),
        "skipped_layers": skipped,
        "source_weight_bytes": source_bytes,
        "packed_weight_bytes": packed_bytes,
        "performance_validated": False,
    }
