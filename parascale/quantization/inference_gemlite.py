# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Optional channelwise INT8 weights with fused GemLite CUDA execution."""

import ast
import copy
from dataclasses import asdict
from importlib.metadata import version

import torch
from torch import nn


def _requires_safe_gemm(features, outputs, rows):
    """Keep only shapes whose unmasked upstream K/N tiles are all divisible."""
    if rows == 1:
        safe_n = outputs % 64 == 0 or outputs & (outputs - 1) == 0
        # Fast GEMV pruning can eliminate every candidate for small K divisors.
        area = min(features & -features, 256) * min(outputs, 16)
        return not (features % 64 == 0 and safe_n and area >= 1024)
    return not all(
        value % 512 == 0 or value & (value - 1) == 0 for value in (features, outputs)
    )


def _int8_metadata(features, contiguous):
    # Pinned GemLite 0.5.1 ABI: BF16=2, FP32=0; channelwise weight-only INT8.
    return [0, 8, features, 255, 1, 2, 2, 0, 0, 0, 2, contiguous]


class GemliteInt8Linear(nn.Module):
    """Inference-only Linear.forward adapter; functional weight access unsupported."""

    def __init__(self, kernel):
        super().__init__()
        metadata = kernel.get_meta_args()
        if (
            not isinstance(metadata, (list, tuple))
            or len(metadata) != 12
            or any(type(value) is not int for value in metadata)
            or metadata[-1] not in (0, 1)
            or list(metadata) != _int8_metadata(kernel.in_features, metadata[-1])
        ):
            raise ValueError("Unsupported GemLite INT8 execution metadata")
        self.in_features, self.out_features = kernel.in_features, kernel.out_features
        self._data_contiguous = metadata[-1]
        self._weight_stride = tuple(kernel.W_q.stride())
        for name in ("W_q", "scales", "zeros", "bias"):
            value = getattr(kernel, name)
            self.register_buffer(name, value.detach() if value is not None else None)
        self._safe_forward = None
        if any(
            _requires_safe_gemm(self.in_features, self.out_features, rows)
            for rows in (1, 2)
        ):
            from .inference_gemlite_tail import gemlite_tail

            self._safe_forward = gemlite_tail
        self.eval()

    @property
    def weight(self):
        raise RuntimeError(
            "INT8 requires Linear.forward; functional weight access is unsupported"
        )

    def forward(self, inputs):
        if inputs.dtype != torch.bfloat16 or self.scales.dtype != torch.float32:
            raise ValueError(
                "INT8 requires BF16 inputs and FP32 scales; set dtype before conversion"
            )
        if inputs.device != self.W_q.device:
            raise ValueError("INT8 inputs and weights must share a device")
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("INT8 input feature dimension mismatch")
        if (
            self.W_q.dtype != torch.int8
            or self.W_q.shape != (self.in_features, self.out_features)
            or self.W_q.stride() != self._weight_stride
            or self.scales.shape != (1, self.out_features)
            or self.scales.device != inputs.device
            or self.zeros.numel() != 0
            or self.zeros.device != inputs.device
            or (
                self.bias is not None
                and (
                    self.bias.shape != (self.out_features,)
                    or self.bias.dtype != torch.bfloat16
                    or self.bias.device != inputs.device
                )
            )
        ):
            raise ValueError("Invalid INT8 packed tensor state")
        if inputs.numel() == 0:
            return inputs.new_empty((*inputs.shape[:-1], self.out_features))
        if _requires_safe_gemm(
            self.in_features, self.out_features, inputs.numel() // self.in_features
        ):
            return self._safe_forward(
                inputs,
                self.W_q,
                self.scales,
                self.zeros,
                self.bias,
            )
        return torch.ops.gemlite.forward_functional.default(
            inputs,
            self.bias,
            [self.W_q, self.scales, self.zeros],
            _int8_metadata(self.in_features, self._data_contiguous),
        )


def _load_gemlite():
    gem_version = version("gemlite")
    if (
        torch.__version__.split("+")[0] != "2.9.1"
        or gem_version != "0.5.1"
        or version("triton") != "3.5.1"
    ):
        raise RuntimeError("INT8 requires torch 2.9.1, triton 3.5.1 and gemlite 0.5.1")
    from gemlite.core import GEMLITE_TRITON_CONFIG_CACHE
    from gemlite.helper import A16W8_INT8

    _validate_kernel_cache(GEMLITE_TRITON_CONFIG_CACHE)

    return A16W8_INT8, gem_version


def _validate_kernel_cache(cache):
    """Check bundled/external INT8 cache entries that bypass upstream pruning."""
    for mode in ("GEMM", "GEMM_SPLITK", "GEMV_SPLITK"):
        for signature, config in cache.get(mode, {}).items():
            try:
                rows, outputs, features, group, elements, dtype_id = ast.literal_eval(
                    signature
                )
                if elements != 1 or dtype_id != 108:
                    continue
                block_k, block_n = config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"]
                block_m = config["BLOCK_SIZE_M"]
                split = config.get("SPLIT_K", 1)
                values = (
                    rows,
                    outputs,
                    features,
                    group,
                    block_k,
                    block_n,
                    block_m,
                    split,
                    config["GROUP_SIZE_M"],
                    config["num_warps"],
                    config["num_stages"],
                )
                if (
                    any(type(value) is not int or value <= 0 for value in values)
                    or any(
                        value & (value - 1)
                        for value in (block_k, block_n, block_m, split)
                    )
                    or (mode != "GEMV_SPLITK" and block_k < 16)
                    or features % (block_k * split)
                    or outputs % block_n
                    or (mode == "GEMV_SPLITK" and config["BLOCK_SIZE_M"] != 1)
                ):
                    raise ValueError("non-divisible launch")
            except (KeyError, TypeError, ValueError, SyntaxError):
                raise RuntimeError("Unsafe external GemLite kernel cache") from None


def convert_int8(model, config, selected, skipped, *, inplace):
    processor, gem_version = _load_gemlite()
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
            kernel = processor(
                device=str(source.weight.device), dtype=torch.bfloat16, fp32_scale=True
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
