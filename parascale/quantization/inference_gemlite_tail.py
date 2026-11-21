# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午7:31
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Launch GemLite 0.5.1's INT8 GEMM with divisible, non-atomic tiles.

The upstream autotuner does not reject all unmasked K/N tails. Reuse its
computational kernel, bypassing only unsafe autotuning, with no global changes.
This optional module is loaded after the pinned backend has been checked.
"""

import torch
import triton
import triton.language as tl
from gemlite.triton_kernels.gemm_kernels import gemm_INT_kernel

_KERNEL = gemm_INT_kernel.fn


@torch.library.custom_op("parascale::gemlite_tail", mutates_args=())
def gemlite_tail(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Return BF16 linear output using unchanged INT8 codes and FP32 scales."""
    if (
        inputs.device.type != "cuda"
        or inputs.dtype != torch.bfloat16
        or weight.dtype != torch.int8
        or weight.ndim != 2
        or scales.dtype != torch.float32
        or scales.shape != (1, weight.shape[1])
        or weight.device != inputs.device
        or scales.device != inputs.device
        or zeros.device != inputs.device
        or zeros.numel() != 0
        or inputs.ndim < 1
        or inputs.shape[-1] != weight.shape[0]
        or weight.shape[0] % 32
        or weight.shape[1] % 8
        or min(weight.shape) <= 0
    ):
        raise ValueError("Invalid GemLite tail INT8/BF16 execution contract")
    if bias is not None and (
        bias.dtype != torch.bfloat16
        or bias.shape != (weight.shape[1],)
        or bias.device != inputs.device
    ):
        raise ValueError("Invalid GemLite tail bias")
    shape = (*inputs.shape[:-1], weight.shape[1])
    if not inputs.numel():
        return inputs.new_empty(shape)
    with torch.cuda.device(inputs.device):
        data = inputs.contiguous().view(-1, inputs.shape[-1])
        rows, features = data.shape
        outputs = weight.shape[1]
        result = inputs.new_empty((rows, outputs))
        block_m = 16 if rows <= 16 else 32
        block_n = min(64, outputs & -outputs)
        block_k = min(64, features & -features)
        grid = (triton.cdiv(rows, block_m) * triton.cdiv(outputs, block_n),)
        _KERNEL[grid](
            data,
            weight,
            result,
            scales,
            zeros,
            None,
            rows,
            outputs,
            features,
            rows,
            8,
            features,
            255,
            1,
            0,
            2,
            1,
            data.stride(0),
            data.stride(1),
            weight.stride(0),
            weight.stride(1),
            result.stride(0),
            result.stride(1),
            None,
            None,
            scales.stride(0),
            scales.stride(1),
            load_scales_as_block=False,
            input_dtype=tl.bfloat16,
            output_dtype=tl.bfloat16,
            acc_dtype=tl.float32,
            meta_dtype=tl.float32,
            channel_scale_mode=0,
            W_group_mode=2,
            zero_is_scalar=False,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=8,
            NUM_STAGES=2,
            A_load_order=0,
            data_contiguous=True,
            num_stages=2,
            num_warps=4,
        )
        if bias is not None:
            result.add_(bias)
        return result.view(shape)


@gemlite_tail.register_fake
def _fake_gemlite_tail(inputs, weight, scales, zeros, bias):
    return inputs.new_empty((*inputs.shape[:-1], weight.shape[1]))
