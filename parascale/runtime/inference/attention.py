# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午9:40
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Scoped attention selection for isolated inference workers."""

from contextlib import contextmanager


def _state(cuda) -> dict:
    return {
        "math": cuda.math_sdp_enabled(),
        "flash": cuda.flash_sdp_enabled(),
        "efficient": cuda.mem_efficient_sdp_enabled(),
        "cudnn": cuda.cudnn_sdp_enabled(),
        "math_reduced_precision": cuda.fp16_bf16_reduction_math_sdp_allowed(),
    }


@contextmanager
def attention_policy(policy: str):
    """Yield requested/enabled SDPA settings; restore them even if a worker fails.

    auto leaves backend selection untouched. math enables only math SDPA and
    disables its optional FP16/BF16 reduction. Use in an isolated worker, not
    concurrent serving threads. Unknown policies raise ValueError. Enabled flags
    describe allowed dispatch, not proof of an observed kernel or model quality.
    """
    if policy not in ("auto", "math"):
        raise ValueError("Unsupported attention policy")
    from torch.backends import cuda
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if policy == "auto":
        yield {"requested": policy, "enabled": _state(cuda)}
        return
    previous = cuda.fp16_bf16_reduction_math_sdp_allowed()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            cuda.allow_fp16_bf16_reduction_math_sdp(False)
            yield {"requested": policy, "enabled": _state(cuda)}
    finally:
        cuda.allow_fp16_bf16_reduction_math_sdp(previous)
