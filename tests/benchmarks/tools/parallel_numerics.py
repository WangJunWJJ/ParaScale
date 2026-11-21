# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Independent FP32 state/update gates, not a performance or quality metric."""

import math

import torch

GATES = {
    "loss": (1e-5, 1e-6),
    "output": (1e-5, 3e-6),
    "gradient": (5e-4, 1e-7),
    "parameter": (1e-5, 1e-7),
    "update": (2e-2, 5e-8),
    "exp_avg": (5e-4, 1e-9),
    "exp_avg_sq": (1e-3, 1e-12),
    "step": (0.0, 0.0),
}


def compare_training_tensor(actual, expected, kind, *, logits_atol=3e-6):
    """Require both peak and RMS bounds; tiny Adam moments get their own floor."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("tensor shape/dtype mismatch")
    delta = (actual.detach() - expected.detach()).float()
    if not torch.isfinite(delta).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"nonfinite {kind}")
    rtol, atol = GATES[kind]
    if kind == "output":
        atol = logits_atol
    maximum = delta.abs().max().item()
    maximum_index = int(delta.abs().argmax().item())
    rms = delta.norm().item() / math.sqrt(delta.numel())
    expected_rms = expected.float().norm().item() / math.sqrt(expected.numel())
    peak_limit = atol + rtol * expected.abs().max().item()
    rms_limit = atol + rtol * expected_rms
    if kind == "output":
        peak_limit = logits_atol
        rms_limit = 1e-7 + rtol * expected_rms
    return {
        "passed": maximum <= peak_limit and rms <= rms_limit,
        "max_abs": maximum,
        "rms": rms,
        "expected_rms": expected_rms,
        "relative_l2": rms / max(expected_rms, 1e-30),
        "peak_limit": peak_limit,
        "rms_limit": rms_limit,
        "max_flat_index": maximum_index,
        "actual_at_max": actual.detach().reshape(-1)[maximum_index].item(),
        "expected_at_max": expected.detach().reshape(-1)[maximum_index].item(),
    }
