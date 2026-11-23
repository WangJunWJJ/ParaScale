# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit op-level precision with FP32 stage parameters and optimizer state."""

from typing import Any


def validate_precision(precision: str, device: Any) -> None:
    """Reject unsupported policies/devices before any model or mesh mutation."""
    if precision not in ("fp32", "bf16"):
        raise ValueError("Stage precision must be fp32 or bf16")
    if precision == "bf16":
        import torch

        if device.type != "cuda":
            raise ValueError("Stage bf16 precision requires native CUDA BF16 support")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError(
                    "Stage bf16 precision requires native CUDA BF16 support"
                )


def autocast_context(precision: str, device: Any) -> Any:
    """Override ambient autocast without retaining weight casts across steps."""
    import torch

    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
        cache_enabled=False,
    )
