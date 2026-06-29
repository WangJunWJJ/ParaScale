# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:48
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Shared helpers for built-in workload implementations."""

from __future__ import annotations

import warnings
from typing import Any, Dict


def _suppress_activation_checkpointing_future_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message="`torch.cpu.amp.autocast\\(args\\.\\.\\.\\)` is deprecated.*",
        category=FutureWarning,
        module="torch.utils.checkpoint",
    )


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise ImportError("Torch runtime factory workloads require PyTorch.") from exc
    return torch


def _require_pil_image():
    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError("DataComp WDS workloads require Pillow.") from exc
    return Image


def _select_torch_device(torch: Any, requested: str):
    requested = (requested or "auto").lower()
    if requested == "auto":
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0") or 0)
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    if requested == "cuda":
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0") or 0)
        return torch.device(f"cuda:{local_rank}")
    return torch.device(requested)


def _patchify(torch: Any, images: Any, patch_size: int):
    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("synthetic vision image_size must be divisible by patch_size")
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(batch, -1, channels * patch_size * patch_size)
