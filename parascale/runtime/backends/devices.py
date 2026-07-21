# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午12:06
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Backend device helpers."""

from __future__ import annotations

import os
from typing import Any


def npu_is_available(torch: Any) -> bool:
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            if npu.is_available():
                return True
        except Exception:
            pass
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def current_accelerator(torch: Any) -> str:
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            if cuda.is_available():
                return "cuda"
        except Exception:
            pass
    if npu_is_available(torch):
        return "npu"
    return "cpu"


def resolve_torch_device(
    torch: Any,
    *,
    local_rank: int = 0,
    requested_device: str | None = None,
) -> Any:
    if requested_device:
        return torch.device(str(requested_device))
    accelerator = current_accelerator(torch)
    if accelerator == "cuda":
        return torch.device(f"cuda:{int(local_rank)}")
    if accelerator == "npu":
        device_id = resolve_ascend_device_id(local_rank, torch)
        return torch.device(f"npu:{device_id}")
    return torch.device("cpu")


def select_torch_device(torch: Any, requested: str | None = "auto") -> Any:
    requested_text = (requested or "auto").lower()
    requested_device = None if requested_text == "auto" else requested
    return resolve_torch_device(
        torch,
        local_rank=int(os.environ.get("LOCAL_RANK", "0") or 0),
        requested_device=requested_device,
    )


def set_current_device(
    torch: Any,
    *,
    local_rank: int = 0,
    requested_device: str | None = None,
) -> Any:
    device = resolve_torch_device(
        torch,
        local_rank=local_rank,
        requested_device=requested_device,
    )
    device_text = str(device)
    if device_text.startswith("cuda:"):
        torch.cuda.set_device(int(device_text.split(":", 1)[1]))
    elif device_text.startswith("npu:"):
        torch.npu.set_device(int(device_text.split(":", 1)[1]))
    return device


def resolve_ascend_device_id(local_rank: int, torch: Any | None = None) -> int:
    device_id = int(local_rank)
    if torch is None or not hasattr(torch, "npu"):
        return device_id
    try:
        device_count = int(torch.npu.device_count())
    except Exception:
        return device_id
    if device_id >= device_count:
        raise RuntimeError(
            "LOCAL_RANK exceeds torch_npu visible logical device count: "
            f"local_rank={local_rank}, device_count={device_count}. "
            "Check ASCEND_RT_VISIBLE_DEVICES and mounted /dev/davinci* devices; "
            "this runtime may require contiguous visible NPUs."
        )
    return device_id


def move_batch_to_device(batch: Any, device: str) -> Any:
    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device) for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    move = getattr(batch, "to", None)
    if callable(move):
        current_device = getattr(batch, "device", None)
        if current_device is not None and str(current_device) == str(device):
            return batch
        try:
            return move(device, non_blocking=True)
        except TypeError:
            return move(device)
    return batch


__all__ = [
    "current_accelerator",
    "move_batch_to_device",
    "npu_is_available",
    "resolve_ascend_device_id",
    "resolve_torch_device",
    "select_torch_device",
    "set_current_device",
]
