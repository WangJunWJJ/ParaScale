# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Normalization of mean losses over samples or explicitly declared units."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def normalization_weight(loss_fn: Any, batch: Any) -> float:
    """Return the denominator of a mean loss, before forward execution.

    Loss callables may expose normalization_weight(batch), for example the
    number of valid shifted labels. Otherwise the loss is a per-sample mean.
    Invalid or empty denominators raise ValueError instead of producing NaNs.
    """
    custom = getattr(loss_fn, "normalization_weight", None)
    if callable(custom):
        value = custom(batch)
    elif isinstance(batch, Mapping):
        value = next(
            (
                batch[key]
                for key in ("num_pairs", "num_images", "batch_size", "num_samples")
                if batch.get(key) is not None
            ),
            None,
        )
        if value is None:
            tensors = [
                batch.get(key) for key in ("labels", "input_ids", "pixel_values", "x")
            ]
            value = next(
                (item.shape[0] for item in tensors if getattr(item, "ndim", 0) > 0), 1
            )
    elif getattr(batch, "ndim", 0) > 0:
        value = batch.shape[0]
    else:
        value = len(batch) if isinstance(batch, (list, tuple)) else 1
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("loss normalization weight must be finite and positive")
    return value


def global_normalization(
    local_weight: float, backend: Any, model: Any
) -> tuple[Any, int, Any]:
    """Return denominator, replica count and an optional pending collective.

    Wait on the collective after forward and before using the denominator, so
    communication can overlap independent computation without a CPU readback.
    """
    if getattr(backend, "name", "") not in {"native_ddp", "fsdp", "deepspeed"}:
        return local_weight, 1, None
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return local_weight, 1, None
    parameter = next(model.parameters())
    weight = torch.tensor(local_weight, device=parameter.device, dtype=torch.float32)
    work = dist.all_reduce(weight, op=dist.ReduceOp.SUM, async_op=True)
    # Keep the collective result on-device; a host scalar would serialize the
    # CPU training loop with GPU communication on every optimizer update.
    return weight, dist.get_world_size(), work
