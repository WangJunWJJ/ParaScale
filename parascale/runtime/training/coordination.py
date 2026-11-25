# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Coordinate local setup/I/O failures before entering the next collective phase."""

from __future__ import annotations

import json
from typing import Any


def _tensor_identity(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "tensor_dtype": str(value.dtype),
            "tensor_shape": list(value.shape),
            "tensor_values": value.detach().cpu().tolist(),
        }
    raise TypeError(f"Unsupported coordinated identity value: {type(value).__name__}")


def coordinated_call(
    operation: Any, *, phase: str, identity: Any = None, fallback_barrier: Any = None
) -> Any:
    """Run on every rank, propagating errors and optionally matching JSON identity.

    All ranks must call this in the same order. The operation and identity callback
    must be local: a rank-local failure cannot protect collectives inside them.
    """
    result, error, key = None, None, None
    try:
        result = operation()
        if identity is not None:
            key = json.dumps(
                identity(result),
                sort_keys=True,
                allow_nan=False,
                default=_tensor_identity,
            )
    except Exception as exc:
        error = exc
    try:
        import torch.distributed as dist
    except ImportError:
        dist = None
    if dist is not None and dist.is_available() and dist.is_initialized():
        reports = [None] * dist.get_world_size()
        message = str(error) if error is not None else None
        dist.all_gather_object(reports, message if identity is None else (message, key))
        errors = reports if identity is None else [report[0] for report in reports]
        if error is not None:
            raise error
        if any(item is not None for item in errors):
            raise RuntimeError(f"{phase} phase failed on another rank: {errors}")
        if identity is not None and any(report[1] != key for report in reports):
            raise ValueError(f"{phase} identity mismatch across ranks")
    elif fallback_barrier is not None:
        fallback_barrier()
    if error is not None:
        raise error
    return result
