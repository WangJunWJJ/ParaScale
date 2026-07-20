# -*- coding: utf-8 -*-
# @Time : 2026/7/12 下午12:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Runtime lifecycle helpers shared by CLI commands and orchestrators."""

from __future__ import annotations

import os
from typing import Any, Dict

from parascale.runtime.backends.devices import npu_is_available, set_current_device


def is_distributed_launch() -> bool:
    """Return whether the current process was started by a distributed launcher."""

    return int(os.environ.get("WORLD_SIZE", "1") or 1) > 1


def validate_distributed_topology(config_data: Dict[str, Any]) -> None:
    """Validate configured distributed topology against launcher environment."""

    distributed = _section(config_data, "distributed")
    if not distributed:
        return
    nnodes = max(1, int(distributed.get("nnodes", 1) or 1))
    nproc = max(1, int(distributed.get("nproc_per_node", 1) or 1))
    world_size = int(os.environ.get("WORLD_SIZE", str(nnodes * nproc)) or 1)
    expected = nnodes * nproc
    if world_size != expected:
        raise ValueError(
            "distributed topology mismatch: "
            f"WORLD_SIZE={world_size}, nnodes={nnodes}, "
            f"nproc_per_node={nproc}, expected={expected}"
        )


def initialize_distributed_for_backend(backend: str) -> None:
    """Initialize torch.distributed for a requested training backend."""

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        raise ImportError(
            f"CLI backend '{backend}' requires torch and torch.distributed."
        ) from exc

    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    set_current_device(torch, local_rank=local_rank)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if not dist.is_initialized():
        dist_backend = distributed_backend_for_torch(torch)
        dist.init_process_group(backend=dist_backend)


def destroy_distributed_runtime() -> None:
    """Destroy an initialized torch.distributed process group if one exists."""

    try:
        import torch.distributed as dist
    except Exception:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_backend_for_torch(torch: Any) -> str:
    """Choose the torch.distributed backend for the current accelerator runtime."""

    if torch.cuda.is_available():
        return "nccl"
    if npu_is_available(torch):
        return "hccl"
    return "gloo"


def distributed_rank() -> int:
    """Return the launcher-provided global rank, defaulting to rank zero."""

    return int(os.environ.get("RANK", "0") or 0)


def model_device(model: Any) -> str:
    """Return the first parameter device for a model-like object."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "unknown"
    try:
        return str(next(parameters()).device)
    except StopIteration:
        return "unknown"
    except Exception:
        return "unknown"


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


__all__ = [
    "destroy_distributed_runtime",
    "distributed_backend_for_torch",
    "distributed_rank",
    "initialize_distributed_for_backend",
    "is_distributed_launch",
    "model_device",
    "validate_distributed_topology",
]
