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

    return (execution_world_size() or 1) > 1


def initialized_distributed() -> Any | None:
    """Return the active process group API without requiring Torch for planning."""
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    return dist if dist.is_available() and dist.is_initialized() else None


def execution_world_size() -> int | None:
    """Resolve allocation from an active group or launcher, rejecting disagreement."""
    declared = os.environ.get("WORLD_SIZE")
    expected = int(declared) if declared is not None else None
    if expected is not None and expected < 1:
        raise ValueError("WORLD_SIZE must be positive")
    dist = initialized_distributed()
    if dist is None:
        return expected
    actual = int(dist.get_world_size())
    if expected is not None and expected != actual:
        raise ValueError("WORLD_SIZE conflicts with the initialized process group")
    return actual


def validate_launcher_context() -> int | None:
    """Require complete, consistent launcher metadata for product data sharding."""
    world_size = execution_world_size()
    if (world_size or 1) <= 1:
        return world_size
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if any(name not in os.environ for name in required):
        raise ValueError("Distributed product workloads require complete launcher metadata")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    dist = initialized_distributed()
    if not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid rank in launcher metadata")
    if dist is not None and rank != int(dist.get_rank()):
        raise ValueError("launcher RANK conflicts with the initialized process group")
    return world_size


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

    dist = initialized_distributed()
    return int(dist.get_rank()) if dist is not None else int(os.environ.get("RANK", "0") or 0)


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
    "execution_world_size",
    "initialized_distributed",
    "initialize_distributed_for_backend",
    "is_distributed_launch",
    "model_device",
    "validate_distributed_topology",
    "validate_launcher_context",
]
