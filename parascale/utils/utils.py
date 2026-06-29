# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:11
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""General utility helpers for logging, ranks, and filesystem setup."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure process-wide logging."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=format_string, handlers=handlers)


def print_rank_0(msg: str, rank: int = 0) -> None:
    """Log a message only on the selected rank."""

    if dist.is_initialized() and dist.get_rank() != rank:
        return
    logger.info(msg)


def get_rank() -> int:
    """Return the current distributed rank, or 0 outside distributed mode."""

    return int(dist.get_rank()) if dist.is_initialized() else 0


def get_world_size() -> int:
    """Return the world size, or 1 outside distributed mode."""

    return int(dist.get_world_size()) if dist.is_initialized() else 1


def get_local_rank() -> int:
    """Return the local rank from the launch environment."""

    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Return True when the current process is rank 0."""

    return get_rank() == 0


def ensure_directory(directory: str) -> None:
    """Create a directory and its parents when missing."""

    os.makedirs(directory, exist_ok=True)


def barrier() -> None:
    """Synchronize all distributed ranks when a process group exists."""

    if dist.is_initialized():
        dist.barrier()


def get_node_rank() -> int:
    """Return the inferred node rank."""

    if "NODE_RANK" in os.environ:
        return int(os.environ["NODE_RANK"])
    world_size = get_world_size()
    rank = get_rank()
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", world_size))
    return rank // max(1, gpus_per_node)


def get_num_nodes() -> int:
    """Return the inferred number of nodes."""

    if "NUM_NODES" in os.environ:
        return int(os.environ["NUM_NODES"])
    if "SLURM_NNODES" in os.environ:
        return int(os.environ["SLURM_NNODES"])
    world_size = get_world_size()
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", world_size))
    return (world_size + max(1, gpus_per_node) - 1) // max(1, gpus_per_node)
