# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:10
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Distributed environment detection and initialization helpers."""

from __future__ import annotations

import logging
import os
import socket
from datetime import timedelta
from typing import Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

DistributedEnv = Tuple[int, int, int, str, int]


def get_available_port() -> int:
    """Return an available local TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def get_master_address() -> str:
    """Return the configured or best-effort master address."""

    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr:
        return master_addr
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "localhost"


def detect_slurm_environment() -> Optional[DistributedEnv]:
    """Detect a SLURM launch environment."""

    if "SLURM_PROCID" not in os.environ:
        return None

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    master_addr = _resolve_slurm_master()
    master_port = int(os.environ.get("MASTER_PORT", _slurm_default_port()))

    logger.info(
        "Detected SLURM environment: rank=%s world_size=%s master=%s:%s",
        rank,
        world_size,
        master_addr,
        master_port,
    )
    return rank, world_size, local_rank, master_addr, master_port


def detect_torchrun_environment() -> Optional[DistributedEnv]:
    """Detect a torchrun or torch.distributed.launch environment."""

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))

    logger.info(
        "Detected torchrun environment: rank=%s world_size=%s master=%s:%s",
        rank,
        world_size,
        master_addr,
        master_port,
    )
    return rank, world_size, local_rank, master_addr, master_port


def detect_mpi_environment() -> Optional[DistributedEnv]:
    """Detect OpenMPI or MPICH launch metadata."""

    if "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    elif "PMI_RANK" in os.environ:
        rank = int(os.environ["PMI_RANK"])
        world_size = int(os.environ["PMI_SIZE"])
        local_rank = int(os.environ.get("PMI_LOCAL_RANK", 0))
    else:
        return None

    master_addr = get_master_address()
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    logger.info(
        "Detected MPI environment: rank=%s world_size=%s master=%s:%s",
        rank,
        world_size,
        master_addr,
        master_port,
    )
    return rank, world_size, local_rank, master_addr, master_port


def initialize_distributed(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Initialize torch.distributed from explicit args or launch metadata."""

    if dist.is_initialized():
        return (
            int(dist.get_rank()),
            int(dist.get_world_size()),
            int(os.environ.get("LOCAL_RANK", 0)),
        )

    env_info = _detect_environment()
    if env_info is not None:
        auto_rank, auto_world_size, auto_local_rank, auto_addr, auto_port = env_info
        rank = auto_rank if rank is None else rank
        world_size = auto_world_size if world_size is None else world_size
        local_rank = auto_local_rank if local_rank is None else local_rank
        master_addr = auto_addr if master_addr is None else master_addr
        master_port = auto_port if master_port is None else master_port
    else:
        rank = 0 if rank is None else rank
        world_size = 1 if world_size is None else world_size
        local_rank = 0 if local_rank is None else local_rank
        master_addr = "localhost" if master_addr is None else master_addr
        master_port = get_available_port() if master_port is None else master_port

    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    init_method = init_method or "env://"
    timeout_seconds = 1800 if timeout is None else int(timeout)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    logger.info(
        "Initializing distributed environment: backend=%s rank=%s world_size=%s local_rank=%s",
        backend,
        rank,
        world_size,
        local_rank,
    )
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=int(world_size),
            rank=int(rank),
            timeout=timedelta(seconds=timeout_seconds),
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))
    except Exception as exc:
        raise RuntimeError(f"distributed initialization failed: {exc}") from exc

    return int(rank), int(world_size), int(local_rank)


def cleanup_distributed() -> None:
    """Destroy the active process group if one exists."""

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed environment cleaned up")


def get_distributed_info() -> dict:
    """Return current distributed environment metadata."""

    info = {
        "initialized": dist.is_initialized(),
        "backend": dist.get_backend() if dist.is_initialized() else None,
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
        "master_port": int(os.environ.get("MASTER_PORT", 29500)),
        "environment": _environment_name(),
    }
    if "SLURM_JOB_ID" in os.environ:
        info["job_id"] = os.environ.get("SLURM_JOB_ID")
    return info


def print_distributed_info() -> None:
    """Log distributed metadata on rank 0 only."""

    from .utils import print_rank_0

    info = get_distributed_info()
    print_rank_0("=" * 60)
    print_rank_0("Distributed environment")
    print_rank_0("=" * 60)
    print_rank_0(f"environment: {info['environment']}")
    print_rank_0(f"initialized: {info['initialized']}")
    if info["initialized"]:
        print_rank_0(f"backend: {info['backend']}")
        print_rank_0(f"rank: {info['rank']} / {info['world_size']}")
        print_rank_0(f"local_rank: {info['local_rank']}")
        print_rank_0(f"master: {info['master_addr']}:{info['master_port']}")
    if "job_id" in info:
        print_rank_0(f"job_id: {info['job_id']}")
    print_rank_0("=" * 60)


def _detect_environment() -> Optional[DistributedEnv]:
    for detector in (
        detect_torchrun_environment,
        detect_slurm_environment,
        detect_mpi_environment,
    ):
        env = detector()
        if env is not None:
            return env
    return None


def _environment_name() -> str:
    if "SLURM_JOB_ID" in os.environ:
        return "SLURM"
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        return "OpenMPI"
    if "PMI_RANK" in os.environ:
        return "MPICH"
    return "torchrun/local"


def _resolve_slurm_master() -> str:
    nodelist = os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return get_master_address()
    master_node = nodelist.split(",")[0].split("[")[0]
    try:
        return socket.gethostbyname(master_node)
    except OSError:
        return master_node


def _slurm_default_port() -> int:
    return int(os.environ.get("SLURM_JOB_ID", get_available_port())) % 10000 + 29500
