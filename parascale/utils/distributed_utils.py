# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : distributed_utils.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 分布式训练工具模块

本模块提供多节点分布式训练的自动初始化支持，
兼容 torchrun、SLURM、MPI 等多种环境。

特性:
    - 自动环境检测：支持 torchrun/SLURM/MPI
    - 完整的参数验证：防止静默错误
    - 超时机制：防止通信永久阻塞
    - 详细的日志：便于调试和问题排查
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any
import socket
import logging
from datetime import timedelta
import warnings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 1800
DEFAULT_PORT_RANGE = (29500, 29600)


class DistributedValidationError(Exception):
    """分布式环境验证失败异常"""
    pass


class MasterAddressError(Exception):
    """主节点地址配置错误异常"""
    pass


def get_available_port() -> int:
    """
    获取一个可用的端口号

    Returns:
        可用的端口号

    Raises:
        socket.error: 当无法获取可用端口时抛出
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port
    except socket.error as e:
        raise RuntimeError(f"Failed to get available port: {e}")


def _validate_ip_address(ip: str) -> bool:
    """
    验证 IP 地址格式

    Args:
        ip: IP 地址字符串

    Returns:
        是否为有效 IP 地址
    """
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False


def _validate_port(port: int) -> bool:
    """
    验证端口号

    Args:
        port: 端口号

    Returns:
        是否为有效端口号
    """
    return 1 <= port <= 65535


def get_master_address() -> str:
    """
    获取主节点地址

    优先从环境变量获取，否则尝试自动检测。

    Returns:
        主节点 IP 地址

    Raises:
        MasterAddressError: 当无法获取有效地址时抛出
    """
    master_addr = os.environ.get("MASTER_ADDR")

    if master_addr:
        if not _validate_ip_address(master_addr):
            warnings.warn(
                f"Invalid MASTER_ADDR format: {master_addr}, treating as hostname"
            )
        return master_addr

    try:
        hostname = socket.gethostname()
        master_addr = socket.gethostbyname(hostname)
        if master_addr == "127.0.0.1":
            warnings.warn(
                "Master address resolved to localhost. "
                "This may not work in multi-node training. "
                "Please set MASTER_ADDR explicitly."
            )
        return master_addr
    except socket.gaierror as e:
        raise MasterAddressError(
            f"Failed to resolve master address: {e}. "
            "Please set MASTER_ADDR environment variable."
        )


def detect_slurm_environment() -> Optional[Tuple[int, int, int, str, int]]:
    """
    检测 SLURM 集群环境

    Returns:
        如果检测到 SLURM 环境，返回 (rank, world_size, local_rank, master_addr, master_port)
        否则返回 None

    Raises:
        DistributedValidationError: 当 SLURM 环境变量格式不正确时抛出
    """
    if "SLURM_PROCID" not in os.environ:
        return None

    try:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

        if rank < 0:
            raise DistributedValidationError(
                f"Invalid SLURM_PROCID: {rank}"
            )
        if world_size <= 0:
            raise DistributedValidationError(
                f"Invalid SLURM_NTASKS: {world_size}"
            )
        if local_rank < 0:
            raise DistributedValidationError(
                f"Invalid SLURM_LOCALID: {local_rank}"
            )
    except ValueError as e:
        raise DistributedValidationError(
            f"Invalid SLURM environment variable format: {e}"
        )

    if "SLURM_NODELIST" in os.environ:
        nodelist = os.environ["SLURM_NODELIST"]
        master_node = nodelist.split(',')[0].split('[')[0]
        try:
            master_addr = socket.gethostbyname(master_node)
        except socket.gaierror:
            logger.warning(
                f"Failed to resolve SLURM_NODELIST {master_node}, using as-is"
            )
            master_addr = master_node
    else:
        master_addr = get_master_address()

    slurm_port = os.environ.get("SLURM_JOB_ID", "")
    if slurm_port:
        try:
            master_port = (int(slurm_port) % 10000) + DEFAULT_PORT_RANGE[0]
        except ValueError:
            master_port = get_available_port()
    else:
        master_port = get_available_port()

    if not _validate_port(master_port):
        warnings.warn(
            f"Invalid port {master_port}, using default range"
        )
        master_port = DEFAULT_PORT_RANGE[0]

    logger.info(
        f"Detected SLURM environment: rank={rank}, world_size={world_size}, "
        f"local_rank={local_rank}, master_addr={master_addr}, master_port={master_port}"
    )

    return rank, world_size, local_rank, master_addr, master_port


def detect_torchrun_environment() -> Optional[Tuple[int, int, int, str, int]]:
    """
    检测 torchrun / torch.distributed.launch 环境

    Returns:
        如果检测到 torchrun 环境，返回 (rank, world_size, local_rank, master_addr, master_port)
        否则返回 None

    Raises:
        DistributedValidationError: 当环境变量格式不正确时抛出
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None

    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if rank < 0:
            raise DistributedValidationError(f"Invalid RANK: {rank}")
        if world_size <= 0:
            raise DistributedValidationError(f"Invalid WORLD_SIZE: {world_size}")
        if local_rank < 0:
            raise DistributedValidationError(f"Invalid LOCAL_RANK: {local_rank}")
        if rank >= world_size:
            raise DistributedValidationError(
                f"RANK {rank} >= WORLD_SIZE {world_size}"
            )
    except ValueError as e:
        raise DistributedValidationError(
            f"Invalid torchrun environment variable format: {e}"
        )

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port_str = os.environ.get("MASTER_PORT", "")

    try:
        master_port = int(master_port_str) if master_port_str else get_available_port()
    except ValueError:
        warnings.warn(
            f"Invalid MASTER_PORT: {master_port_str}, using available port"
        )
        master_port = get_available_port()

    if not _validate_port(master_port):
        warnings.warn(f"Invalid port {master_port}, using available port")
        master_port = get_available_port()

    logger.info(
        f"Detected torchrun environment: rank={rank}, world_size={world_size}, "
        f"local_rank={local_rank}, master_addr={master_addr}, master_port={master_port}"
    )

    return rank, world_size, local_rank, master_addr, master_port


def detect_mpi_environment() -> Optional[Tuple[int, int, int, str, int]]:
    """
    检测 MPI 环境

    Returns:
        如果检测到 MPI 环境，返回 (rank, world_size, local_rank, master_addr, master_port)
        否则返回 None

    Raises:
        DistributedValidationError: 当环境变量格式不正确时抛出
    """
    if "OMPI_COMM_WORLD_RANK" not in os.environ and "PMI_RANK" not in os.environ:
        return None

    try:
        if "OMPI_COMM_WORLD_RANK" in os.environ:
            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        else:
            rank = int(os.environ["PMI_RANK"])
            world_size = int(os.environ["PMI_SIZE"])
            local_rank = int(os.environ.get("PMI_LOCAL_RANK", 0))

        if rank < 0 or world_size <= 0 or local_rank < 0:
            raise DistributedValidationError(
                f"Invalid MPI environment: rank={rank}, world_size={world_size}, local_rank={local_rank}"
            )
    except ValueError as e:
        raise DistributedValidationError(
            f"Invalid MPI environment variable format: {e}"
        )

    master_addr = get_master_address()
    master_port_str = os.environ.get("MASTER_PORT", "")

    try:
        master_port = int(master_port_str) if master_port_str else get_available_port()
    except ValueError:
        master_port = get_available_port()

    logger.info(
        f"Detected MPI environment: rank={rank}, world_size={world_size}, "
        f"local_rank={local_rank}, master_addr={master_addr}, master_port={master_port}"
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
    timeout: Optional[int] = None
) -> Tuple[int, int, int]:
    """
    自动初始化分布式训练环境

    自动检测并初始化多种分布式环境（torchrun、SLURM、MPI），
    支持单节点多 GPU 和多节点训练。

    Args:
        backend: 通信后端，默认为 "nccl"（GPU）或 "gloo"（CPU）
        init_method: 初始化方法 URL，默认为 env://
        rank: 当前进程 rank，自动检测时可为 None
        world_size: 进程总数，自动检测时可为 None
        local_rank: 本地 rank，自动检测时可为 None
        master_addr: 主节点地址，自动检测时可为 None
        master_port: 主节点端口，自动检测时可为 None
        timeout: 初始化超时时间（秒），默认 1800 秒

    Returns:
        元组 (rank, world_size, local_rank)

    Raises:
        DistributedValidationError: 当参数验证失败时抛出
        RuntimeError: 初始化失败时抛出

    Example:
        >>> # 自动检测环境并初始化
        >>> rank, world_size, local_rank = initialize_distributed()

        >>> # 手动指定参数
        >>> rank, world_size, local_rank = initialize_distributed(
        ...     backend="nccl",
        ...     master_addr="192.168.1.100",
        ...     master_port=29500
        ... )
    """
    if dist.is_initialized():
        logger.warning("Distributed environment already initialized")
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))

    env_info = None
    detected_env = "unknown"

    if env_info is None:
        env_info = detect_torchrun_environment()
        if env_info is not None:
            detected_env = "torchrun"

    if env_info is None:
        env_info = detect_slurm_environment()
        if env_info is not None:
            detected_env = "SLURM"

    if env_info is None:
        env_info = detect_mpi_environment()
        if env_info is not None:
            detected_env = "MPI"

    if env_info is not None:
        auto_rank, auto_world_size, auto_local_rank, auto_master_addr, auto_master_port = env_info
        rank = rank if rank is not None else auto_rank
        world_size = world_size if world_size is not None else auto_world_size
        local_rank = local_rank if local_rank is not None else auto_local_rank
        master_addr = master_addr if master_addr is not None else auto_master_addr
        master_port = master_port if master_port is not None else auto_master_port
        logger.info(f"Using auto-detected {detected_env} environment")
    else:
        rank = rank if rank is not None else 0
        world_size = world_size if world_size is not None else 1
        local_rank = local_rank if local_rank is not None else 0
        master_addr = master_addr if master_addr is not None else "localhost"
        master_port = master_port if master_port is not None else get_available_port()
        logger.info("No distributed environment detected, using local mode")

    if rank < 0:
        raise DistributedValidationError(f"Rank must be non-negative, got {rank}")
    if world_size <= 0:
        raise DistributedValidationError(f"World size must be positive, got {world_size}")
    if local_rank < 0:
        raise DistributedValidationError(f"Local rank must be non-negative, got {local_rank}")
    if rank >= world_size:
        raise DistributedValidationError(
            f"Rank {rank} must be less than world_size {world_size}"
        )

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if backend not in ["nccl", "gloo", "mpi"]:
        raise DistributedValidationError(
            f"Unsupported backend: {backend}. Must be one of: nccl, gloo, mpi"
        )

    if init_method is None:
        init_method = "env://"

    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    logger.info(
        f"Initializing distributed environment: "
        f"backend={backend}, rank={rank}, world_size={world_size}, "
        f"local_rank={local_rank}, master_addr={master_addr}, master_port={master_port}"
    )

    try:
        timeout_delta = timedelta(seconds=timeout)

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout_delta
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logger.info(f"Set CUDA device: cuda:{local_rank}")

        logger.info("Distributed environment initialized successfully")

    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize distributed environment: {e}"
        ) from e

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """
    清理分布式环境

    销毁进程组，释放资源。

    Example:
        >>> initialize_distributed()
        >>> # 训练代码...
        >>> cleanup_distributed()
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed environment cleaned up")
    else:
        logger.warning("No distributed environment to clean up")


def get_distributed_info() -> Dict[str, Any]:
    """
    获取分布式环境信息

    Returns:
        包含分布式环境信息的字典

    Raises:
        RuntimeError: 当分布式环境未初始化时抛出
    """
    info = {
        "initialized": dist.is_initialized(),
    }

    if info["initialized"]:
        info.update({
            "backend": dist.get_backend(),
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
        })
    else:
        info.update({
            "backend": None,
            "rank": 0,
            "world_size": 1,
        })

    info.update({
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
        "master_port": int(os.environ.get("MASTER_PORT", 0)),
    })

    if "SLURM_JOB_ID" in os.environ:
        info["environment"] = "SLURM"
        info["job_id"] = os.environ.get("SLURM_JOB_ID")
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        info["environment"] = "OpenMPI"
    elif "PMI_RANK" in os.environ:
        info["environment"] = "MPICH"
    elif "RANK" in os.environ:
        info["environment"] = "torchrun"
    else:
        info["environment"] = "local"

    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_device_count"] = torch.cuda.device_count()
    else:
        info["cuda_available"] = False
        info["cuda_device_count"] = 0

    return info


def print_distributed_info() -> None:
    """
    打印分布式环境信息

    只在主进程（rank 0）上打印。
    """
    from .utils import print_rank_0

    info = get_distributed_info()

    print_rank_0("=" * 60)
    print_rank_0("Distributed Environment Information")
    print_rank_0("=" * 60)
    print_rank_0(f"Environment: {info['environment']}")
    print_rank_0(f"Initialized: {info['initialized']}")
    print_rank_0(f"Backend: {info['backend']}")
    print_rank_0(f"Rank: {info['rank']} / {info['world_size']}")
    print_rank_0(f"Local Rank: {info['local_rank']}")
    print_rank_0(f"Master: {info['master_addr']}:{info['master_port']}")
    if info.get("cuda_available"):
        print_rank_0(f"CUDA Devices: {info['cuda_device_count']}")
    if "job_id" in info:
        print_rank_0(f"Job ID: {info['job_id']}")
    print_rank_0("=" * 60)