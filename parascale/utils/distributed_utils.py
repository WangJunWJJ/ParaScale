# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : distributed_utils.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 分布式训练工具模块

本模块提供多节点分布式训练的自动初始化支持，
兼容 torchrun、SLURM、MPI 等多种环境。
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple
import socket
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


def get_available_port() -> int:
    """
    获取一个可用的端口号
    
    Returns:
        可用的端口号
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def get_master_address() -> str:
    """
    获取主节点地址
    
    尝试从环境变量或网络接口获取主节点地址。
    
    Returns:
        主节点 IP 地址
    """
    # 尝试从环境变量获取
    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr:
        return master_addr
    
    # 尝试获取本机 IP
    try:
        # 获取主机名对应的 IP
        hostname = socket.gethostname()
        master_addr = socket.gethostbyname(hostname)
        return master_addr
    except Exception:
        pass
    
    # 默认使用 localhost
    return "localhost"


def detect_slurm_environment() -> Optional[Tuple[int, int, int, str, int]]:
    """
    检测 SLURM 集群环境
    
    Returns:
        如果检测到 SLURM 环境，返回 (rank, world_size, local_rank, master_addr, master_port)
        否则返回 None
    """
    if "SLURM_PROCID" not in os.environ:
        return None
    
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    
    # 获取主节点地址
    if "SLURM_NODELIST" in os.environ:
        # SLURM_NODELIST 格式如：node[01-04] 或 node01,node02
        nodelist = os.environ["SLURM_NODELIST"]
        # 简化处理，取第一个节点
        master_node = nodelist.split(',')[0].split('[')[0]
        try:
            master_addr = socket.gethostbyname(master_node)
        except:
            master_addr = master_node
    else:
        master_addr = get_master_address()
    
    # 获取或生成端口
    master_port = int(os.environ.get("SLURM_JOB_ID", get_available_port())) % 10000 + 29500
    
    logger.info(f"检测到 SLURM 环境: rank={rank}, world_size={world_size}, "
                f"master_addr={master_addr}, master_port={master_port}")
    
    return rank, world_size, local_rank, master_addr, master_port


def detect_torchrun_environment() -> Optional[Tuple[int, int, int, str, int]]:
    """
    检测 torchrun / torch.distributed.launch 环境
    
    Returns:
        如果检测到 torchrun 环境，返回 (rank, world_size, local_rank, master_addr, master_port)
        否则返回 None
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    
    logger.info(f"检测到 torchrun 环境: rank={rank}, world_size={world_size}, "
                f"master_addr={master_addr}, master_port={master_port}")
    
    return rank, world_size, local_rank, master_addr, master_port


def detect_mpi_environment() -> Optional[Tuple[int, int, int, str, int]]:
    """
    检测 MPI 环境
    
    Returns:
        如果检测到 MPI 环境，返回 (rank, world_size, local_rank, master_addr, master_port)
        否则返回 None
    """
    if "OMPI_COMM_WORLD_RANK" not in os.environ and "PMI_RANK" not in os.environ:
        return None
    
    # OpenMPI
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    # MPICH
    else:
        rank = int(os.environ["PMI_RANK"])
        world_size = int(os.environ["PMI_SIZE"])
        local_rank = int(os.environ.get("PMI_LOCAL_RANK", 0))
    
    master_addr = get_master_address()
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    
    logger.info(f"检测到 MPI 环境: rank={rank}, world_size={world_size}, "
                f"master_addr={master_addr}, master_port={master_port}")
    
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
        timeout: 初始化超时时间（秒）
    
    Returns:
        元组 (rank, world_size, local_rank)
    
    Raises:
        RuntimeError: 初始化失败时抛出
    
    Example:
        >>> # 自动检测环境并初始化
        >>> rank, world_size, local_rank = initialize_distributed()
        >>> 
        >>> # 手动指定参数
        >>> rank, world_size, local_rank = initialize_distributed(
        ...     backend="nccl",
        ...     master_addr="192.168.1.100",
        ...     master_port=29500
        ... )
    """
    # 如果已经初始化，直接返回当前信息
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    
    # 自动检测环境
    env_info = None
    
    # 优先级：torchrun > SLURM > MPI
    if env_info is None:
        env_info = detect_torchrun_environment()
    if env_info is None:
        env_info = detect_slurm_environment()
    if env_info is None:
        env_info = detect_mpi_environment()
    
    # 使用检测到的环境或手动指定的参数
    if env_info is not None:
        auto_rank, auto_world_size, auto_local_rank, auto_master_addr, auto_master_port = env_info
        rank = rank if rank is not None else auto_rank
        world_size = world_size if world_size is not None else auto_world_size
        local_rank = local_rank if local_rank is not None else auto_local_rank
        master_addr = master_addr if master_addr is not None else auto_master_addr
        master_port = master_port if master_port is not None else auto_master_port
    else:
        # 单机单卡/多卡模式
        rank = rank if rank is not None else 0
        world_size = world_size if world_size is not None else 1
        local_rank = local_rank if local_rank is not None else 0
        master_addr = master_addr if master_addr is not None else "localhost"
        master_port = master_port if master_port is not None else get_available_port()
    
    # 设置环境变量
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    
    # 自动选择后端
    if backend is None:
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
    
    # 设置初始化方法
    if init_method is None:
        init_method = "env://"
    
    # 设置超时
    if timeout is None:
        timeout = 1800  # 默认 30 分钟
    
    logger.info(f"初始化分布式环境: backend={backend}, rank={rank}, "
                f"world_size={world_size}, local_rank={local_rank}")
    
    try:
        # 初始化进程组
        if timeout is not None:
            timeout = timedelta(seconds=timeout)
        else:
            timeout = timedelta(seconds=1800)  # 默认 30 分钟
        
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout
        )
        
        # 设置当前设备
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logger.info(f"设置 CUDA 设备: cuda:{local_rank}")
        
        logger.info("分布式环境初始化成功")
        
    except Exception as e:
        raise RuntimeError(f"分布式初始化失败: {e}")
    
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
        logger.info("分布式环境已清理")


def get_distributed_info() -> dict:
    """
    获取分布式环境信息
    
    Returns:
        包含分布式环境信息的字典
    """
    info = {
        "initialized": dist.is_initialized(),
        "backend": dist.get_backend() if dist.is_initialized() else None,
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
        "master_port": int(os.environ.get("MASTER_PORT", 29500)),
    }
    
    # 检测环境类型
    if "SLURM_JOB_ID" in os.environ:
        info["environment"] = "SLURM"
        info["job_id"] = os.environ.get("SLURM_JOB_ID")
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        info["environment"] = "OpenMPI"
    elif "PMI_RANK" in os.environ:
        info["environment"] = "MPICH"
    else:
        info["environment"] = "torchrun/local"
    
    return info


def print_distributed_info() -> None:
    """
    打印分布式环境信息
    
    只在主进程（rank 0）上打印。
    """
    from .utils import print_rank_0
    
    info = get_distributed_info()
    
    print_rank_0("=" * 60)
    print_rank_0("分布式环境信息")
    print_rank_0("=" * 60)
    print_rank_0(f"环境类型: {info['environment']}")
    print_rank_0(f"初始化状态: {'已初始化' if info['initialized'] else '未初始化'}")
    if info['initialized']:
        print_rank_0(f"后端: {info['backend']}")
        print_rank_0(f"Rank: {info['rank']} / {info['world_size']}")
        print_rank_0(f"Local Rank: {info['local_rank']}")
        print_rank_0(f"主节点: {info['master_addr']}:{info['master_port']}")
    if 'job_id' in info:
        print_rank_0(f"作业 ID: {info['job_id']}")
    print_rank_0("=" * 60)
