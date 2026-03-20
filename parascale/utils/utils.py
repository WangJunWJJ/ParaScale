# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : utils.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 工具函数模块

本模块提供了 ParaScale 框架的通用工具函数，
包括日志管理、分布式训练辅助函数和文件系统操作。
"""

import torch.distributed as dist
import os
import logging
from typing import Optional

# 获取日志记录器
logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    配置日志系统
    
    设置日志级别、输出格式和输出目标（控制台和/或文件）。
    
    Args:
        level: 日志级别，默认为 INFO。常用级别：
               - logging.DEBUG: 调试信息
               - logging.INFO: 一般信息
               - logging.WARNING: 警告信息
               - logging.ERROR: 错误信息
        log_file: 日志文件路径，如果为 None 则只输出到控制台
        format_string: 日志格式字符串，默认包含时间、模块名、级别和消息
    
    Example:
        >>> setup_logging(level=logging.DEBUG, log_file="train.log")
    """
    handlers = [logging.StreamHandler()]
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )


def print_rank_0(msg: str, rank: int = 0) -> None:
    """
    仅在指定 rank 上打印消息
    
    在分布式训练中，通常只需要在主进程（rank 0）上打印日志信息，
    避免多个进程同时打印导致输出混乱。
    
    Args:
        msg: 要打印的消息字符串
        rank: 要打印消息的 rank，默认为 0（主进程）
    
    Example:
        >>> print_rank_0("Training started")
        >>> print_rank_0(f"Loss: {loss.item():.4f}")
    """
    if dist.is_initialized():
        if dist.get_rank() == rank:
            logger.info(msg)
    else:
        logger.info(msg)


def get_rank() -> int:
    """
    获取当前进程的 rank
    
    在分布式训练中，每个进程都有一个唯一的 rank 标识。
    主进程的 rank 为 0。
    
    Returns:
        当前进程的 rank。如果分布式未初始化，返回 0。
    
    Example:
        >>> rank = get_rank()
        >>> print(f"Current rank: {rank}")
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    获取世界大小（进程总数）
    
    在分布式训练中，world_size 表示参与训练的进程总数。
    
    Returns:
        进程总数。如果分布式未初始化，返回 1。
    
    Example:
        >>> world_size = get_world_size()
        >>> print(f"Training with {world_size} processes")
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """
    获取本地 rank（当前节点内的 GPU 编号）
    
    在多节点训练中，每个节点有独立的 local_rank。
    
    Returns:
        本地 rank。如果未设置环境变量，返回 0。
    
    Example:
        >>> local_rank = get_local_rank()
        >>> torch.cuda.set_device(local_rank)
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """
    判断当前进程是否为主进程
    
    主进程通常负责打印日志、保存检查点等操作。
    
    Returns:
        如果当前进程是主进程（rank 0）返回 True，否则返回 False。
    
    Example:
        >>> if is_main_process():
        ...     save_checkpoint()
    """
    return get_rank() == 0


def ensure_directory(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    递归创建目录结构，类似于 `mkdir -p` 命令。
    
    Args:
        directory: 目录路径
    
    Example:
        >>> ensure_directory("./checkpoints/experiment1")
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def barrier() -> None:
    """
    同步所有进程
    
    阻塞当前进程，直到所有进程都到达此同步点。
    用于确保所有进程在某个时刻达到一致状态。
    
    Example:
        >>> # 确保所有进程都完成了数据加载
        >>> load_data()
        >>> barrier()
        >>> # 所有进程同时开始训练
        >>> train()
    """
    if dist.is_initialized():
        dist.barrier()


def get_node_rank() -> int:
    """
    获取当前节点的编号
    
    在多节点训练中，用于标识当前是第几个节点。
    
    Returns:
        节点编号。如果未设置环境变量，返回 0。
    
    Example:
        >>> node_rank = get_node_rank()
        >>> print(f"当前是第 {node_rank} 个节点")
    """
    # 尝试从环境变量获取
    if "NODE_RANK" in os.environ:
        return int(os.environ["NODE_RANK"])
    
    # 计算节点编号
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()
    
    # 假设每个节点 GPU 数量相同
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", world_size))
    
    return rank // gpus_per_node


def get_num_nodes() -> int:
    """
    获取节点总数
    
    Returns:
        节点总数。如果无法确定，返回 1。
    
    Example:
        >>> num_nodes = get_num_nodes()
        >>> print(f"共 {num_nodes} 个节点参与训练")
    """
    world_size = get_world_size()
    
    # 尝试从环境变量获取
    if "NUM_NODES" in os.environ:
        return int(os.environ["NUM_NODES"])
    
    # 尝试从 SLURM 获取
    if "SLURM_NNODES" in os.environ:
        return int(os.environ["SLURM_NNODES"])
    
    # 计算节点数
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", world_size))
    return (world_size + gpus_per_node - 1) // gpus_per_node
