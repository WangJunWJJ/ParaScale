# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : __init__.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 工具函数模块

本模块提供了 ParaScale 框架的通用工具函数，
包括日志管理、分布式训练辅助函数和文件系统操作。
"""

from .utils import (
    setup_logging,
    print_rank_0,
    get_rank,
    get_world_size,
    get_local_rank,
    get_node_rank,
    get_num_nodes,
    is_main_process,
    ensure_directory,
    barrier,
)

from .distributed_utils import (
    initialize_distributed,
    cleanup_distributed,
    get_distributed_info,
    print_distributed_info,
    get_available_port,
    get_master_address,
)

from .hardware_monitor import (
    RealTimeHardwareMonitor,
    HardwareMetrics,
    create_hardware_monitor,
)

__all__ = [
    # 日志和打印
    "setup_logging",
    "print_rank_0",
    # 分布式信息
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "get_node_rank",
    "get_num_nodes",
    "is_main_process",
    # 其他工具
    "ensure_directory",
    "barrier",
    # 分布式工具
    "initialize_distributed",
    "cleanup_distributed",
    "get_distributed_info",
    "print_distributed_info",
    "get_available_port",
    "get_master_address",
    # 实时硬件监控
    "RealTimeHardwareMonitor",
    "HardwareMetrics",
    "create_hardware_monitor",
]
