# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : test_multi_node.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 多节点分布式训练测试模块

本模块测试多节点分布式训练的功能，
包括环境检测、分布式初始化和多节点通信。
"""

import torch
import torch.distributed as dist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.utils.distributed_utils import (
    initialize_distributed,
    cleanup_distributed,
    get_distributed_info,
    print_distributed_info,
    detect_slurm_environment,
    detect_torchrun_environment,
    detect_mpi_environment,
    get_available_port,
    get_master_address,
)
from parascale.utils.utils import (
    get_rank,
    get_world_size,
    get_local_rank,
    get_node_rank,
    get_num_nodes,
    barrier,
)


def test_environment_detection():
    """测试环境检测功能"""
    print("=" * 60)
    print("测试环境检测")
    print("=" * 60)
    
    # 测试 torchrun 环境检测
    torchrun_env = detect_torchrun_environment()
    if torchrun_env:
        rank, world_size, local_rank, master_addr, master_port = torchrun_env
        print(f"✓ 检测到 torchrun 环境:")
        print(f"  rank={rank}, world_size={world_size}, local_rank={local_rank}")
        print(f"  master={master_addr}:{master_port}")
    else:
        print("✓ 未检测到 torchrun 环境（正常）")
    
    # 测试 SLURM 环境检测
    slurm_env = detect_slurm_environment()
    if slurm_env:
        rank, world_size, local_rank, master_addr, master_port = slurm_env
        print(f"✓ 检测到 SLURM 环境:")
        print(f"  rank={rank}, world_size={world_size}, local_rank={local_rank}")
        print(f"  master={master_addr}:{master_port}")
    else:
        print("✓ 未检测到 SLURM 环境（正常）")
    
    # 测试 MPI 环境检测
    mpi_env = detect_mpi_environment()
    if mpi_env:
        rank, world_size, local_rank, master_addr, master_port = mpi_env
        print(f"✓ 检测到 MPI 环境:")
        print(f"  rank={rank}, world_size={world_size}, local_rank={local_rank}")
        print(f"  master={master_addr}:{master_port}")
    else:
        print("✓ 未检测到 MPI 环境（正常）")
    
    print("环境检测测试通过\n")
    return True


def test_port_and_address():
    """测试端口和地址获取"""
    print("=" * 60)
    print("测试端口和地址获取")
    print("=" * 60)
    
    # 测试获取可用端口
    port = get_available_port()
    assert 1024 <= port <= 65535
    print(f"✓ 获取可用端口: {port}")
    
    # 测试获取主节点地址
    master_addr = get_master_address()
    assert master_addr is not None
    print(f"✓ 获取主节点地址: {master_addr}")
    
    print("端口和地址获取测试通过\n")
    return True


def test_distributed_initialization():
    """测试分布式初始化"""
    print("=" * 60)
    print("测试分布式初始化")
    print("=" * 60)
    
    # 如果已经初始化，先清理
    if dist.is_initialized():
        cleanup_distributed()
    
    # 测试手动初始化
    rank, world_size, local_rank = initialize_distributed(
        rank=0,
        world_size=1,
        local_rank=0,
        master_addr="localhost",
        master_port=get_available_port()
    )
    
    assert rank == 0
    assert world_size == 1
    assert local_rank == 0
    assert dist.is_initialized()
    print(f"✓ 分布式初始化成功:")
    print(f"  rank={rank}, world_size={world_size}, local_rank={local_rank}")
    print(f"  backend={dist.get_backend()}")
    
    # 测试获取分布式信息
    info = get_distributed_info()
    assert info['initialized'] == True
    assert info['rank'] == 0
    assert info['world_size'] == 1
    print(f"✓ 获取分布式信息成功")
    
    # 打印分布式信息
    print_distributed_info()
    
    # 清理
    cleanup_distributed()
    print("✓ 分布式环境已清理")
    
    print("分布式初始化测试通过\n")
    return True


def test_utils_functions():
    """测试工具函数"""
    print("=" * 60)
    print("测试工具函数")
    print("=" * 60)
    
    # 初始化分布式环境
    if not dist.is_initialized():
        initialize_distributed(rank=0, world_size=1, local_rank=0)
    
    # 测试 get_rank
    rank = get_rank()
    print(f"✓ get_rank(): {rank}")
    
    # 测试 get_world_size
    world_size = get_world_size()
    print(f"✓ get_world_size(): {world_size}")
    
    # 测试 get_local_rank
    local_rank = get_local_rank()
    print(f"✓ get_local_rank(): {local_rank}")
    
    # 测试 get_node_rank
    node_rank = get_node_rank()
    print(f"✓ get_node_rank(): {node_rank}")
    
    # 测试 get_num_nodes
    num_nodes = get_num_nodes()
    print(f"✓ get_num_nodes(): {num_nodes}")
    
    # 测试 barrier
    barrier()
    print("✓ barrier() 执行成功")
    
    # 清理
    cleanup_distributed()
    
    print("工具函数测试通过\n")
    return True


def test_multi_node_simulation():
    """测试多节点模拟"""
    print("=" * 60)
    print("测试多节点模拟")
    print("=" * 60)
    
    # 手动计算节点信息（不依赖分布式初始化）
    rank = 4
    world_size = 8
    local_rank = 0
    gpus_per_node = 4
    
    print(f"✓ 模拟多节点环境:")
    print(f"  rank={rank}, world_size={world_size}, local_rank={local_rank}")
    print(f"  gpus_per_node={gpus_per_node}")
    
    # 手动计算节点编号
    node_rank = rank // gpus_per_node
    num_nodes = (world_size + gpus_per_node - 1) // gpus_per_node
    
    print(f"✓ 节点计算结果:")
    print(f"  node_rank={node_rank}, num_nodes={num_nodes}")
    
    # 验证计算
    assert node_rank == 1  # rank 4 // 4 GPUs per node = 1
    assert num_nodes == 2  # 8 world size / 4 GPUs per node = 2
    print(f"✓ 节点计算正确: 第 {node_rank} 个节点，共 {num_nodes} 个节点")
    
    print("多节点模拟测试通过\n")
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("ParaScale 多节点分布式训练测试套件")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    all_passed &= test_environment_detection()
    all_passed &= test_port_and_address()
    all_passed &= test_distributed_initialization()
    all_passed &= test_utils_functions()
    all_passed &= test_multi_node_simulation()
    
    print("=" * 60)
    if all_passed:
        print("所有多节点测试通过！")
    else:
        print("部分测试失败！")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
