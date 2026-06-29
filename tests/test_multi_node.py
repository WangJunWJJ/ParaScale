# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import torch.distributed as dist

from parascale.utils.distributed_utils import (
    cleanup_distributed,
    detect_mpi_environment,
    detect_slurm_environment,
    detect_torchrun_environment,
    get_available_port,
    get_distributed_info,
    get_master_address,
    initialize_distributed,
)
from parascale.utils.utils import (
    barrier,
    get_local_rank,
    get_node_rank,
    get_num_nodes,
    get_rank,
    get_world_size,
)


def test_environment_detection_returns_optional_tuples():
    for detector in [
        detect_torchrun_environment,
        detect_slurm_environment,
        detect_mpi_environment,
    ]:
        env = detector()
        if env is not None:
            rank, world_size, local_rank, master_addr, master_port = env
            assert rank >= 0
            assert world_size >= 1
            assert local_rank >= 0
            assert master_addr
            assert int(master_port) > 0


def test_port_and_master_address_helpers():
    port = get_available_port()
    assert 1024 <= port <= 65535
    assert get_master_address()


def test_distributed_initialization_and_info():
    if dist.is_initialized():
        cleanup_distributed()

    rank, world_size, local_rank = initialize_distributed(
        rank=0,
        world_size=1,
        local_rank=0,
        master_addr="localhost",
        master_port=get_available_port(),
    )

    try:
        assert rank == 0
        assert world_size == 1
        assert local_rank == 0
        assert dist.is_initialized()

        info = get_distributed_info()
        assert info["initialized"] is True
        assert info["rank"] == 0
        assert info["world_size"] == 1
    finally:
        cleanup_distributed()


def test_distributed_utility_functions_on_single_process_group():
    if not dist.is_initialized():
        initialize_distributed(
            rank=0, world_size=1, local_rank=0, master_port=get_available_port()
        )

    try:
        assert get_rank() == 0
        assert get_world_size() == 1
        assert get_local_rank() == 0
        assert get_node_rank() == 0
        assert get_num_nodes() == 1
        barrier()
    finally:
        cleanup_distributed()


def test_multi_node_rank_math():
    rank = 4
    world_size = 8
    gpus_per_node = 4

    node_rank = rank // gpus_per_node
    num_nodes = (world_size + gpus_per_node - 1) // gpus_per_node

    assert node_rank == 1
    assert num_nodes == 2
