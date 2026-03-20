# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : test_refactoring.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.config import ParaScaleConfig
from parascale.parallel.base import BaseParallel
from parascale.parallel.data_parallel import DataParallel
from parascale.parallel.tensor_parallel import TensorParallel
from parascale.parallel.pipeline_parallel import PipelineParallel
from parascale.utils.utils import setup_logging, print_rank_0

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3072, 10)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class PipelineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ])
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

def test_config():
    print("=" * 50)
    print("Testing ParaScaleConfig...")
    print("=" * 50)
    
    config = ParaScaleConfig()
    print(f"Default config created successfully")
    
    config.update({'data_parallel_size': 2, 'tensor_parallel_mode': 'column'})
    assert config.data_parallel_size == 2
    assert config.tensor_parallel_mode == 'column'
    print(f"Config update works correctly")
    
    try:
        bad_config = ParaScaleConfig(batch_size=0)
        print("ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"Validation works: {e}")
    
    d = config.to_dict()
    config2 = ParaScaleConfig.from_dict(d)
    assert config2.data_parallel_size == config.data_parallel_size
    print("to_dict/from_dict works correctly")
    
    print("ParaScaleConfig: PASSED\n")
    return True

def test_base_parallel():
    print("=" * 50)
    print("Testing BaseParallel...")
    print("=" * 50)
    
    model = SimpleModel()
    
    try:
        parallel = DataParallel(model, rank=-1, world_size=2)
        print("ERROR: Should have raised ValueError for negative rank")
        return False
    except ValueError as e:
        print(f"Validation works: {e}")
    
    try:
        parallel = DataParallel(model, rank=0, world_size=0)
        print("ERROR: Should have raised ValueError for zero world_size")
        return False
    except ValueError as e:
        print(f"Validation works: {e}")
    
    print("BaseParallel: PASSED\n")
    return True

def test_distributed():
    print("=" * 50)
    print("Testing Distributed Parallel...")
    print("=" * 50)
    
    if not dist.is_initialized():
        print("Distributed not initialized, skipping distributed tests")
        return True
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    model = SimpleModel()
    
    print(f"Testing DataParallel on rank {rank}...")
    dp = DataParallel(model, rank, world_size)
    x = torch.randn(32, 3, 32, 32)
    output = dp.forward(x)
    print(f"DataParallel output shape: {output.shape}")
    assert output.shape == (32, 10)
    
    print(f"Testing TensorParallel on rank {rank}...")
    model2 = SimpleModel()
    tp = TensorParallel(model2, rank, world_size, mode="row")
    output = tp.forward(x)
    print(f"TensorParallel output shape: {output.shape}")
    
    print(f"Testing PipelineParallel on rank {rank}...")
    model3 = PipelineModel()
    pp = PipelineParallel(model3, rank, world_size, chunks=1)
    output = pp.forward(x)
    if rank == world_size - 1:
        print(f"PipelineParallel output shape: {output.shape}")
        assert output.shape[0] == 32
    
    print("Distributed tests: PASSED\n")
    return True

def main():
    setup_logging(level=20)
    
    print("\n" + "=" * 60)
    print("ParaScale Refactoring Test Suite")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    all_passed &= test_config()
    all_passed &= test_base_parallel()
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        all_passed &= test_distributed()
        dist.destroy_process_group()
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit(main())
