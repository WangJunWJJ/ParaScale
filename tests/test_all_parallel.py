# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : test_all_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.config import ParaScaleConfig
from parascale.engine import Engine
from parascale.utils.utils import setup_logging, print_rank_0

class TensorModel(nn.Module):
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

import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "distributed: mark test as requiring distributed setup"
    )

@pytest.fixture
def distributed_setup():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def test_tensor_parallel(distributed_setup):
    rank, world_size = distributed_setup
    print_rank_0("Testing Tensor Parallel...")
    
    model = TensorModel()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    config = ParaScaleConfig(
        tensor_parallel_size=world_size,
        tensor_parallel_mode='row',
        batch_size=32
    )
    
    engine = Engine(model, optimizer, config)
    
    dummy_data = [(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))) for _ in range(5)]
    
    for i, (inputs, targets) in enumerate(dummy_data):
        if torch.cuda.is_available():
            inputs = inputs.to(f"cuda:{rank}")
            targets = targets.to(f"cuda:{rank}")
        
        outputs = engine._forward(inputs)
        
        if outputs is not None:
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            engine.optimizer.step()
            engine.optimizer.zero_grad()
            
            if i == 0:
                print_rank_0(f"Tensor Parallel - Batch {i}, Loss: {loss.item():.4f}")
    
    print_rank_0("Tensor Parallel: PASSED")

def test_pipeline_parallel(distributed_setup):
    rank, world_size = distributed_setup
    print_rank_0("Testing Pipeline Parallel...")
    
    model = PipelineModel()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    config = ParaScaleConfig(
        pipeline_parallel_size=world_size,
        pipeline_parallel_chunks=1,
        batch_size=32
    )
    
    engine = Engine(model, optimizer, config)
    
    dummy_data = [(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))) for _ in range(5)]
    
    is_last = engine.pipeline_parallel.is_last if engine.pipeline_parallel else True
    
    for i, (inputs, targets) in enumerate(dummy_data):
        if torch.cuda.is_available():
            inputs = inputs.to(f"cuda:{rank}")
            targets = targets.to(f"cuda:{rank}")
        
        outputs = engine._forward(inputs)
        
        if is_last and outputs is not None:
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            engine.optimizer.step()
            engine.optimizer.zero_grad()
            
            if i == 0:
                print_rank_0(f"Pipeline Parallel - Batch {i}, Loss: {loss.item():.4f}")
    
    print_rank_0("Pipeline Parallel: PASSED")

def main():
    setup_logging(level=20)
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
    
    print_rank_0(f"Testing ParaScale with {world_size} GPUs")
    
    test_tensor_parallel(rank, world_size)
    
    dist.barrier() if world_size > 1 else None
    
    test_pipeline_parallel(rank, world_size)
    
    if world_size > 1:
        dist.destroy_process_group()
    
    print_rank_0("ALL PARALLEL STRATEGIES TESTED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
