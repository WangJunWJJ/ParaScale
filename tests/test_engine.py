# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : test_engine.py
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

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3072, 10)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def main():
    setup_logging(level=20)
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
    
    print_rank_0(f"Testing ParaScale Engine with {world_size} GPUs")
    
    model = SimpleModel()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    config = ParaScaleConfig(
        data_parallel_size=world_size,
        batch_size=32,
        learning_rate=1e-3
    )
    
    print_rank_0(f"Config: {config}")
    
    engine = Engine(model, optimizer, config)
    
    print_rank_0("Creating dummy data...")
    dummy_data = [(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))) for _ in range(10)]
    
    print_rank_0("Starting training...")
    for i, (inputs, targets) in enumerate(dummy_data):
        if torch.cuda.is_available():
            inputs = inputs.to(f"cuda:{rank}")
            targets = targets.to(f"cuda:{rank}")
        
        outputs = engine._forward(inputs)
        
        if outputs is not None:
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            
            engine._gather_gradients()
            engine.optimizer.step()
            engine.optimizer.zero_grad()
            
            if i % 5 == 0:
                print_rank_0(f"Batch {i}, Loss: {loss.item():.4f}")
    
    print_rank_0("Training completed successfully!")
    
    if world_size > 1:
        dist.destroy_process_group()
    
    print_rank_0("ALL TESTS PASSED!")

if __name__ == '__main__':
    main()
