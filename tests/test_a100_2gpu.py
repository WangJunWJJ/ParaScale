# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_a100_2gpu.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 2x A100 GPU 完整测试脚本

本脚本在2x A100-40GB GPU环境下测试所有并行策略的性能。

运行方式:
    torchrun --nproc_per_node=2 tests/test_a100_2gpu.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import os
import sys
import time
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.parallel import (
    DataParallel,
    ModelParallel,
    TensorParallel,
    TensorParallelConfig,
    HybridParallel,
    HybridParallelConfig,
    PipelineParallel,
    ParallelStrategy,
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, input_size=784, hidden_size=2048, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class MediumModel(nn.Module):
    """中等规模模型"""
    def __init__(self, input_size=1024, hidden_size=4096, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
            for i in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def test_data_parallel(rank: int, world_size: int) -> Dict[str, any]:
    """测试数据并行"""
    device = torch.device(f"cuda:{rank}")
    
    model = SimpleModel().to(device)
    dp = DataParallel(model, rank=rank, world_size=world_size)
    optimizer = optim.AdamW(dp.model.parameters(), lr=1e-4)
    
    batch_size = 64
    input_data = torch.randn(batch_size, 784, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    
    # 预热
    for _ in range(3):
        optimizer.zero_grad()
        output = dp(input_data)
        loss = criterion(output, target)
        loss.backward()
        dp.gather_gradients()
        optimizer.step()
    
    # 测试
    torch.cuda.reset_peak_memory_stats()
    times = []
    
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        output = dp(input_data)
        loss = criterion(output, target)
        loss.backward()
        dp.gather_gradients()
        optimizer.step()
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        'strategy': 'DataParallel',
        'avg_time_ms': sum(times) / len(times) * 1000,
        'throughput': batch_size / (sum(times) / len(times)),
        'memory_gb': torch.cuda.max_memory_allocated() / 1024**3,
    }


def test_tensor_parallel(rank: int, world_size: int) -> Dict[str, any]:
    """测试张量并行"""
    device = torch.device(f"cuda:{rank}")
    
    model = MediumModel().to(device)
    config = TensorParallelConfig(tp_size=world_size, strategy=ParallelStrategy.SIMPLE)
    tp = TensorParallel(model, rank=rank, world_size=world_size, config=config)
    optimizer = optim.AdamW(tp.model.parameters(), lr=1e-4)
    
    batch_size = 32
    input_data = torch.randn(batch_size, 1024, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    
    # 预热
    for _ in range(3):
        optimizer.zero_grad()
        output = tp(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 测试
    torch.cuda.reset_peak_memory_stats()
    times = []
    
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        output = tp(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        'strategy': 'TensorParallel',
        'avg_time_ms': sum(times) / len(times) * 1000,
        'throughput': batch_size / (sum(times) / len(times)),
        'memory_gb': torch.cuda.max_memory_allocated() / 1024**3,
    }


def test_pipeline_parallel(rank: int, world_size: int) -> Dict[str, any]:
    """测试流水线并行"""
    device = torch.device(f"cuda:{rank}")
    
    model = MediumModel(num_layers=16).to(device)
    pp = PipelineParallel(model, rank=rank, world_size=world_size, chunks=4)
    
    batch_size = 64
    input_data = torch.randn(batch_size, 1024, device=device)
    
    # 预热
    for _ in range(3):
        output = pp(input_data)
        if output is not None:
            loss = output.sum()
            loss.backward()
    
    # 测试
    torch.cuda.reset_peak_memory_stats()
    times = []
    
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = pp(input_data)
        if output is not None:
            loss = output.sum()
            loss.backward()
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        'strategy': 'PipelineParallel',
        'avg_time_ms': sum(times) / len(times) * 1000,
        'throughput': batch_size / (sum(times) / len(times)),
        'memory_gb': torch.cuda.max_memory_allocated() / 1024**3,
    }


def main():
    """主函数"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("\n" + "="*80)
        print("ParaScale 2x A100 GPU 性能测试")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"World Size: {world_size}")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("="*80)
    
    dist.barrier()
    
    all_results = []
    
    # 测试 DataParallel
    if rank == 0:
        print("\n测试 DataParallel...")
    result = test_data_parallel(rank, world_size)
    all_results.append(result)
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 测试 TensorParallel
    if rank == 0:
        print("测试 TensorParallel...")
    result = test_tensor_parallel(rank, world_size)
    all_results.append(result)
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 测试 PipelineParallel
    if rank == 0:
        print("测试 PipelineParallel...")
    result = test_pipeline_parallel(rank, world_size)
    all_results.append(result)
    dist.barrier()
    
    # 打印报告
    if rank == 0:
        print("\n" + "="*80)
        print("测试报告汇总")
        print("="*80)
        print(f"{'策略':<20} {'时间(ms)':<15} {'吞吐量(samples/s)':<20} {'内存(GB)':<10}")
        print("-"*80)
        
        for result in all_results:
            print(f"{result['strategy']:<20} "
                  f"{result['avg_time_ms']:<15.2f} "
                  f"{result['throughput']:<20.2f} "
                  f"{result['memory_gb']:<10.2f}")
        
        print("\n" + "="*80)
        print("测试完成 - 所有并行策略运行正常")
        print("="*80)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
