# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_all_parallel_torchrun_v2.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 全并行策略测试脚本 v2 (修复版)

使用torchrun启动多进程，测试所有并行策略。

运行方式:
    torchrun --nproc_per_node=2 tests/test_all_parallel_torchrun_v2.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.parallel import (
    DataParallel, ModelParallel, TensorParallel,
    PipelineParallel, HybridParallel
)


# =============================================================================
# 测试模型
# =============================================================================
class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
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
    """中等模型 - 用于MP测试"""
    def __init__(self, input_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)


# =============================================================================
# 测试结果记录
# =============================================================================
class TestResult:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.status = "pending"
        self.duration = 0.0
        self.error = None
        self.metrics = {}
        self.rank = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category,
            'status': self.status,
            'duration': self.duration,
            'error': self.error,
            'metrics': self.metrics,
            'rank': self.rank
        }


# =============================================================================
# 测试函数
# =============================================================================
def test_data_parallel():
    """测试数据并行"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    result = TestResult("DataParallel基础", "数据并行")
    result.rank = rank
    
    try:
        model = SimpleModel().cuda(rank)
        dp = DataParallel(model, rank=rank, world_size=world_size)
        
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        start = time.time()
        output = dp.forward(inputs)
        forward_time = time.time() - start
        
        loss = output.sum()
        loss.backward()
        
        start = time.time()
        dp.gather_gradients()
        comm_time = time.time() - start
        
        result.status = "passed"
        result.duration = forward_time + comm_time
        result.metrics = {
            'forward_time': forward_time,
            'comm_time': comm_time,
            'output_shape': list(output.shape),
            'world_size': world_size
        }
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_model_parallel():
    """测试模型并行"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    result = TestResult("ModelParallel", "模型并行")
    result.rank = rank
    
    try:
        # 使用正确的输入尺寸
        model = MediumModel(input_size=1024)
        mp = ModelParallel(
            model,
            rank=rank,
            world_size=world_size,
            balance_strategy="compute_cost"
        )
        
        # 使用与模型输入匹配的tensor
        inputs = torch.randn(8, 1024).cuda(rank)
        start = time.time()
        output = mp.forward(inputs)
        forward_time = time.time() - start
        
        # ModelParallel没有is_last_stage属性，使用pp_rank判断
        is_last_stage = (rank == world_size - 1)
        
        load_report = mp.get_load_balance_report()
        
        result.status = "passed"
        result.duration = forward_time
        result.metrics = {
            'forward_time': forward_time,
            'is_last_stage': is_last_stage,
            'total_layers': load_report.get('total_layers'),
            'my_layers': load_report['ranks'][rank]['layers'] if rank < len(load_report['ranks']) else 0
        }
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_tensor_parallel():
    """测试张量并行"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    result = TestResult("TensorParallel", "张量并行")
    result.rank = rank
    
    try:
        model = SimpleModel().cuda(rank)
        tp = TensorParallel(
            model,
            rank=rank,
            world_size=world_size,
            mode="row"
        )
        
        inputs = torch.randn(8, 1, 28, 28).cuda(rank)
        start = time.time()
        output = tp.forward(inputs)
        forward_time = time.time() - start
        
        result.status = "passed"
        result.duration = forward_time
        result.metrics = {
            'forward_time': forward_time,
            'output_shape': list(output.shape) if output is not None else None,
            'mode': 'row'
        }
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_pipeline_parallel():
    """测试流水线并行"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    result = TestResult("PipelineParallel", "流水线并行")
    result.rank = rank
    
    try:
        # 使用正确的参数名 chunks 而不是 num_chunks
        model = MediumModel(input_size=1024).cuda(rank)
        pp = PipelineParallel(
            model,
            rank=rank,
            world_size=world_size,
            chunks=2  # 使用chunks而不是num_chunks
        )
        
        inputs = torch.randn(8, 1024).cuda(rank)
        start = time.time()
        output = pp.forward(inputs)
        forward_time = time.time() - start
        
        result.status = "passed"
        result.duration = forward_time
        result.metrics = {
            'forward_time': forward_time,
            'has_output': output is not None,
            'chunks': 2
        }
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_hybrid_parallel():
    """测试3D混合并行"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    result = TestResult("HybridParallel (DP=2)", "3D混合并行")
    result.rank = rank
    
    try:
        model = SimpleModel().cuda(rank)
        hp = HybridParallel(
            model,
            rank=rank,
            world_size=world_size,
            dp_size=2,
            tp_size=1,
            pp_size=1,
            tensor_parallel_mode="row",
            pipeline_chunks=1
        )
        
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        start = time.time()
        output = hp.forward(inputs)
        forward_time = time.time() - start
        
        if output is not None:
            loss = output.sum()
            loss.backward()
            hp.gather_gradients()
        
        parallel_info = hp.get_parallel_info()
        
        result.status = "passed"
        result.duration = forward_time
        result.metrics = {
            'forward_time': forward_time,
            'dp_size': parallel_info.get('dp_size'),
            'tp_size': parallel_info.get('tp_size'),
            'pp_size': parallel_info.get('pp_size'),
            'has_output': output is not None
        }
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


# =============================================================================
# 主函数
# =============================================================================
def main():
    """主测试函数"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("=" * 80)
        print("ParaScale 全并行策略测试 v2 (修复版)")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"World Size: {world_size}")
        print("=" * 80)
    
    dist.barrier()
    
    all_results = []
    
    # 1. 测试数据并行
    if rank == 0:
        print("\n【测试1: 数据并行】")
    dist.barrier()
    result = test_data_parallel()
    all_results.append(result.to_dict())
    
    # 2. 测试模型并行
    if rank == 0:
        print("\n【测试2: 模型并行】")
    dist.barrier()
    result = test_model_parallel()
    all_results.append(result.to_dict())
    
    # 3. 测试张量并行
    if rank == 0:
        print("\n【测试3: 张量并行】")
    dist.barrier()
    result = test_tensor_parallel()
    all_results.append(result.to_dict())
    
    # 4. 测试流水线并行
    if rank == 0:
        print("\n【测试4: 流水线并行】")
    dist.barrier()
    result = test_pipeline_parallel()
    all_results.append(result.to_dict())
    
    # 5. 测试3D混合并行
    if rank == 0:
        print("\n【测试5: 3D混合并行】")
    dist.barrier()
    result = test_hybrid_parallel()
    all_results.append(result.to_dict())
    
    dist.barrier()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("生成测试报告...")
        print("=" * 80)
        
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r['status'] == "passed")
        failed_tests = sum(1 for r in all_results if r['status'] == "failed")
        
        categories = {}
        for r in all_results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        print(f"\n总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✓")
        print(f"失败: {failed_tests} ✗")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")
        print()
        
        for cat, results in sorted(categories.items()):
            print(f"\n【{cat}】")
            for r in results:
                status = "✓" if r['status'] == "passed" else "✗"
                print(f"  {status} {r['name']}: {r['status']} ({r['duration']:.3f}s)")
                if r['metrics']:
                    for key, value in r['metrics'].items():
                        if isinstance(value, float):
                            print(f"      {key}: {value:.4f}")
                        else:
                            print(f"      {key}: {value}")
                if r['error']:
                    print(f"      Error: {r['error']}")
        
        print("\n" + "=" * 80)
        print("测试报告结束")
        print("=" * 80)
        
        report_file = f"/wangjun/ParaScale/parallel_test_report_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'world_size': world_size,
                'results': all_results
            }, f, indent=2)
        print(f"\n报告已保存: {report_file}")
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
