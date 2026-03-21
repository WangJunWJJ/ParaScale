# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_gpu_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale GPU实际测试模块

基于2块A100-40GB显卡进行实际的分布式并行测试。
使用torch.multiprocessing启动多进程模拟多GPU环境。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.parallel import DataParallel, ModelParallel, HybridParallel
from parascale.parallel.communication import OptimizedCommunicator, TopKCompressor
from parascale.optimizers.zero_optimizer import ZeroOptimizer, ZeroAdamW, ZeroStage


# 测试配置
TEST_CONFIG = {
    'world_size': 2,
    'backend': 'nccl',  # NCCL后端用于GPU
    'epochs': 3,
    'batch_size': 32,
    'learning_rate': 1e-3,
}


class TestModel(nn.Module):
    """测试模型 - 适合2 GPU测试"""
    def __init__(self, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class LargeTestModel(nn.Module):
    """大测试模型 - 用于模型并行测试"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1024, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10)
        ])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
        return self.layers[-1](x)


def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(
        backend=TEST_CONFIG['backend'],
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    
    return True


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_data_parallel(rank, world_size, results):
    """测试DataParallel"""
    try:
        setup_distributed(rank, world_size)
        
        # 创建模型
        model = TestModel().cuda(rank)
        
        # 创建DataParallel
        dp = DataParallel(
            model,
            rank=rank,
            world_size=world_size,
            compression_ratio=None,  # 先不压缩
            overlap_comm=False
        )
        
        # 前向传播
        batch_size = 16
        inputs = torch.randn(batch_size, 1, 28, 28).cuda(rank)
        
        start_time = time.time()
        output = dp.forward(inputs)
        forward_time = time.time() - start_time
        
        # 反向传播
        loss = output.sum()
        loss.backward()
        
        # 梯度同步
        start_time = time.time()
        dp.gather_gradients()
        comm_time = time.time() - start_time
        
        # 获取通信统计
        comm_stats = dp.get_comm_stats()
        
        results[rank] = {
            'status': 'success',
            'forward_time': forward_time,
            'comm_time': comm_time,
            'comm_stats': comm_stats,
            'output_shape': list(output.shape),
        }
        
        cleanup_distributed()
        
    except Exception as e:
        results[rank] = {'status': 'failed', 'error': str(e)}
        cleanup_distributed()


def test_data_parallel_with_compression(rank, world_size, results):
    """测试DataParallel with梯度压缩"""
    try:
        setup_distributed(rank, world_size)
        
        model = TestModel().cuda(rank)
        
        # 启用梯度压缩
        dp = DataParallel(
            model,
            rank=rank,
            world_size=world_size,
            compression_ratio=0.1,  # 10%压缩
            overlap_comm=False
        )
        
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        output = dp.forward(inputs)
        loss = output.sum()
        loss.backward()
        
        start_time = time.time()
        dp.gather_gradients()
        comm_time = time.time() - start_time
        
        comm_stats = dp.get_comm_stats()
        
        results[rank] = {
            'status': 'success',
            'comm_time': comm_time,
            'compression_ratio': comm_stats['compression_ratio'],
        }
        
        cleanup_distributed()
        
    except Exception as e:
        results[rank] = {'status': 'failed', 'error': str(e)}
        cleanup_distributed()


def test_model_parallel(rank, world_size, results):
    """测试ModelParallel"""
    try:
        setup_distributed(rank, world_size)
        
        model = LargeTestModel()
        
        # 创建ModelParallel
        mp = ModelParallel(
            model,
            rank=rank,
            world_size=world_size,
            balance_strategy="compute_cost"
        )
        
        # 前向传播
        inputs = torch.randn(8, 1, 32, 32).cuda(rank)
        
        start_time = time.time()
        output = mp.forward(inputs)
        forward_time = time.time() - start_time
        
        # 获取负载均衡报告
        load_report = mp.get_load_balance_report()
        
        results[rank] = {
            'status': 'success',
            'forward_time': forward_time,
            'is_last_stage': mp.is_last_stage,
            'load_report': load_report if mp.is_last_stage else None,
        }
        
        cleanup_distributed()
        
    except Exception as e:
        results[rank] = {'status': 'failed', 'error': str(e)}
        cleanup_distributed()


def test_hybrid_parallel(rank, world_size, results):
    """测试HybridParallel (DP=2, TP=1, PP=1 - 纯数据并行)"""
    try:
        setup_distributed(rank, world_size)
        
        model = TestModel()
        
        # 2 GPU: DP=2, TP=1, PP=1
        hp = HybridParallel(
            model,
            rank=rank,
            world_size=world_size,
            dp_size=2,
            tp_size=1,
            pp_size=1,
            pipeline_chunks=1
        )
        
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        
        start_time = time.time()
        output = hp.forward(inputs)
        forward_time = time.time() - start_time
        
        if output is not None:
            loss = output.sum()
            loss.backward()
            hp.gather_gradients()
        
        parallel_info = hp.get_parallel_info()
        
        results[rank] = {
            'status': 'success',
            'forward_time': forward_time,
            'parallel_info': parallel_info,
            'has_output': output is not None,
        }
        
        cleanup_distributed()
        
    except Exception as e:
        results[rank] = {'status': 'failed', 'error': str(e)}
        cleanup_distributed()


def test_zero_optimizer(rank, world_size, results):
    """测试ZeRO优化器"""
    try:
        setup_distributed(rank, world_size)
        
        model = TestModel().cuda(rank)
        
        # 创建ZeRO Stage 1优化器
        optimizer = ZeroAdamW(
            model,
            lr=1e-3,
            stage=ZeroStage.OPTIMIZER_STATES,
            offload_optimizer=False
        )
        
        # 训练步骤
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        target = torch.randint(0, 10, (16,)).cuda(rank)
        
        start_time = time.time()
        
        optimizer.zero_grad()
        output = model(inputs)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        train_time = time.time() - start_time
        
        # 获取内存统计
        memory_stats = optimizer.get_memory_stats()
        
        results[rank] = {
            'status': 'success',
            'train_time': train_time,
            'loss': loss.item(),
            'memory_stats': memory_stats,
        }
        
        cleanup_distributed()
        
    except Exception as e:
        results[rank] = {'status': 'failed', 'error': str(e)}
        cleanup_distributed()


def test_communication_optimization(rank, world_size, results):
    """测试通信优化"""
    try:
        setup_distributed(rank, world_size)
        
        model = TestModel().cuda(rank)
        
        # 测试带压缩的通信器
        comm = OptimizedCommunicator(
            compression_ratio=0.1,
            overlap_enabled=False,
            bucket_size=10*1024*1024
        )
        
        # 生成梯度
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        
        # 同步梯度
        params = list(model.parameters())
        start_time = time.time()
        comm.synchronize_gradients(params)
        sync_time = time.time() - start_time
        
        stats = comm.get_stats()
        
        results[rank] = {
            'status': 'success',
            'sync_time': sync_time,
            'comm_stats': stats,
        }
        
        cleanup_distributed()
        
    except Exception as e:
        results[rank] = {'status': 'failed', 'error': str(e)}
        cleanup_distributed()


def run_test(test_func, test_name):
    """运行多进程测试"""
    print(f"\n{'='*60}")
    print(f"运行测试: {test_name}")
    print(f"{'='*60}")
    
    world_size = TEST_CONFIG['world_size']
    
    # 使用spawn方法启动多进程
    mp.set_start_method('spawn', force=True)
    
    # 共享结果字典
    manager = mp.Manager()
    results = manager.dict()
    
    # 启动进程
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_func, args=(rank, world_size, results))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join(timeout=60)  # 60秒超时
    
    # 检查结果
    success = all(r['status'] == 'success' for r in results.values())
    
    print(f"\n测试结果 ({test_name}):")
    for rank in range(world_size):
        result = results.get(rank, {'status': 'timeout'})
        status = "✓" if result['status'] == 'success' else "✗"
        print(f"  Rank {rank}: {status} {result['status']}")
        
        if result['status'] == 'success':
            # 打印关键指标
            for key in ['forward_time', 'comm_time', 'train_time', 'sync_time']:
                if key in result:
                    print(f"    {key}: {result[key]:.4f}s")
    
    return success, dict(results)


def main():
    """主函数 - 运行所有GPU测试"""
    print("\n" + "="*60)
    print("ParaScale GPU实际测试")
    print("="*60)
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NCCL可用: {torch.distributed.is_nccl_available()}")
    
    all_results = {}
    
    # 测试1: DataParallel
    success, results = run_test(test_data_parallel, "DataParallel (基础)")
    all_results['data_parallel'] = {'success': success, 'results': results}
    
    # 测试2: DataParallel with压缩
    success, results = run_test(test_data_parallel_with_compression, "DataParallel (梯度压缩)")
    all_results['data_parallel_compression'] = {'success': success, 'results': results}
    
    # 测试3: ModelParallel
    success, results = run_test(test_model_parallel, "ModelParallel")
    all_results['model_parallel'] = {'success': success, 'results': results}
    
    # 测试4: HybridParallel
    success, results = run_test(test_hybrid_parallel, "HybridParallel (DP=2)")
    all_results['hybrid_parallel'] = {'success': success, 'results': results}
    
    # 测试5: ZeRO优化器
    success, results = run_test(test_zero_optimizer, "ZeRO优化器 (Stage 1)")
    all_results['zero_optimizer'] = {'success': success, 'results': results}
    
    # 测试6: 通信优化
    success, results = run_test(test_communication_optimization, "通信优化")
    all_results['communication'] = {'success': success, 'results': results}
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results.values() if r['success'])
    
    for test_name, result in all_results.items():
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("\n🎉 所有GPU测试通过！")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} 个测试失败")
    
    return passed_tests == total_tests


if __name__ == '__main__':
    # 忽略一些警告
    warnings.filterwarnings('ignore')
    
    success = main()
    sys.exit(0 if success else 1)
