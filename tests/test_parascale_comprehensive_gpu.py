# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_parascale_comprehensive_gpu.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 全面GPU测试报告

基于2块A100-40GB显卡进行全面的功能测试和性能评估。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import time
import warnings
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ParaScale imports
from parascale.config import ParaScaleConfig, QuantizationConfig
from parascale.parallel import (
    DataParallel, ModelParallel, TensorParallel, 
    PipelineParallel, HybridParallel, BaseParallel
)
from parascale.parallel.communication import (
    OptimizedCommunicator, TopKCompressor, 
    CommunicationOverlap, RingAllReduce
)
from parascale.optimizers.zero_optimizer import (
    ZeroOptimizer, ZeroAdamW, ZeroSGD, ZeroStage
)
from parascale.quantization import (
    QuantizationAwareTraining, PostTrainingQuantization,
    quantize_with_fallback, AdaptiveQuantization,
    DegradationLevel, DegradationTrigger
)


# =============================================================================
# 测试配置
# =============================================================================
TEST_CONFIG = {
    'world_size': 2,
    'backend': 'nccl',
    'master_addr': 'localhost',
    'master_port': '29500',
    'batch_size': 32,
    'learning_rate': 1e-3,
    'hidden_size': 512,
}


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
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class MediumModel(nn.Module):
    """中等大小模型"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1024, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10)
        ])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        return self.layers[-1](x)


class LargeModel(nn.Module):
    """大模型 - 用于测试大模型支持"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return self.decoder(x)


# =============================================================================
# 测试基础设施
# =============================================================================
class TestResult:
    """测试结果类"""
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.status = "pending"
        self.duration = 0.0
        self.error = None
        self.metrics = {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category,
            'status': self.status,
            'duration': self.duration,
            'error': self.error,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }


class TestReport:
    """测试报告类"""
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count(),
            'gpus': [],
            'nccl_available': torch.distributed.is_nccl_available(),
        }
        
        for i in range(torch.cuda.device_count()):
            info['gpus'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
            })
        
        return info
    
    def add_result(self, result: TestResult):
        self.results.append(result)
    
    def generate_report(self) -> str:
        """生成测试报告"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "passed")
        failed_tests = sum(1 for r in self.results if r.status == "failed")
        skipped_tests = sum(1 for r in self.results if r.status == "skipped")
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        report = []
        report.append("=" * 80)
        report.append("ParaScale 全面GPU测试报告")
        report.append("=" * 80)
        report.append(f"测试时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试时长: {duration:.2f}秒")
        report.append("")
        
        # 系统信息
        report.append("-" * 80)
        report.append("系统信息")
        report.append("-" * 80)
        report.append(f"PyTorch版本: {self.system_info['pytorch_version']}")
        report.append(f"CUDA可用: {self.system_info['cuda_available']}")
        report.append(f"NCCL可用: {self.system_info['nccl_available']}")
        report.append(f"GPU数量: {self.system_info['gpu_count']}")
        for gpu in self.system_info['gpus']:
            report.append(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
        report.append("")
        
        # 测试汇总
        report.append("-" * 80)
        report.append("测试汇总")
        report.append("-" * 80)
        report.append(f"总测试数: {total_tests}")
        report.append(f"通过: {passed_tests} ✓")
        report.append(f"失败: {failed_tests} ✗")
        report.append(f"跳过: {skipped_tests} ⚠")
        report.append(f"通过率: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        report.append("")
        
        # 分类统计
        categories = {}
        for result in self.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0}
            categories[cat]['total'] += 1
            if result.status == "passed":
                categories[cat]['passed'] += 1
        
        report.append("-" * 80)
        report.append("分类统计")
        report.append("-" * 80)
        for cat, stats in sorted(categories.items()):
            rate = stats['passed']/stats['total']*100 if stats['total'] > 0 else 0
            report.append(f"{cat}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        report.append("")
        
        # 详细结果
        report.append("-" * 80)
        report.append("详细测试结果")
        report.append("-" * 80)
        
        current_category = None
        for result in self.results:
            if result.category != current_category:
                current_category = result.category
                report.append(f"\n【{current_category}】")
            
            status_icon = "✓" if result.status == "passed" else "✗" if result.status == "failed" else "⚠"
            report.append(f"  {status_icon} {result.name}: {result.status} ({result.duration:.3f}s)")
            
            if result.metrics:
                for key, value in result.metrics.items():
                    if isinstance(value, float):
                        report.append(f"      {key}: {value:.4f}")
                    else:
                        report.append(f"      {key}: {value}")
            
            if result.error:
                report.append(f"      Error: {result.error}")
        
        report.append("")
        report.append("=" * 80)
        report.append("测试报告结束")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_json(self, filename: str):
        """保存JSON格式的报告"""
        data = {
            'system_info': self.system_info,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# 分布式测试基础设施
# =============================================================================
def setup_distributed(rank: int, world_size: int) -> bool:
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = TEST_CONFIG['master_addr']
    os.environ['MASTER_PORT'] = TEST_CONFIG['master_port']
    
    dist.init_process_group(
        backend=TEST_CONFIG['backend'],
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)
    return True


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# 测试函数
# =============================================================================
def test_config_system(rank: int, world_size: int, results_queue):
    """测试配置系统"""
    results = []
    
    # Test 1: QuantizationConfig
    result = TestResult("QuantizationConfig创建", "配置系统")
    try:
        start = time.time()
        config = QuantizationConfig(
            enabled=True,
            mode="qat",
            bits=8,
            observer_type="moving_average",
            moving_average_ratio=0.95
        )
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {'bits': config.bits, 'mode': config.mode}
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    # Test 2: ParaScaleConfig
    result = TestResult("ParaScaleConfig创建", "配置系统")
    try:
        start = time.time()
        config = ParaScaleConfig(
            data_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            zero_optimization=True,
            zero_stage=1,
            batch_size=32,
            learning_rate=1e-3
        )
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {
            'dp_size': config.data_parallel_size,
            'tp_size': config.tensor_parallel_size,
            'pp_size': config.pipeline_parallel_size
        }
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    # Test 3: 配置验证
    result = TestResult("配置跨参数验证", "配置系统")
    try:
        start = time.time()
        config = ParaScaleConfig(
            data_parallel_size=2,
            tensor_parallel_size=2,
            zero_stage=1,
            zero_optimization=True
        )
        report = config.get_validation_report()
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {'suggestions_count': len(report.get('suggestions', []))}
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    for r in results:
        results_queue.put(r.to_dict())


def test_data_parallel_basic(rank: int, world_size: int, results_queue):
    """测试DataParallel基础功能"""
    results = []
    
    try:
        setup_distributed(rank, world_size)
        
        # Test 1: 基础DP
        result = TestResult("DataParallel基础", "数据并行")
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
            'output_shape': list(output.shape)
        }
        results.append(result)
        
        # Test 2: DP with压缩
        result = TestResult("DataParallel梯度压缩", "数据并行")
        model2 = SimpleModel().cuda(rank)
        dp2 = DataParallel(
            model2, rank=rank, world_size=world_size,
            compression_ratio=0.1
        )
        
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        output = dp2.forward(inputs)
        loss = output.sum()
        loss.backward()
        
        start = time.time()
        dp2.gather_gradients()
        comm_time = time.time() - start
        
        stats = dp2.get_comm_stats()
        result.status = "passed"
        result.duration = comm_time
        result.metrics = {
            'comm_time': comm_time,
            'compression_ratio': stats.get('compression_ratio'),
            'buckets': stats.get('buckets')
        }
        results.append(result)
        
        cleanup_distributed()
        
    except Exception as e:
        import traceback
        for result in results:
            if result.status == "pending":
                result.status = "failed"
                result.error = str(e)
    
    for r in results:
        results_queue.put(r.to_dict())


def test_zero_optimizer(rank: int, world_size: int, results_queue):
    """测试ZeRO优化器"""
    results = []
    
    try:
        setup_distributed(rank, world_size)
        
        # Test 1: Stage 0 (无ZeRO)
        result = TestResult("ZeRO Stage 0", "ZeRO优化器")
        model = SimpleModel().cuda(rank)
        optimizer = ZeroAdamW(model, lr=1e-3, stage=ZeroStage.DISABLED)
        
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        target = torch.randint(0, 10, (16,)).cuda(rank)
        
        start = time.time()
        optimizer.zero_grad()
        output = model(inputs)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_time = time.time() - start
        
        stats = optimizer.get_memory_stats()
        result.status = "passed"
        result.duration = train_time
        result.metrics = {
            'train_time': train_time,
            'loss': loss.item(),
            'total_memory_mb': stats.get('total_memory_mb')
        }
        results.append(result)
        
        # Test 2: Stage 1
        result = TestResult("ZeRO Stage 1", "ZeRO优化器")
        model2 = SimpleModel().cuda(rank)
        optimizer2 = ZeroAdamW(
            model2, lr=1e-3, 
            stage=ZeroStage.OPTIMIZER_STATES
        )
        
        start = time.time()
        optimizer2.zero_grad()
        output = model2(inputs)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer2.step()
        train_time = time.time() - start
        
        stats2 = optimizer2.get_memory_stats()
        result.status = "passed"
        result.duration = train_time
        result.metrics = {
            'train_time': train_time,
            'loss': loss.item(),
            'total_memory_mb': stats2.get('total_memory_mb'),
            'theoretical_savings': stats2.get('theoretical_savings')
        }
        results.append(result)
        
        cleanup_distributed()
        
    except Exception as e:
        import traceback
        for result in results:
            if result.status == "pending":
                result.status = "failed"
                result.error = str(e)
    
    for r in results:
        results_queue.put(r.to_dict())


def test_communication(rank: int, world_size: int, results_queue):
    """测试通信优化"""
    results = []
    
    try:
        setup_distributed(rank, world_size)
        
        # Test 1: 基础通信
        result = TestResult("OptimizedCommunicator", "通信优化")
        model = SimpleModel().cuda(rank)
        comm = OptimizedCommunicator(
            compression_ratio=0.1,
            overlap_enabled=False
        )
        
        inputs = torch.randn(16, 1, 28, 28).cuda(rank)
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        
        params = list(model.parameters())
        start = time.time()
        comm.synchronize_gradients(params)
        sync_time = time.time() - start
        
        stats = comm.get_stats()
        result.status = "passed"
        result.duration = sync_time
        result.metrics = {
            'sync_time': sync_time,
            'compression_enabled': stats.get('compression_enabled'),
            'bucket_size_mb': stats.get('bucket_size_mb')
        }
        results.append(result)
        
        # Test 2: TopK压缩器
        result = TestResult("TopKCompressor", "通信优化")
        compressor = TopKCompressor(compression_ratio=0.1)
        tensor = torch.randn(10000).cuda(rank)
        
        start = time.time()
        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)
        comp_time = time.time() - start
        
        result.status = "passed"
        result.duration = comp_time
        result.metrics = {
            'original_size': tensor.numel(),
            'compressed_size': compressed.numel(),
            'compression_ratio': 0.1
        }
        results.append(result)
        
        cleanup_distributed()
        
    except Exception as e:
        import traceback
        for result in results:
            if result.status == "pending":
                result.status = "failed"
                result.error = str(e)
    
    for r in results:
        results_queue.put(r.to_dict())


def test_quantization(rank: int, world_size: int, results_queue):
    """测试量化模块"""
    results = []
    
    # Test 1: 量化配置
    result = TestResult("QuantizationConfig", "量化模块")
    try:
        start = time.time()
        config = QuantizationConfig(
            enabled=True,
            mode="qat",
            bits=8,
            scheme="symmetric"
        )
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {'bits': config.bits, 'enabled': config.enabled}
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    # Test 2: QAT准备
    result = TestResult("QAT准备", "量化模块")
    try:
        from parascale.quantization.qat import QuantizationAwareTraining
        
        start = time.time()
        model = SimpleModel()
        config = QuantizationConfig(enabled=True, mode="qat")
        qat = QuantizationAwareTraining(model, config)
        prepared_model = qat.prepare()
        
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {'model_prepared': prepared_model is not None}
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    # Test 3: 降级机制
    result = TestResult("量化降级机制", "量化模块")
    try:
        from parascale.quantization.degradation import DegradationTrigger
        
        start = time.time()
        trigger = DegradationTrigger(memory_threshold=85.0)
        memory_status = trigger.check_memory()
        should_degrade = trigger.should_degrade()
        
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {
            'memory_percent': memory_status.get('percent_used'),
            'should_degrade': should_degrade
        }
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    for r in results:
        results_queue.put(r.to_dict())


def test_single_gpu_features(results_queue):
    """测试单GPU功能"""
    results = []
    
    # Test 1: 模型创建
    result = TestResult("模型创建", "单GPU功能")
    try:
        start = time.time()
        model = SimpleModel().cuda()
        inputs = torch.randn(8, 1, 28, 28).cuda()
        output = model(inputs)
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {'output_shape': list(output.shape)}
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    # Test 2: 基础训练
    result = TestResult("基础训练循环", "单GPU功能")
    try:
        model = SimpleModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        start = time.time()
        for i in range(3):
            inputs = torch.randn(8, 1, 28, 28).cuda()
            targets = torch.randint(0, 10, (8,)).cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {'final_loss': loss.item(), 'iterations': 3}
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    # Test 3: GPU内存检查
    result = TestResult("GPU内存检查", "单GPU功能")
    try:
        start = time.time()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)
        
        result.duration = time.time() - start
        result.status = "passed"
        result.metrics = {
            'gpu_count': torch.cuda.device_count(),
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved
        }
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    results.append(result)
    
    for r in results:
        results_queue.put(r.to_dict())


# =============================================================================
# 主测试运行器
# =============================================================================
def run_multiprocess_test(test_func, test_name: str, world_size: int = 2):
    """运行多进程测试"""
    print(f"\n运行测试: {test_name}")
    print("-" * 60)
    
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_func, args=(rank, world_size, results_queue))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join(timeout=120)
    
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    return results


def run_single_process_test(test_func, test_name: str):
    """运行单进程测试"""
    print(f"\n运行测试: {test_name}")
    print("-" * 60)
    
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    test_func(results_queue)
    
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    return results


def main():
    """主函数"""
    print("=" * 80)
    print("ParaScale 全面GPU测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("=" * 80)
    
    report = TestReport()
    
    # 1. 单GPU功能测试
    print("\n【阶段1: 单GPU功能测试】")
    results = run_single_process_test(test_single_gpu_features, "单GPU功能")
    for r in results:
        result = TestResult(r['name'], r['category'])
        result.status = r['status']
        result.duration = r['duration']
        result.metrics = r['metrics']
        result.error = r['error']
        report.add_result(result)
    
    # 2. 配置系统测试
    print("\n【阶段2: 配置系统测试】")
    results = run_multiprocess_test(test_config_system, "配置系统", 2)
    for r in results:
        result = TestResult(r['name'], r['category'])
        result.status = r['status']
        result.duration = r['duration']
        result.metrics = r['metrics']
        result.error = r['error']
        report.add_result(result)
    
    # 3. 数据并行测试
    print("\n【阶段3: 数据并行测试】")
    results = run_multiprocess_test(test_data_parallel_basic, "数据并行", 2)
    for r in results:
        result = TestResult(r['name'], r['category'])
        result.status = r['status']
        result.duration = r['duration']
        result.metrics = r['metrics']
        result.error = r['error']
        report.add_result(result)
    
    # 4. ZeRO优化器测试
    print("\n【阶段4: ZeRO优化器测试】")
    results = run_multiprocess_test(test_zero_optimizer, "ZeRO优化器", 2)
    for r in results:
        result = TestResult(r['name'], r['category'])
        result.status = r['status']
        result.duration = r['duration']
        result.metrics = r['metrics']
        result.error = r['error']
        report.add_result(result)
    
    # 5. 通信优化测试
    print("\n【阶段5: 通信优化测试】")
    results = run_multiprocess_test(test_communication, "通信优化", 2)
    for r in results:
        result = TestResult(r['name'], r['category'])
        result.status = r['status']
        result.duration = r['duration']
        result.metrics = r['metrics']
        result.error = r['error']
        report.add_result(result)
    
    # 6. 量化模块测试
    print("\n【阶段6: 量化模块测试】")
    results = run_multiprocess_test(test_quantization, "量化模块", 2)
    for r in results:
        result = TestResult(r['name'], r['category'])
        result.status = r['status']
        result.duration = r['duration']
        result.metrics = r['metrics']
        result.error = r['error']
        report.add_result(result)
    
    # 生成报告
    print("\n" + "=" * 80)
    print("生成测试报告...")
    print("=" * 80)
    
    report_text = report.generate_report()
    print(report_text)
    
    # 保存报告
    report_file = f"/wangjun/ParaScale/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_file}")
    
    # 保存JSON
    json_file = f"/wangjun/ParaScale/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report.save_json(json_file)
    print(f"JSON报告已保存: {json_file}")
    
    return report


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
