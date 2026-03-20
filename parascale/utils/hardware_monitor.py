# -*- coding: utf-8 -*-
# @Time    : 2026/3/19 下午15:30
# @Author  : Jun Wang
# @File    : hardware_monitor.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 实时硬件监控模块

本模块提供了实时监控 GPU 内存、计算能力、通信带宽的功能，
支持在训练过程中持续收集硬件状态信息。
"""

import torch
import torch.distributed as dist
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class HardwareMetrics:
    """
    硬件指标数据类
    
    存储单个时间点的硬件状态信息。
    
    Attributes:
        timestamp: 时间戳
        gpu_memory_used: 已使用的 GPU 内存（字节）
        gpu_memory_peak: 峰值 GPU 内存（字节）
        gpu_memory_percent: 内存使用百分比
        gpu_utilization: GPU 利用率（0-100）
        gpu_temperature: GPU 温度（摄氏度）
        communication_bandwidth: 通信带宽（GB/s）
        compute_throughput: 计算吞吐量（TFLOPS）
    """
    timestamp: float = field(default_factory=time.time)
    gpu_memory_used: int = 0
    gpu_memory_peak: int = 0
    gpu_memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    communication_bandwidth: float = 0.0
    compute_throughput: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'gpu_memory_used_mb': self.gpu_memory_used / (1024 ** 2),
            'gpu_memory_peak_mb': self.gpu_memory_peak / (1024 ** 2),
            'gpu_memory_percent': self.gpu_memory_percent,
            'gpu_utilization': self.gpu_utilization,
            'gpu_temperature': self.gpu_temperature,
            'communication_bandwidth_gb_s': self.communication_bandwidth,
            'compute_throughput_tflops': self.compute_throughput
        }


class RealTimeHardwareMonitor:
    """
    实时硬件监控器
    
    在训练过程中持续监控 GPU 内存、计算能力、通信带宽等硬件状态，
    为性能优化和资源管理提供实时数据支持。
    
    Attributes:
        local_rank: 本地 rank
        metrics_history: 指标历史记录
        max_history_size: 最大历史记录数量
        last_bandwidth_check: 上次带宽检查时间
        bandwidth_test_interval: 带宽测试间隔（秒）
        last_compute_check: 上次计算能力检查时间
        compute_check_interval: 计算能力检查间隔（秒）
    """
    
    def __init__(
        self,
        local_rank: int = 0,
        max_history_size: int = 1000,
        bandwidth_test_interval: float = 60.0,
        compute_check_interval: float = 30.0
    ):
        """
        初始化实时硬件监控器
        
        Args:
            local_rank: 本地 rank
            max_history_size: 最大历史记录数量
            bandwidth_test_interval: 带宽测试间隔（秒）
            compute_check_interval: 计算能力检查间隔（秒）
        """
        self.local_rank = local_rank
        self.metrics_history: list[HardwareMetrics] = []
        self.max_history_size = max_history_size
        self.last_bandwidth_check = 0.0
        self.bandwidth_test_interval = bandwidth_test_interval
        self.last_compute_check = 0.0
        self.compute_check_interval = compute_check_interval
        self.start_time = time.time()
        
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cpu')
    
    def collect_metrics(self) -> HardwareMetrics:
        """
        收集当前硬件指标
        
        Returns:
            当前硬件指标
        """
        current_time = time.time()
        
        metrics = HardwareMetrics(timestamp=current_time)
        
        if torch.cuda.is_available():
            metrics = self._collect_gpu_metrics(metrics)
            metrics = self._collect_compute_metrics(metrics)
        
        if dist.is_initialized():
            metrics = self._check_bandwidth(metrics, current_time)
        
        self._add_to_history(metrics)
        
        return metrics
    
    def _collect_gpu_metrics(self, metrics: HardwareMetrics) -> HardwareMetrics:
        """
        收集 GPU 相关指标
        
        Args:
            metrics: 硬件指标对象
        
        Returns:
            更新后的硬件指标
        """
        if not torch.cuda.is_available():
            return metrics
        
        total_memory = torch.cuda.get_device_properties(self.local_rank).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.local_rank)
        reserved_memory = torch.cuda.memory_reserved(self.local_rank)
        peak_memory = torch.cuda.max_memory_allocated(self.local_rank)
        
        metrics.gpu_memory_used = allocated_memory
        metrics.gpu_memory_peak = peak_memory
        metrics.gpu_memory_percent = (allocated_memory / total_memory) * 100.0
        
        try:
            utilization = torch.cuda.utilization(self.local_rank)
            if utilization is not None:
                metrics.gpu_utilization = utilization * 100.0
        except Exception as e:
            logger.debug(f"无法获取 GPU 利用率: {e}")
        
        try:
            temperature = torch.cuda.temperature(self.local_rank)
            if temperature is not None:
                metrics.gpu_temperature = temperature
        except Exception as e:
            logger.debug(f"无法获取 GPU 温度: {e}")
        
        return metrics
    
    def _collect_compute_metrics(self, metrics: HardwareMetrics) -> HardwareMetrics:
        """
        收集计算能力指标
        
        Args:
            metrics: 硬件指标对象
        
        Returns:
            更新后的硬件指标
        """
        current_time = time.time()
        
        if current_time - self.last_compute_check < self.compute_check_interval:
            return metrics
        
        if not torch.cuda.is_available():
            return metrics
        
        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            test_tensor = torch.randn(1024, 1024, device=self.device)
            
            torch.cuda.synchronize()
            start_time.record()
            
            for _ in range(10):
                result = torch.matmul(test_tensor, test_tensor)
            
            end_time.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0
            flops = 2 * 1024 ** 3 * 10
            throughput = flops / elapsed_time / 1e12
            
            metrics.compute_throughput = throughput
            self.last_compute_check = current_time
            
        except Exception as e:
            logger.debug(f"计算能力检测失败: {e}")
        
        return metrics
    
    def _check_bandwidth(self, metrics: HardwareMetrics, current_time: float) -> HardwareMetrics:
        """
        检测通信带宽
        
        Args:
            metrics: 硬件指标对象
            current_time: 当前时间
        
        Returns:
            更新后的硬件指标
        """
        if not dist.is_initialized():
            return metrics
        
        if current_time - self.last_bandwidth_check < self.bandwidth_test_interval:
            return metrics
        
        if not torch.cuda.is_available():
            return metrics
        
        try:
            test_tensor = torch.randn(100 * 1024 * 1024, device=self.device)
            
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            dist.all_reduce(test_tensor)
            end_time.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0
            data_size = 2 * test_tensor.element_size() * test_tensor.numel()
            bandwidth = data_size / elapsed_time / (1024 ** 3)
            
            metrics.communication_bandwidth = bandwidth
            self.last_bandwidth_check = current_time
            
        except Exception as e:
            logger.debug(f"带宽检测失败: {e}")
        
        return metrics
    
    def _add_to_history(self, metrics: HardwareMetrics) -> None:
        """
        添加指标到历史记录
        
        Args:
            metrics: 硬件指标对象
        """
        self.metrics_history.append(metrics)
        
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
    
    def get_average_metrics(self, last_n: int = 100) -> Optional[HardwareMetrics]:
        """
        获取最近 N 个指标的平均值
        
        Args:
            last_n: 要平均的指标数量
        
        Returns:
            平均硬件指标
        """
        if not self.metrics_history:
            return None
        
        recent_metrics = self.metrics_history[-last_n:]
        
        avg_metrics = HardwareMetrics()
        count = len(recent_metrics)
        
        for metrics in recent_metrics:
            avg_metrics.gpu_memory_used += metrics.gpu_memory_used
            avg_metrics.gpu_memory_peak += metrics.gpu_memory_peak
            avg_metrics.gpu_memory_percent += metrics.gpu_memory_percent
            avg_metrics.gpu_utilization += metrics.gpu_utilization
            avg_metrics.gpu_temperature += metrics.gpu_temperature
            avg_metrics.communication_bandwidth += metrics.communication_bandwidth
            avg_metrics.compute_throughput += metrics.compute_throughput
        
        avg_metrics.gpu_memory_used //= count
        avg_metrics.gpu_memory_peak //= count
        avg_metrics.gpu_memory_percent /= count
        avg_metrics.gpu_utilization /= count
        avg_metrics.gpu_temperature /= count
        avg_metrics.communication_bandwidth /= count
        avg_metrics.compute_throughput /= count
        
        return avg_metrics
    
    def get_peak_metrics(self) -> Optional[HardwareMetrics]:
        """
        获取历史记录中的峰值指标
        
        Returns:
            峰值硬件指标
        """
        if not self.metrics_history:
            return None
        
        peak_metrics = HardwareMetrics()
        
        for metrics in self.metrics_history:
            peak_metrics.gpu_memory_peak = max(
                peak_metrics.gpu_memory_peak,
                metrics.gpu_memory_peak
            )
            peak_metrics.gpu_utilization = max(
                peak_metrics.gpu_utilization,
                metrics.gpu_utilization
            )
            peak_metrics.gpu_temperature = max(
                peak_metrics.gpu_temperature,
                metrics.gpu_temperature
            )
            peak_metrics.communication_bandwidth = max(
                peak_metrics.communication_bandwidth,
                metrics.communication_bandwidth
            )
            peak_metrics.compute_throughput = max(
                peak_metrics.compute_throughput,
                metrics.compute_throughput
            )
        
        return peak_metrics
    
    def print_metrics_summary(self) -> None:
        """
        打印指标摘要
        """
        if not self.metrics_history:
            logger.info("暂无硬件监控数据")
            return
        
        avg_metrics = self.get_average_metrics()
        peak_metrics = self.get_peak_metrics()
        
        logger.info("=" * 60)
        logger.info("硬件监控摘要")
        logger.info("=" * 60)
        
        if avg_metrics:
            logger.info(f"平均 GPU 内存使用: {avg_metrics.gpu_memory_used / (1024**2):.2f} MB")
            logger.info(f"平均 GPU 内存峰值: {avg_metrics.gpu_memory_peak / (1024**2):.2f} MB")
            logger.info(f"平均 GPU 利用率: {avg_metrics.gpu_utilization:.1f}%")
            logger.info(f"平均 GPU 温度: {avg_metrics.gpu_temperature:.1f}°C")
            logger.info(f"平均通信带宽: {avg_metrics.communication_bandwidth:.2f} GB/s")
            logger.info(f"平均计算吞吐量: {avg_metrics.compute_throughput:.2f} TFLOPS")
        
        if peak_metrics:
            logger.info(f"峰值 GPU 内存: {peak_metrics.gpu_memory_peak / (1024**2):.2f} MB")
            logger.info(f"峰值 GPU 利用率: {peak_metrics.gpu_utilization:.1f}%")
            logger.info(f"峰值 GPU 温度: {peak_metrics.gpu_temperature:.1f}°C")
            logger.info(f"峰值通信带宽: {peak_metrics.communication_bandwidth:.2f} GB/s")
            logger.info(f"峰值计算吞吐量: {peak_metrics.compute_throughput:.2f} TFLOPS")
        
        logger.info(f"监控时长: {time.time() - self.start_time:.2f} 秒")
        logger.info(f"数据点数: {len(self.metrics_history)}")
        logger.info("=" * 60)
    
    def clear_history(self) -> None:
        """
        清空历史记录
        """
        self.metrics_history.clear()
        self.start_time = time.time()
        logger.info("硬件监控历史记录已清空")


def create_hardware_monitor(
    local_rank: int = 0,
    max_history_size: int = 1000,
    bandwidth_test_interval: float = 60.0,
    compute_check_interval: float = 30.0
) -> RealTimeHardwareMonitor:
    """
    创建实时硬件监控器的便捷函数
    
    Args:
        local_rank: 本地 rank
        max_history_size: 最大历史记录数量
        bandwidth_test_interval: 带宽测试间隔（秒）
        compute_check_interval: 计算能力检查间隔（秒）
    
    Returns:
        实时硬件监控器实例
    
    Example:
        >>> monitor = create_hardware_monitor(local_rank=0)
        >>> metrics = monitor.collect_metrics()
        >>> print(metrics.to_dict())
    """
    return RealTimeHardwareMonitor(
        local_rank=local_rank,
        max_history_size=max_history_size,
        bandwidth_test_interval=bandwidth_test_interval,
        compute_check_interval=compute_check_interval
    )
