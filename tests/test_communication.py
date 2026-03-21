# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_communication.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
通信优化模块测试

测试梯度压缩、通信计算重叠等功能。
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.parallel.communication import (
    TopKCompressor,
    OneBitAdamCompressor,
    CommunicationOverlap,
    OptimizedCommunicator,
)


class TestTopKCompressor:
    """测试Top-K梯度压缩器"""
    
    def test_initialization(self):
        """测试初始化"""
        compressor = TopKCompressor(compression_ratio=0.1)
        assert compressor.compression_ratio == 0.1
        assert compressor.error_feedback == True
    
    def test_invalid_compression_ratio(self):
        """测试无效压缩比例"""
        with pytest.raises(ValueError):
            TopKCompressor(compression_ratio=0)
        
        with pytest.raises(ValueError):
            TopKCompressor(compression_ratio=1.5)
    
    def test_compress_decompress(self):
        """测试压缩和解压缩"""
        compressor = TopKCompressor(compression_ratio=0.5)
        
        # 创建测试梯度
        tensor = torch.randn(100)
        
        # 压缩
        compressed, metadata = compressor.compress(tensor)
        
        # 验证压缩后大小
        assert compressed.numel() == 50  # 50%压缩
        assert 'indices' in metadata
        assert 'shape' in metadata
        
        # 解压缩
        decompressed = compressor.decompress(compressed, metadata)
        
        # 验证形状
        assert decompressed.shape == tensor.shape
    
    def test_error_feedback(self):
        """测试误差补偿"""
        compressor = TopKCompressor(compression_ratio=0.1, error_feedback=True)
        
        tensor = torch.randn(100)
        
        # 多次压缩
        for _ in range(3):
            compressed, metadata = compressor.compress(tensor, param_id=0)
        
        # 验证误差被记录
        assert 0 in compressor.residual_errors
    
    def test_get_compression_stats(self):
        """测试获取压缩统计"""
        compressor = TopKCompressor(compression_ratio=0.1)
        
        stats = compressor.get_compression_stats()
        
        assert 'compression_ratio' in stats
        assert 'error_feedback' in stats
        assert stats['compression_ratio'] == 0.1


class TestOneBitAdamCompressor:
    """测试1-bit Adam压缩器"""
    
    def test_initialization(self):
        """测试初始化"""
        compressor = OneBitAdamCompressor(error_feedback=True)
        assert compressor.error_feedback == True
    
    def test_compress_decompress(self):
        """测试压缩和解压缩"""
        compressor = OneBitAdamCompressor(error_feedback=False)
        
        tensor = torch.randn(100)
        
        # 压缩
        compressed, metadata = compressor.compress(tensor)
        
        # 验证压缩后类型
        assert compressed.dtype == torch.uint8
        assert 'scale' in metadata
        assert 'shape' in metadata
        
        # 解压缩
        decompressed = compressor.decompress(compressed, metadata)
        
        # 验证形状
        assert decompressed.shape == tensor.shape


class TestCommunicationOverlap:
    """测试通信计算重叠"""
    
    def test_initialization(self):
        """测试初始化"""
        overlap = CommunicationOverlap(overlap_enabled=False)
        assert overlap.overlap_enabled == False
    
    def test_compute_context(self):
        """测试计算上下文"""
        overlap = CommunicationOverlap(overlap_enabled=False)
        
        # 应该返回nullcontext
        ctx = overlap.compute_context()
        assert ctx is not None
    
    def test_sync_gradients_async_no_distributed(self):
        """测试非分布式环境下的异步同步"""
        overlap = CommunicationOverlap(overlap_enabled=False)
        
        # 创建测试参数
        param = nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)
        
        # 在非分布式环境下应该同步执行
        events = overlap.sync_gradients_async([param])
        assert events == []


class TestOptimizedCommunicator:
    """测试优化通信器"""
    
    def test_initialization_no_compression(self):
        """测试无压缩初始化"""
        comm = OptimizedCommunicator(
            compression_ratio=None,
            overlap_enabled=False
        )
        
        assert comm.compressor is None
        assert comm.overlap.overlap_enabled == False
    
    def test_initialization_with_compression(self):
        """测试带压缩初始化"""
        comm = OptimizedCommunicator(
            compression_ratio=0.1,
            overlap_enabled=False
        )
        
        assert comm.compressor is not None
        assert isinstance(comm.compressor, TopKCompressor)
    
    def test_synchronize_gradients_no_distributed(self):
        """测试非分布式环境下的梯度同步"""
        comm = OptimizedCommunicator()
        
        # 创建测试参数
        param = nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)
        
        # 在非分布式环境下应该返回None
        result = comm.synchronize_gradients([param])
        assert result is None
    
    def test_get_stats(self):
        """测试获取统计信息"""
        comm = OptimizedCommunicator(
            compression_ratio=0.1,
            overlap_enabled=False
        )
        
        stats = comm.get_stats()
        
        assert 'compression_enabled' in stats
        assert 'compression_stats' in stats
        assert 'overlap_enabled' in stats
        assert 'ring_allreduce_enabled' in stats
        assert 'bucket_size_mb' in stats
        
        assert stats['compression_enabled'] == True
        assert stats['overlap_enabled'] == False


class TestCompressionAccuracy:
    """测试压缩精度"""
    
    def test_topk_preserves_top_values(self):
        """测试TopK保留最大值"""
        compressor = TopKCompressor(compression_ratio=0.1, error_feedback=False)
        
        # 创建有明确最大值的张量
        tensor = torch.zeros(100)
        tensor[5] = 10.0
        tensor[10] = 5.0
        tensor[15] = 3.0
        
        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)
        
        # 验证最大值位置被保留
        assert decompressed[5] != 0
        assert decompressed[10] != 0
        assert decompressed[15] != 0
    
    def test_onebit_preserves_sign(self):
        """测试1-bit保留符号"""
        compressor = OneBitAdamCompressor(error_feedback=False)
        
        # 创建有正有负的张量
        tensor = torch.tensor([1.0, -2.0, 3.0, -4.0, 0.5])
        
        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)
        
        # 验证符号被保留
        assert (decompressed[0] > 0) == (tensor[0] > 0)
        assert (decompressed[1] > 0) == (tensor[1] > 0)
        assert (decompressed[2] > 0) == (tensor[2] > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
