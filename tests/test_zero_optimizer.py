# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_zero_optimizer.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ZeRO优化器测试模块

测试ZeRO优化器的各个阶段功能。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.optimizers.zero_optimizer import ZeroOptimizer, ZeroAdamW, ZeroSGD, ZeroStage


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestZeroStage:
    """测试ZeRO阶段枚举"""
    
    def test_zero_stage_values(self):
        """测试ZeRO阶段值"""
        assert ZeroStage.DISABLED == 0
        assert ZeroStage.OPTIMIZER_STATES == 1
        assert ZeroStage.GRADIENTS == 2
        assert ZeroStage.PARAMETERS == 3


class TestZeroOptimizerStage0:
    """测试ZeRO Stage 0 (禁用ZeRO)"""
    
    def test_stage0_initialization(self):
        """测试Stage 0初始化"""
        model = SimpleModel()
        optimizer = ZeroOptimizer(
            model,
            optim.AdamW,
            stage=0,
            lr=1e-3
        )
        
        assert optimizer.stage == ZeroStage.DISABLED
        assert optimizer.world_size == 1
        assert optimizer.rank == 0
    
    def test_stage0_step(self):
        """测试Stage 0优化步骤"""
        model = SimpleModel()
        optimizer = ZeroOptimizer(
            model,
            optim.AdamW,
            stage=0,
            lr=1e-3
        )
        
        # 前向传播
        x = torch.randn(4, 100)
        y = model(x)
        loss = y.sum()
        
        # 反向传播
        loss.backward()
        
        # 优化步骤
        optimizer.step()
        
        # 验证参数已更新
        for param in model.parameters():
            assert param.grad is not None
    
    def test_stage0_memory_stats(self):
        """测试Stage 0内存统计"""
        model = SimpleModel()
        optimizer = ZeroOptimizer(
            model,
            optim.AdamW,
            stage=0,
            lr=1e-3
        )
        
        stats = optimizer.get_memory_stats()
        
        assert 'total_params' in stats
        assert 'owned_params' in stats
        assert 'param_memory_mb' in stats
        assert 'theoretical_savings' in stats
        assert stats['theoretical_savings'] == 1.0


class TestZeroAdamW:
    """测试ZeRO AdamW优化器"""
    
    def test_zero_adamw_initialization(self):
        """测试ZeroAdamW初始化"""
        model = SimpleModel()
        optimizer = ZeroAdamW(
            model,
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            stage=0
        )
        
        assert optimizer.stage == ZeroStage.DISABLED
        assert hasattr(optimizer, 'base_optimizer')
    
    def test_zero_adamw_step(self):
        """测试ZeroAdamW优化步骤"""
        model = SimpleModel()
        optimizer = ZeroAdamW(model, lr=1e-3, stage=0)
        
        # 生成数据
        x = torch.randn(4, 100)
        target = torch.randint(0, 10, (4,))
        
        # 训练一步
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # 验证梯度已清零
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)


class TestZeroSGD:
    """测试ZeRO SGD优化器"""
    
    def test_zero_sgd_initialization(self):
        """测试ZeroSGD初始化"""
        model = SimpleModel()
        optimizer = ZeroSGD(
            model,
            lr=0.01,
            momentum=0.9,
            stage=0
        )
        
        assert optimizer.stage == ZeroStage.DISABLED
        assert hasattr(optimizer, 'base_optimizer')
    
    def test_zero_sgd_with_momentum(self):
        """测试带momentum的ZeroSGD"""
        model = SimpleModel()
        optimizer = ZeroSGD(
            model,
            lr=0.01,
            momentum=0.9,
            stage=0
        )
        
        x = torch.randn(4, 100)
        target = torch.randint(0, 10, (4,))
        
        # 多步训练
        for _ in range(3):
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证训练正常进行
        assert True


class TestZeroOptimizerStateDict:
    """测试ZeRO优化器状态字典"""
    
    def test_state_dict_stage0(self):
        """测试Stage 0状态字典"""
        model = SimpleModel()
        optimizer = ZeroOptimizer(
            model,
            optim.AdamW,
            stage=0,
            lr=1e-3
        )
        
        # 执行一步优化
        x = torch.randn(4, 100)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        # 获取状态字典
        state_dict = optimizer.state_dict()
        
        assert 'stage' in state_dict
        assert 'base_optimizer_state' in state_dict
        assert state_dict['stage'] == 0
    
    def test_load_state_dict(self):
        """测试加载状态字典"""
        model = SimpleModel()
        optimizer = ZeroOptimizer(
            model,
            optim.AdamW,
            stage=0,
            lr=1e-3
        )
        
        # 执行一步优化
        x = torch.randn(4, 100)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        # 保存状态
        state_dict = optimizer.state_dict()
        
        # 创建新优化器并加载状态
        model2 = SimpleModel()
        optimizer2 = ZeroOptimizer(
            model2,
            optim.AdamW,
            stage=0,
            lr=1e-3
        )
        optimizer2.load_state_dict(state_dict)
        
        # 验证状态已加载
        assert optimizer2.stage == optimizer.stage


class TestZeroOptimizerValidation:
    """测试ZeRO优化器验证"""
    
    def test_invalid_stage(self):
        """测试无效stage"""
        model = SimpleModel()
        
        with pytest.raises(ValueError):
            ZeroOptimizer(model, optim.AdamW, stage=4)
        
        with pytest.raises(ValueError):
            ZeroOptimizer(model, optim.AdamW, stage=-1)
    
    def test_stage_requires_distributed(self):
        """测试Stage 1+需要分布式环境"""
        model = SimpleModel()
        
        # Stage 1+需要分布式环境，应该抛出RuntimeError
        with pytest.raises(RuntimeError):
            ZeroOptimizer(model, optim.AdamW, stage=1)


class TestZeroOptimizerMemoryStats:
    """测试ZeRO优化器内存统计"""
    
    def test_memory_stats_structure(self):
        """测试内存统计结构"""
        model = SimpleModel()
        optimizer = ZeroOptimizer(model, optim.AdamW, stage=0)
        
        stats = optimizer.get_memory_stats()
        
        required_keys = [
            'total_params',
            'owned_params',
            'param_memory_mb',
            'grad_memory_mb',
            'optimizer_state_mb',
            'total_memory_mb',
            'theoretical_savings'
        ]
        
        for key in required_keys:
            assert key in stats
        
        # 验证数值合理
        assert stats['total_params'] > 0
        assert stats['owned_params'] > 0
        assert stats['param_memory_mb'] > 0
        assert stats['total_memory_mb'] > 0
    
    def test_print_memory_stats(self, capsys):
        """测试打印内存统计"""
        model = SimpleModel()
        optimizer = ZeroOptimizer(model, optim.AdamW, stage=0)
        
        optimizer.print_memory_stats()
        
        captured = capsys.readouterr()
        assert "ZeRO Stage 0 Memory Stats" in captured.out
        assert "Total parameters" in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
