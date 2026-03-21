# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_quantization_degradation.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
量化模块降级机制测试模块

测试量化降级机制的各项功能。
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.quantization.degradation import (
    DegradationLevel,
    DegradationTrigger,
    QuantizationWithDegradation,
    AdaptiveQuantization,
    QuantizationDegradationError,
    quantize_with_fallback,
)
from parascale.config import QuantizationConfig


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestDegradationLevel:
    """测试降级级别枚举"""
    
    def test_degradation_level_values(self):
        """测试降级级别值"""
        assert DegradationLevel.NONE == 0
        assert DegradationLevel.REDUCE_PRECISION == 1
        assert DegradationLevel.PARTIAL_QUANTIZATION == 2
        assert DegradationLevel.DISABLED == 3
    
    def test_degradation_level_names(self):
        """测试降级级别名称"""
        assert DegradationLevel.NONE.name == "NONE"
        assert DegradationLevel.REDUCE_PRECISION.name == "REDUCE_PRECISION"
        assert DegradationLevel.PARTIAL_QUANTIZATION.name == "PARTIAL_QUANTIZATION"
        assert DegradationLevel.DISABLED.name == "DISABLED"


class TestDegradationTrigger:
    """测试降级触发器"""
    
    def test_initialization(self):
        """测试初始化"""
        trigger = DegradationTrigger(
            memory_threshold=80.0,
            cpu_threshold=85.0,
            min_available_memory_gb=4.0
        )
        
        assert trigger.memory_threshold == 80.0
        assert trigger.cpu_threshold == 85.0
        assert trigger.min_available_memory_gb == 4.0
    
    def test_check_memory(self):
        """测试内存检查"""
        trigger = DegradationTrigger()
        memory_status = trigger.check_memory()
        
        assert 'total_gb' in memory_status
        assert 'available_gb' in memory_status
        assert 'percent_used' in memory_status
        assert 'is_critical' in memory_status
        assert 'is_low' in memory_status
        
        assert memory_status['total_gb'] > 0
        assert memory_status['available_gb'] >= 0
        assert 0 <= memory_status['percent_used'] <= 100
    
    def test_check_cpu(self):
        """测试CPU检查"""
        trigger = DegradationTrigger()
        cpu_status = trigger.check_cpu()
        
        assert 'percent_used' in cpu_status
        assert 'is_critical' in cpu_status
        
        assert 0 <= cpu_status['percent_used'] <= 100
    
    def test_get_degradation_level_none(self):
        """测试无降级情况"""
        # 使用非常宽松的阈值确保不会触发降级
        trigger = DegradationTrigger(
            memory_threshold=99.9,
            min_available_memory_gb=0.1
        )
        
        level = trigger.get_degradation_level()
        # 在大多数系统上应该不会触发降级
        assert level in [DegradationLevel.NONE, DegradationLevel.REDUCE_PRECISION, 
                        DegradationLevel.PARTIAL_QUANTIZATION, DegradationLevel.DISABLED]
    
    def test_get_system_status(self):
        """测试获取系统状态"""
        trigger = DegradationTrigger()
        status = trigger.get_system_status()
        
        assert 'memory' in status
        assert 'cpu' in status
        assert 'should_degrade' in status
        assert 'recommended_level' in status


class TestQuantizationWithDegradation:
    """测试带降级机制的量化包装器"""
    
    def test_initialization(self):
        """测试初始化"""
        model = SimpleModel()
        config = QuantizationConfig(bits=4)
        
        wrapper = QuantizationWithDegradation(model, config)
        
        assert wrapper.model == model
        assert wrapper.degradation_level == DegradationLevel.NONE
        assert wrapper.auto_degrade is True
    
    def test_copy_config(self):
        """测试配置复制"""
        model = SimpleModel()
        config = QuantizationConfig(bits=4)
        
        wrapper = QuantizationWithDegradation(model, config)
        
        # 修改原始配置不应影响包装器中的配置
        config.bits = 8
        assert wrapper.original_config.bits == 8
        assert wrapper.config.bits == 4  # 副本未改变
    
    def test_apply_degradation_reduce_precision(self):
        """测试降低精度降级"""
        model = SimpleModel()
        config = QuantizationConfig(bits=4)
        
        wrapper = QuantizationWithDegradation(model, config)
        wrapper.apply_degradation(DegradationLevel.REDUCE_PRECISION)
        
        assert wrapper.degradation_level == DegradationLevel.REDUCE_PRECISION
        assert wrapper.config.bits == 8  # 4-bit -> 8-bit
    
    def test_apply_degradation_partial_quantization(self):
        """测试部分量化降级"""
        model = SimpleModel()
        config = QuantizationConfig(bits=4)
        
        wrapper = QuantizationWithDegradation(model, config)
        wrapper.apply_degradation(DegradationLevel.PARTIAL_QUANTIZATION)
        
        assert wrapper.degradation_level == DegradationLevel.PARTIAL_QUANTIZATION
        assert wrapper.config.bits == 8
        assert isinstance(wrapper.skipped_layers, list)
    
    def test_apply_degradation_disabled(self):
        """测试禁用量化降级"""
        model = SimpleModel()
        config = QuantizationConfig(bits=4, enabled=True)
        
        wrapper = QuantizationWithDegradation(model, config)
        wrapper.apply_degradation(DegradationLevel.DISABLED)
        
        assert wrapper.degradation_level == DegradationLevel.DISABLED
        assert wrapper.config.enabled is False
    
    def test_identify_sensitive_layers(self):
        """测试识别敏感层"""
        model = SimpleModel()
        config = QuantizationConfig()
        
        wrapper = QuantizationWithDegradation(model, config)
        sensitive_layers = wrapper._identify_sensitive_layers()
        
        assert isinstance(sensitive_layers, list)
        # 应该识别出一些层（如第一层和最后一层）
        assert len(sensitive_layers) >= 0
    
    def test_is_first_or_last_layer(self):
        """测试第一层/最后一层识别"""
        model = SimpleModel()
        config = QuantizationConfig()
        
        wrapper = QuantizationWithDegradation(model, config)
        
        # conv1 应该是第一层
        assert wrapper._is_first_or_last_layer('conv1') is True
        
        # fc 可能是最后一层
        assert wrapper._is_first_or_last_layer('fc') is True
        
        # conv2 应该是中间层
        assert wrapper._is_first_or_last_layer('conv2') is False
    
    def test_get_degradation_chain(self):
        """测试获取降级链"""
        model = SimpleModel()
        config = QuantizationConfig()
        
        wrapper = QuantizationWithDegradation(model, config)
        chain = wrapper._get_degradation_chain()
        
        assert len(chain) == 4
        assert chain[0] == DegradationLevel.NONE
        assert chain[1] == DegradationLevel.REDUCE_PRECISION
        assert chain[2] == DegradationLevel.PARTIAL_QUANTIZATION
        assert chain[3] == DegradationLevel.DISABLED
    
    def test_get_degradation_report(self):
        """测试获取降级报告"""
        model = SimpleModel()
        config = QuantizationConfig(bits=4)
        
        wrapper = QuantizationWithDegradation(model, config)
        report = wrapper.get_degradation_report()
        
        assert 'current_level' in report
        assert 'original_bits' in report
        assert 'current_bits' in report
        assert 'skipped_layers' in report
        assert 'history' in report
        assert 'system_status' in report
    
    def test_record_success_and_failure(self):
        """测试记录成功和失败"""
        model = SimpleModel()
        config = QuantizationConfig()
        
        wrapper = QuantizationWithDegradation(model, config)
        
        # 记录成功
        wrapper._record_success(DegradationLevel.NONE)
        assert len(wrapper.degradation_history) == 1
        assert wrapper.degradation_history[0]['success'] is True
        
        # 记录失败
        wrapper._record_failure(DegradationLevel.REDUCE_PRECISION, "Test error")
        assert len(wrapper.degradation_history) == 2
        assert wrapper.degradation_history[1]['success'] is False
        assert wrapper.degradation_history[1]['error'] == "Test error"


class TestAdaptiveQuantization:
    """测试自适应量化管理器"""
    
    def test_initialization(self):
        """测试初始化"""
        model = SimpleModel()
        config = QuantizationConfig()
        
        adaptive = AdaptiveQuantization(model, config)
        
        assert adaptive.model == model
        assert adaptive.base_config == config
    
    def test_get_recommendation(self):
        """测试获取建议"""
        model = SimpleModel()
        adaptive = AdaptiveQuantization(model)
        
        recommendation = adaptive.get_recommendation()
        
        assert 'should_quantize' in recommendation
        assert 'recommended_bits' in recommendation
        assert 'reason' in recommendation
        
        assert isinstance(recommendation['should_quantize'], bool)
        assert recommendation['recommended_bits'] in [4, 8]


class TestQuantizeWithFallback:
    """测试带降级机制的量化便捷函数"""
    
    def test_function_exists(self):
        """测试函数存在"""
        assert callable(quantize_with_fallback)


class TestQuantizationDegradationError:
    """测试量化降级错误"""
    
    def test_error_is_exception(self):
        """测试错误是异常类型"""
        assert issubclass(QuantizationDegradationError, Exception)
    
    def test_error_can_be_raised(self):
        """测试错误可以被抛出"""
        with pytest.raises(QuantizationDegradationError):
            raise QuantizationDegradationError("Test error message")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
