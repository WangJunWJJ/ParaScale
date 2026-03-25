# -*- coding: utf-8 -*-
# @Time    : 2026/3/12
# @Author  : Jun Wang
# @File    : test_para_engine.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaEngine 测试模块

本模块包含 ParaEngine 的单元测试，测试自动并行策略选择功能。
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.engine.para_engine import (
    ParaEngine,
    ModelAnalyzer,
    HardwareMonitor,
    StrategyDecider,
    ModelProfile,
    HardwareProfile,
    ParallelStrategy
)
from parascale.config import ParaScaleConfig


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, input_size=100, hidden_size=256, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MediumModel(nn.Module):
    """中等规模测试模型"""
    def __init__(self, input_size=512, hidden_size=1024, num_layers=8, output_size=100):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class LargeModel(nn.Module):
    """大规模测试模型（模拟）"""
    def __init__(self, input_size=1024, hidden_size=4096, num_layers=12, output_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(50000, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class TestModelAnalyzer(unittest.TestCase):
    """测试模型分析器"""
    
    def test_analyze_simple_model(self):
        """测试分析简单模型"""
        model = SimpleModel()
        analyzer = ModelAnalyzer(model)
        profile = analyzer.analyze()
        
        self.assertGreater(profile.total_params, 0)
        self.assertGreater(profile.total_memory, 0)
        self.assertGreater(profile.num_layers, 0)
        self.assertIn('Linear', profile.layer_types)
        self.assertEqual(profile.model_type, 'mlp')
    
    def test_analyze_medium_model(self):
        """测试分析中等模型"""
        model = MediumModel()
        analyzer = ModelAnalyzer(model)
        profile = analyzer.analyze()
        
        self.assertGreater(profile.total_params, 100000)
        self.assertEqual(profile.model_type, 'mlp')
    
    def test_model_profile_to_dict(self):
        """测试模型配置文件转字典"""
        profile = ModelProfile(
            total_params=1000000,
            total_memory=4000000,
            num_layers=10,
            model_type='transformer'
        )

        profile_dict = profile.to_dict()
        self.assertIn('total_params', profile_dict)
        self.assertIn('total_memory_gb', profile_dict)
        self.assertEqual(profile_dict['total_params'], 1000000)


class TestHardwareMonitor(unittest.TestCase):
    """测试硬件监控器"""
    
    def test_monitor_without_cuda(self):
        """测试在没有 CUDA 的情况下监控硬件"""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = HardwareMonitor()
            profile = monitor.monitor()
            
            self.assertGreaterEqual(profile.num_gpus, 1)
            self.assertGreater(profile.gpu_memory, 0)
    
    def test_hardware_profile_to_dict(self):
        """测试硬件配置文件转字典"""
        profile = HardwareProfile(
            num_gpus=8,
            gpu_memory=32 * 1024 ** 3,
            gpu_compute_capability=8.0
        )
        
        profile_dict = profile.to_dict()
        self.assertIn('num_gpus', profile_dict)
        self.assertIn('gpu_memory_gb', profile_dict)
        self.assertEqual(profile_dict['num_gpus'], 8)


class TestStrategyDecider(unittest.TestCase):
    """测试策略决策器"""
    
    def test_single_gpu_strategy(self):
        """测试单 GPU 策略"""
        model_profile = ModelProfile(total_params=1000000)
        hardware_profile = HardwareProfile(num_gpus=1)
        
        decider = StrategyDecider(model_profile, hardware_profile)
        strategy = decider.decide()
        
        self.assertEqual(strategy.dp_size, 1)
        self.assertEqual(strategy.tp_size, 1)
        self.assertEqual(strategy.pp_size, 1)
        self.assertEqual(strategy.strategy_type, 'single')
    
    def test_small_model_strategy(self):
        """测试小模型策略"""
        model_profile = ModelProfile(total_params=100 * 1024 ** 2)
        hardware_profile = HardwareProfile(num_gpus=8)
        
        decider = StrategyDecider(model_profile, hardware_profile)
        strategy = decider.decide()
        
        self.assertEqual(strategy.dp_size, 8)
        self.assertEqual(strategy.tp_size, 1)
        self.assertEqual(strategy.pp_size, 1)
        self.assertEqual(strategy.strategy_type, 'data')
    
    def test_medium_model_strategy(self):
        """测试中等模型策略"""
        model_profile = ModelProfile(total_params=2 * 1024 ** 3)
        hardware_profile = HardwareProfile(
            num_gpus=8,
            available_memory=16 * 1024 ** 3
        )
        
        decider = StrategyDecider(model_profile, hardware_profile)
        strategy = decider.decide()
        
        self.assertGreaterEqual(strategy.dp_size, 1)
        self.assertGreaterEqual(strategy.tp_size, 1)
        self.assertGreaterEqual(strategy.pp_size, 1)
        self.assertTrue(strategy.validate(8))
    
    def test_large_model_strategy(self):
        """测试大模型策略"""
        model_profile = ModelProfile(total_params=15 * 1024 ** 3)
        hardware_profile = HardwareProfile(
            num_gpus=16,
            available_memory=32 * 1024 ** 3
        )
        
        decider = StrategyDecider(model_profile, hardware_profile)
        strategy = decider.decide()
        
        self.assertGreaterEqual(strategy.dp_size, 1)
        self.assertGreaterEqual(strategy.tp_size, 1)
        self.assertGreaterEqual(strategy.pp_size, 1)
        self.assertTrue(strategy.validate(16))
        self.assertEqual(strategy.strategy_type, 'hybrid')
    
    def test_parallel_strategy_validate(self):
        """测试并行策略验证"""
        strategy = ParallelStrategy(dp_size=2, tp_size=2, pp_size=2)
        self.assertTrue(strategy.validate(8))
        self.assertFalse(strategy.validate(4))


class TestParaEngine(unittest.TestCase):
    """测试 ParaEngine"""
    
    def setUp(self):
        """测试前准备"""
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        input_data = torch.randn(100, 100)
        targets = torch.randint(0, 10, (100,))
        dataset = TensorDataset(input_data, targets)
        self.dataloader = DataLoader(dataset, batch_size=10)
    
    @patch('parascale.engine.para_engine.dist.is_initialized', return_value=False)
    @patch('parascale.engine.para_engine.get_rank', return_value=0)
    @patch('parascale.engine.para_engine.get_world_size', return_value=1)
    @patch('parascale.engine.para_engine.get_local_rank', return_value=0)
    def test_init_auto_parallel(self, mock_local_rank, mock_world_size, mock_rank, mock_init):
        """测试自动并行模式初始化"""
        engine = ParaEngine(
            self.model,
            self.optimizer,
            auto_parallel=True,
            auto_init_distributed=False
        )
        
        self.assertIsNotNone(engine.parallel_strategy)
        self.assertEqual(engine.parallel_strategy.strategy_type, 'single')
    
    @patch('parascale.engine.para_engine.dist.is_initialized', return_value=False)
    @patch('parascale.engine.para_engine.get_rank', return_value=0)
    @patch('parascale.engine.para_engine.get_world_size', return_value=1)
    @patch('parascale.engine.para_engine.get_local_rank', return_value=0)
    def test_init_manual_config(self, mock_local_rank, mock_world_size, mock_rank, mock_init):
        """测试手动配置模式初始化"""
        config = ParaScaleConfig(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
        
        engine = ParaEngine(
            self.model,
            self.optimizer,
            config=config,
            auto_parallel=False,
            auto_init_distributed=False
        )
        
        self.assertIsNotNone(engine.hybrid_parallel)
    
    @patch('parascale.engine.para_engine.dist.is_initialized', return_value=False)
    @patch('parascale.engine.para_engine.get_rank', return_value=0)
    @patch('parascale.engine.para_engine.get_world_size', return_value=1)
    @patch('parascale.engine.para_engine.get_local_rank', return_value=0)
    def test_train_single_step(self, mock_local_rank, mock_world_size, mock_rank, mock_init):
        """测试单步训练"""
        engine = ParaEngine(
            self.model,
            self.optimizer,
            auto_parallel=True,
            auto_init_distributed=False
        )
        
        engine.train(self.dataloader, epochs=1)
        
        self.assertGreaterEqual(engine.global_step, 0)
    
    @patch('parascale.engine.para_engine.dist.is_initialized', return_value=False)
    @patch('parascale.engine.para_engine.get_rank', return_value=0)
    @patch('parascale.engine.para_engine.get_world_size', return_value=1)
    @patch('parascale.engine.para_engine.get_local_rank', return_value=0)
    def test_evaluate(self, mock_local_rank, mock_world_size, mock_rank, mock_init):
        """测试评估"""
        engine = ParaEngine(
            self.model,
            self.optimizer,
            auto_parallel=True,
            auto_init_distributed=False
        )
        
        loss, accuracy = engine.evaluate(self.dataloader)
        
        self.assertGreaterEqual(loss, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 100)
    
    @patch('parascale.engine.para_engine.dist.is_initialized', return_value=False)
    @patch('parascale.engine.para_engine.get_rank', return_value=0)
    @patch('parascale.engine.para_engine.get_world_size', return_value=1)
    @patch('parascale.engine.para_engine.get_local_rank', return_value=0)
    def test_get_parallel_info(self, mock_local_rank, mock_world_size, mock_rank, mock_init):
        """测试获取并行信息"""
        engine = ParaEngine(
            self.model,
            self.optimizer,
            auto_parallel=True,
            auto_init_distributed=False
        )
        
        parallel_info = engine.get_parallel_info()
        
        self.assertIsInstance(parallel_info, dict)
    
    @patch('parascale.engine.para_engine.dist.is_initialized', return_value=False)
    @patch('parascale.engine.para_engine.get_rank', return_value=0)
    @patch('parascale.engine.para_engine.get_world_size', return_value=1)
    @patch('parascale.engine.para_engine.get_local_rank', return_value=0)
    def test_get_strategy(self, mock_local_rank, mock_world_size, mock_rank, mock_init):
        """测试获取策略"""
        engine = ParaEngine(
            self.model,
            self.optimizer,
            auto_parallel=True,
            auto_init_distributed=False
        )
        
        strategy = engine.get_strategy()
        
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, ParallelStrategy)


class TestAutoParallelScenarios(unittest.TestCase):
    """测试自动并行场景"""
    
    @patch('parascale.engine.para_engine.dist.is_initialized', return_value=False)
    @patch('parascale.engine.para_engine.get_rank', return_value=0)
    @patch('parascale.engine.para_engine.get_world_size', return_value=1)
    @patch('parascale.engine.para_engine.get_local_rank', return_value=0)
    def test_small_model_auto_config(self, mock_local_rank, mock_world_size, mock_rank, mock_init):
        """测试小模型自动配置"""
        model = SimpleModel()
        engine = ParaEngine(
            model,
            auto_parallel=True,
            auto_init_distributed=False
        )
        
        strategy = engine.get_strategy()
        self.assertEqual(strategy.strategy_type, 'single')
    
    @patch('parascale.engine.para_engine.dist.is_initialized', return_value=False)
    @patch('parascale.engine.para_engine.get_rank', return_value=0)
    @patch('parascale.engine.para_engine.get_world_size', return_value=8)
    @patch('parascale.engine.para_engine.get_local_rank', return_value=0)
    def test_medium_model_auto_config(self, mock_local_rank, mock_world_size, mock_rank, mock_init):
        """测试中等模型自动配置（模拟 8 GPU）"""
        model = MediumModel()

        # 手动设置硬件配置为 8 GPU
        from parascale.engine.para_engine import HardwareProfile
        hardware_profile = HardwareProfile(num_gpus=8, gpus_per_node=8)

        with patch('parascale.engine.para_engine.HardwareMonitor') as mock_hw_monitor:
            mock_hw_monitor.return_value.monitor.return_value = hardware_profile
            engine = ParaEngine(
                model,
                auto_parallel=True,
                auto_init_distributed=False
            )

            strategy = engine.get_strategy()
            self.assertTrue(strategy.validate(8))
            self.assertIn(strategy.strategy_type, ['data', 'hybrid'])


if __name__ == '__main__':
    unittest.main()
