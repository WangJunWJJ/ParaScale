# -*- coding: utf-8 -*-
# @Time    : 2026/3/9 上午10:30
# @Author  : Jun Wang
# @File    : test_hybrid_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 3D混合并行测试模块

本模块提供了3D混合并行（数据并行+张量并行+流水线并行）的测试用例，
包括单元测试和集成测试。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.parallel.hybrid_parallel import HybridParallel
from parascale.config import ParaScaleConfig


class Simple3DModel(nn.Module):
    """
    用于3D并行测试的简单模型
    
    包含多个线性层，适合流水线并行分割
    """
    def __init__(self, input_dim=3072, hidden_dim=512, output_dim=10, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        
        # 输出层
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


class Transformer3DModel(nn.Module):
    """
    用于3D并行测试的Transformer风格模型
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, output_dim=10):
        super().__init__()
        self.embedding = nn.Linear(3072, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class TestHybridParallel(unittest.TestCase):
    """
    3D混合并行单元测试类
    """
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.rank = int(os.environ.get('RANK', 0))
        cls.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if cls.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(cls.rank)
    
    def test_hybrid_parallel_init(self):
        """测试3D混合并行初始化"""
        if self.world_size < 8:
            self.skipTest("需要至少8个进程进行3D并行测试")
        
        model = Simple3DModel()
        
        # 创建3D并行实例: DP=2, TP=2, PP=2
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=8,
            dp_size=2,
            tp_size=2,
            pp_size=2,
            tensor_parallel_mode="row"
        )
        
        # 验证并行信息
        info = hp.get_parallel_info()
        self.assertEqual(info['dp_size'], 2)
        self.assertEqual(info['tp_size'], 2)
        self.assertEqual(info['pp_size'], 2)
        self.assertEqual(info['world_size'], 8)
        
        print(f"[Rank {self.rank}] 3D并行初始化测试通过")
        print(f"[Rank {self.rank}] 并行信息: {info}")
    
    def test_process_group_creation(self):
        """测试进程组创建"""
        if self.world_size < 8:
            self.skipTest("需要至少8个进程进行3D并行测试")
        
        model = Simple3DModel()
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=8,
            dp_size=2,
            tp_size=2,
            pp_size=2
        )
        
        # 验证进程组已创建
        self.assertIsNotNone(hp.tp_group)
        self.assertIsNotNone(hp.pp_group)
        self.assertIsNotNone(hp.dp_group)
        
        print(f"[Rank {self.rank}] 进程组创建测试通过")
    
    def test_model_partition(self):
        """测试模型分割"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Simple3DModel(num_layers=8)
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=1,
            tp_size=2,
            pp_size=2
        )
        
        # 验证模型已分割
        self.assertIsNotNone(hp.stage_layers)
        
        # 验证每个阶段都有层
        self.assertGreater(len(list(hp.stage_layers.children())), 0)
        
        print(f"[Rank {self.rank}] 模型分割测试通过")
        print(f"[Rank {self.rank}] 当前阶段层数: {len(list(hp.stage_layers.children()))}")
    
    def test_tensor_parallel_in_stage(self):
        """测试阶段内的张量并行"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Simple3DModel(hidden_dim=512)
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=1,
            tp_size=2,
            pp_size=2,
            tensor_parallel_mode="row"
        )
        
        # 验证张量并行已应用
        for name, module in hp.stage_layers.named_modules():
            if isinstance(module, nn.Linear):
                # 行并行模式下，输出维度应该被分割
                if hp.tensor_parallel_mode == "row":
                    self.assertLessEqual(module.out_features, 512)
        
        print(f"[Rank {self.rank}] 阶段内张量并行测试通过")
    
    def test_forward_pass(self):
        """测试前向传播"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Simple3DModel()
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=1,
            tp_size=2,
            pp_size=2
        )
        
        # 创建测试输入
        batch_size = 4
        inputs = torch.randn(batch_size, 3, 32, 32)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # 执行前向传播
        outputs = hp.forward(inputs)
        
        # 验证输出
        if hp.is_last_stage:
            self.assertIsNotNone(outputs)
            self.assertEqual(outputs.shape[0], batch_size)
            self.assertEqual(outputs.shape[1], 10)
        else:
            self.assertIsNone(outputs)
        
        print(f"[Rank {self.rank}] 前向传播测试通过")
    
    def test_micro_batch_forward(self):
        """测试微批次前向传播"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Simple3DModel()
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=1,
            tp_size=2,
            pp_size=2,
            pipeline_chunks=2
        )
        
        # 创建测试输入
        batch_size = 8
        inputs = torch.randn(batch_size, 3, 32, 32)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # 执行前向传播
        outputs = hp.forward(inputs)
        
        # 验证输出
        if hp.is_last_stage:
            self.assertIsNotNone(outputs)
            self.assertEqual(outputs.shape[0], batch_size)
        
        print(f"[Rank {self.rank}] 微批次前向传播测试通过")
    
    def test_gradient_gathering(self):
        """测试梯度收集"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Simple3DModel()
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=2,
            tp_size=2,
            pp_size=1
        )
        
        # 创建测试输入
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        # 前向传播
        outputs = hp.forward(inputs)
        
        # 反向传播
        if outputs is not None:
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
        
        # 收集梯度
        hp.gather_gradients()
        
        print(f"[Rank {self.rank}] 梯度收集测试通过")
    
    def test_row_parallel_mode(self):
        """测试行并行模式"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Simple3DModel(hidden_dim=512)
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=1,
            tp_size=2,
            pp_size=2,
            tensor_parallel_mode="row"
        )
        
        # 验证配置
        self.assertEqual(hp.tensor_parallel_mode, "row")
        
        # 测试前向传播
        inputs = torch.randn(2, 3, 32, 32)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        outputs = hp.forward(inputs)
        
        if hp.is_last_stage:
            self.assertIsNotNone(outputs)
        
        print(f"[Rank {self.rank}] 行并行模式测试通过")
    
    def test_column_parallel_mode(self):
        """测试列并行模式"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Simple3DModel(hidden_dim=512)
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=1,
            tp_size=2,
            pp_size=2,
            tensor_parallel_mode="column"
        )
        
        # 验证配置
        self.assertEqual(hp.tensor_parallel_mode, "column")
        
        # 测试前向传播
        inputs = torch.randn(2, 3, 32, 32)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        outputs = hp.forward(inputs)
        
        if hp.is_last_stage:
            self.assertIsNotNone(outputs)
        
        print(f"[Rank {self.rank}] 列并行模式测试通过")
    
    def test_invalid_config(self):
        """测试无效配置"""
        model = Simple3DModel()
        
        # 测试进程数不匹配
        with self.assertRaises(ValueError):
            HybridParallel(
                model=model,
                rank=self.rank,
                world_size=8,
                dp_size=2,
                tp_size=2,
                pp_size=3  # 2*2*3 = 12 != 8
            )
        
        print(f"[Rank {self.rank}] 无效配置测试通过")
    
    def test_parallel_info(self):
        """测试并行信息获取"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Simple3DModel()
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=1,
            tp_size=2,
            pp_size=2
        )
        
        info = hp.get_parallel_info()
        
        # 验证信息完整性
        required_keys = [
            'global_rank', 'world_size', 'dp_size', 'tp_size', 'pp_size',
            'dp_rank', 'tp_rank', 'pp_rank', 'is_first_stage', 'is_last_stage',
            'tensor_parallel_mode', 'pipeline_chunks'
        ]
        
        for key in required_keys:
            self.assertIn(key, info)
        
        print(f"[Rank {self.rank}] 并行信息测试通过")
        print(f"[Rank {self.rank}] 信息: {info}")


class TestHybridParallelIntegration(unittest.TestCase):
    """
    3D混合并行集成测试类
    """
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.rank = int(os.environ.get('RANK', 0))
        cls.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if cls.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(cls.rank)
    
    def test_end_to_end_training(self):
        """测试端到端训练流程"""
        if self.world_size < 8:
            self.skipTest("需要至少8个进程进行完整3D并行测试")
        
        model = Simple3DModel()
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=8,
            dp_size=2,
            tp_size=2,
            pp_size=2,
            tensor_parallel_mode="row",
            pipeline_chunks=2
        )
        
        optimizer = optim.AdamW(hp.stage_layers.parameters(), lr=1e-3)
        
        # 模拟训练步骤
        num_steps = 3
        for step in range(num_steps):
            # 创建测试数据
            batch_size = 8
            inputs = torch.randn(batch_size, 3, 32, 32)
            targets = torch.randint(0, 10, (batch_size,))
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            # 前向传播
            outputs = hp.forward(inputs)
            
            # 计算损失和反向传播
            if hp.is_last_stage and outputs is not None:
                loss = nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                
                print(f"[Rank {self.rank}] Step {step}, Loss: {loss.item():.4f}")
            
            # 收集梯度
            hp.gather_gradients()
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"[Rank {self.rank}] 端到端训练测试通过")
    
    def test_different_parallel_configs(self):
        """测试不同的并行配置组合"""
        if self.world_size < 8:
            self.skipTest("需要至少8个进程进行测试")
        
        configs = [
            {"dp_size": 4, "tp_size": 2, "pp_size": 1},  # DP+TP
            {"dp_size": 2, "tp_size": 1, "pp_size": 4},  # DP+PP
            {"dp_size": 1, "tp_size": 2, "pp_size": 4},  # TP+PP
            {"dp_size": 2, "tp_size": 2, "pp_size": 2},  # 3D
        ]
        
        for config in configs:
            model = Simple3DModel()
            
            hp = HybridParallel(
                model=model,
                rank=self.rank,
                world_size=8,
                dp_size=config["dp_size"],
                tp_size=config["tp_size"],
                pp_size=config["pp_size"]
            )
            
            # 测试前向传播
            inputs = torch.randn(4, 3, 32, 32)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            outputs = hp.forward(inputs)
            
            print(f"[Rank {self.rank}] 配置 {config} 测试通过")
    
    def test_transformer_model(self):
        """测试Transformer模型"""
        if self.world_size < 4:
            self.skipTest("需要至少4个进程进行测试")
        
        model = Transformer3DModel()
        
        hp = HybridParallel(
            model=model,
            rank=self.rank,
            world_size=4,
            dp_size=1,
            tp_size=2,
            pp_size=2
        )
        
        # 测试前向传播
        inputs = torch.randn(2, 3, 32, 32)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        outputs = hp.forward(inputs)
        
        if hp.is_last_stage:
            self.assertIsNotNone(outputs)
        
        print(f"[Rank {self.rank}] Transformer模型测试通过")


def run_single_process_tests():
    """运行单进程测试"""
    print("=" * 60)
    print("运行3D混合并行单进程测试")
    print("=" * 60)
    
    # 设置单进程环境
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # 测试无效配置
    print("\n测试无效配置...")
    model = Simple3DModel()
    
    try:
        hp = HybridParallel(
            model=model,
            rank=0,
            world_size=8,
            dp_size=2,
            tp_size=2,
            pp_size=3  # 2*2*3 = 12 != 8
        )
        print("错误：应该抛出ValueError")
    except ValueError as e:
        print(f"正确捕获错误: {e}")
    
    print("\n单进程测试完成")


def run_multi_process_tests():
    """运行多进程测试"""
    print("=" * 60)
    print("运行3D混合并行多进程测试")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试用例
    suite.addTests(loader.loadTestsFromTestCase(TestHybridParallel))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridParallelIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """主函数"""
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size == 1:
        # 单进程测试
        run_single_process_tests()
    else:
        # 多进程测试
        success = run_multi_process_tests()
        
        # 清理
        if dist.is_initialized():
            dist.destroy_process_group()
        
        return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
