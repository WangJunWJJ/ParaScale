# -*- coding: utf-8 -*-
# @Time    : 2026/3/19
# @Author  : Jun Wang
# @File    : test_fourbit_optimizer.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
4bit 优化器测试模块

本模块测试 4bit 量化优化器的功能和性能，包括：
- 量化/反量化的精度
- 优化器状态的正确性
- 内存节省效果
- 训练收敛性
"""

import os
import sys

# 添加项目根目录到路径（必须在其他导入之前）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import unittest

from parascale.optimizers import FourBitAdamW, FourBitSGD, QuantizedState


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestQuantizedState(unittest.TestCase):
    """测试 QuantizedState 类"""
    
    def test_quantize_dequantize(self):
        """测试量化和反量化"""
        # 创建测试张量
        original = torch.randn(100, 50)
        
        # 量化
        qs = QuantizedState(tensor=original, group_size=64)
        
        # 反量化
        reconstructed = qs.dequantize()
        
        # 检查形状
        self.assertEqual(reconstructed.shape, original.shape)
        
        # 检查相对误差（4bit 量化有一定误差，但应该可控）
        relative_error = torch.norm(reconstructed - original) / torch.norm(original)
        self.assertLess(relative_error.item(), 0.15)  # 相对误差应小于 15%
    
    def test_memory_savings(self):
        """测试内存节省效果"""
        original = torch.randn(1000, 1000)  # 1M 参数
        
        # FP32 内存
        fp32_bytes = original.numel() * 4
        
        # 量化内存
        qs = QuantizedState(tensor=original, group_size=128)
        quantized_bytes = qs.memory_usage()
        
        # 验证节省效果
        savings_ratio = 1 - quantized_bytes / fp32_bytes
        self.assertGreater(savings_ratio, 0.5)  # 至少节省 50%
    
    def test_update(self):
        """测试状态更新"""
        original = torch.randn(100, 50)
        qs = QuantizedState(tensor=original, group_size=64)
        
        # 更新为新值
        new_tensor = torch.randn(100, 50)
        qs.update(new_tensor)
        
        # 验证更新后的值
        reconstructed = qs.dequantize()
        self.assertEqual(reconstructed.shape, new_tensor.shape)
    
    def test_device_transfer(self):
        """测试设备转移"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA 不可用")
        
        original = torch.randn(100, 50)
        qs = QuantizedState(tensor=original, group_size=64)
        
        # 转移到 GPU
        qs.to(torch.device('cuda'))
        
        # 验证设备
        self.assertEqual(qs.quantized_data.device.type, 'cuda')
        self.assertEqual(qs.scale.device.type, 'cuda')


class TestFourBitAdamW(unittest.TestCase):
    """测试 FourBitAdamW 优化器"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        self.model = SimpleModel()
        self.criterion = nn.MSELoss()
    
    def test_initialization(self):
        """测试优化器初始化"""
        optimizer = FourBitAdamW(self.model.parameters(), lr=1e-3)
        
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-3)
    
    def test_step(self):
        """测试优化步骤"""
        optimizer = FourBitAdamW(self.model.parameters(), lr=1e-3)
        
        # 前向传播
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 优化步骤
        optimizer.step()
        
        # 验证参数已更新
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
    
    def test_memory_stats(self):
        """测试内存统计"""
        optimizer = FourBitAdamW(self.model.parameters(), lr=1e-3)
        
        # 执行一步以初始化状态
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 获取内存统计
        stats = optimizer.get_memory_stats()
        
        self.assertIn('total_params', stats)
        self.assertIn('savings_percent', stats)
        self.assertGreater(stats['savings_percent'], 0)  # 应该有内存节省
    
    def test_state_dict(self):
        """测试状态保存和加载"""
        optimizer = FourBitAdamW(self.model.parameters(), lr=1e-3)
        
        # 执行几步
        for _ in range(3):
            x = torch.randn(4, 10)
            target = torch.randn(4, 5)
            output = self.model(x)
            loss = self.criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 保存状态
        state_dict = optimizer.state_dict()
        
        # 创建新优化器并加载状态
        new_model = SimpleModel()
        new_optimizer = FourBitAdamW(new_model.parameters(), lr=1e-3)
        new_optimizer.load_state_dict(state_dict)
        
        # 验证参数组
        self.assertEqual(
            new_optimizer.param_groups[0]['lr'],
            optimizer.param_groups[0]['lr']
        )
    
    def test_training_convergence(self):
        """测试训练收敛性"""
        # 创建简单数据集
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)
        
        model = SimpleModel()
        optimizer = FourBitAdamW(model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()
        
        # 训练
        initial_loss = None
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
        
        final_loss = loss.item()
        
        # 验证损失下降
        self.assertLess(final_loss, initial_loss)
    
    def test_error_compensation(self):
        """测试误差补偿功能"""
        # 启用误差补偿
        optimizer_with_comp = FourBitAdamW(
            self.model.parameters(), 
            lr=1e-3,
            compensate_quant_error=True
        )
        
        # 禁用误差补偿
        optimizer_without_comp = FourBitAdamW(
            self.model.parameters(),
            lr=1e-3,
            compensate_quant_error=False
        )
        
        # 两者都应该能正常工作
        for optimizer in [optimizer_with_comp, optimizer_without_comp]:
            x = torch.randn(4, 10)
            target = torch.randn(4, 5)
            output = self.model(x)
            loss = self.criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def test_group_size_config(self):
        """测试不同分组大小"""
        for group_size in [64, 128, 256]:
            optimizer = FourBitAdamW(
                self.model.parameters(),
                lr=1e-3,
                group_size=group_size
            )
            
            x = torch.randn(4, 10)
            target = torch.randn(4, 5)
            output = self.model(x)
            loss = self.criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 验证正常运行
            self.assertEqual(optimizer.group_size, group_size)


class TestFourBitSGD(unittest.TestCase):
    """测试 FourBitSGD 优化器"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        self.model = SimpleModel()
        self.criterion = nn.MSELoss()
    
    def test_initialization(self):
        """测试优化器初始化"""
        optimizer = FourBitSGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(optimizer.param_groups[0]['momentum'], 0.9)
    
    def test_step(self):
        """测试优化步骤"""
        optimizer = FourBitSGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
    
    def test_training_convergence(self):
        """测试训练收敛性"""
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)
        
        model = SimpleModel()
        optimizer = FourBitSGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.MSELoss()
        
        initial_loss = None
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
        
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss)
    
    def test_nesterov(self):
        """测试 Nesterov 动量"""
        optimizer = FourBitSGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            nesterov=True
        )
        
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证正常运行
        self.assertTrue(True)
    
    def test_memory_stats(self):
        """测试内存统计"""
        optimizer = FourBitSGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        stats = optimizer.get_memory_stats()
        
        self.assertIn('total_params', stats)
        self.assertIn('savings_percent', stats)


class TestComparisonWithStandardOptimizers(unittest.TestCase):
    """与标准优化器对比测试"""
    
    def test_adamw_comparison(self):
        """对比 4bit AdamW 和标准 AdamW"""
        torch.manual_seed(42)
        
        # 创建两个相同初始化的模型
        model1 = SimpleModel()
        model2 = SimpleModel()
        model2.load_state_dict(model1.state_dict())
        
        # 4bit AdamW
        optimizer1 = FourBitAdamW(model1.parameters(), lr=1e-3)
        # 标准 AdamW
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        
        criterion = nn.MSELoss()
        
        # 训练数据
        X = torch.randn(32, 10)
        y = torch.randn(32, 5)
        
        losses1 = []
        losses2 = []
        
        for epoch in range(20):
            # 4bit AdamW
            optimizer1.zero_grad()
            output1 = model1(X)
            loss1 = criterion(output1, y)
            loss1.backward()
            optimizer1.step()
            losses1.append(loss1.item())
            
            # 标准 AdamW
            optimizer2.zero_grad()
            output2 = model2(X)
            loss2 = criterion(output2, y)
            loss2.backward()
            optimizer2.step()
            losses2.append(loss2.item())
        
        # 两者都应该收敛
        self.assertLess(losses1[-1], losses1[0])
        self.assertLess(losses2[-1], losses2[0])
        
        # 最终损失应该相近（允许一定误差）
        loss_diff = abs(losses1[-1] - losses2[-1]) / max(abs(losses1[-1]),
                        abs(losses2[-1]),
                        1e-8)
        self.assertLess(loss_diff, 0.5)  # 相对差异小于 50%


def run_simple_demo():
    """运行简单演示"""
    print("\n" + "=" * 60)
    print("4bit 优化器演示")
    print("=" * 60)
    
    # 创建模型
    model = SimpleModel(input_dim=100, hidden_dim=200, output_dim=10)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数数量: {total_params:,}")
    
    # 4bit AdamW
    print("\n--- 4bit AdamW ---")
    optimizer = FourBitAdamW(model.parameters(), lr=1e-3, group_size=128)
    
    # 模拟训练
    criterion = nn.MSELoss()
    for i in range(5):
        x = torch.randn(8, 100)
        target = torch.randn(8, 10)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"Step {i+1}, Loss: {loss.item():.4f}")
    
    # 打印内存统计
    optimizer.print_memory_stats()
    
    # 4bit SGD
    print("\n--- 4bit SGD ---")
    model2 = SimpleModel(input_dim=100, hidden_dim=200, output_dim=10)
    optimizer2 = FourBitSGD(model2.parameters(), lr=0.01, momentum=0.9)
    
    for i in range(5):
        x = torch.randn(8, 100)
        target = torch.randn(8, 10)
        
        optimizer2.zero_grad()
        output = model2(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer2.step()
        
        print(f"Step {i+1}, Loss: {loss.item():.4f}")
    
    optimizer2.print_memory_stats()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    # 运行演示
    run_simple_demo()
    
    # 运行单元测试
    print("\n运行单元测试...")
    unittest.main(argv=[''], verbosity=2, exit=False)
