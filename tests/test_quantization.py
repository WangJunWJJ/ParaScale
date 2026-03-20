# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : test_quantization.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 量化训练测试模块

本模块测试量化感知训练（QAT）的功能，
包括伪量化层、观察器和 QAT 流程。
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.quantization import (
    QuantizationConfig,
    QuantizationAwareTraining,
    FakeQuantize,
    FakeQuantizedLinear,
    MinMaxObserver,
    MovingAverageObserver,
    quantize_tensor,
    dequantize_tensor,
    calculate_scale_zero_point,
    get_quantizable_layers,
)


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_quantization_config():
    """测试量化配置"""
    print("=" * 60)
    print("测试量化配置")
    print("=" * 60)
    
    # 默认配置
    config = QuantizationConfig()
    assert config.enabled == False
    assert config.bits == 8
    assert config.scheme == "symmetric"
    print("✓ 默认配置创建成功")
    
    # 自定义配置
    config = QuantizationConfig(
        enabled=True,
        bits=8,
        scheme="asymmetric",
        per_channel=True,
        observer_type="moving_average"
    )
    assert config.enabled == True
    assert config.bits == 8
    assert config.scheme == "asymmetric"
    print("✓ 自定义配置创建成功")
    
    # 测试 qmin/qmax
    qmin, qmax = config.get_qmin_qmax()
    assert qmin == -128
    assert qmax == 127
    print(f"✓ INT8 量化范围: [{qmin}, {qmax}]")
    
    # 测试无效配置
    try:
        bad_config = QuantizationConfig(bits=16)
        print("✗ 应该抛出 ValueError")
        return False
    except ValueError as e:
        print(f"✓ 无效配置检测: {e}")
    
    print("量化配置测试通过\n")
    return True


def test_observer():
    """测试观察器"""
    print("=" * 60)
    print("测试观察器")
    print("=" * 60)
    
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")
    
    # 测试 MinMaxObserver
    observer = MinMaxObserver(config)
    
    # 模拟数据
    x1 = torch.randn(32, 3, 224, 224) * 2  # 范围约 [-4, 4]
    x2 = torch.randn(32, 3, 224, 224) * 3  # 范围约 [-6, 6]
    
    observer.update(x1)
    observer.update(x2)
    
    scale, zero_point = observer.calculate_qparams()
    scale_val = scale.item() if scale.numel() == 1 else scale.mean().item()
    zp_val = zero_point.item() if zero_point.numel() == 1 else zero_point.mean().item()
    print(f"✓ MinMaxObserver: scale={scale_val:.4f}, zero_point={zp_val:.4f}")
    
    # 测试 MovingAverageObserver
    observer = MovingAverageObserver(config)
    
    for i in range(10):
        x = torch.randn(32, 3, 224, 224)
        observer.update(x)
    
    scale, zero_point = observer.calculate_qparams()
    scale_val = scale.item() if scale.numel() == 1 else scale.mean().item()
    zp_val = zero_point.item() if zero_point.numel() == 1 else zero_point.mean().item()
    print(f"✓ MovingAverageObserver: scale={scale_val:.4f}, zero_point={zp_val:.4f}")
    
    print("观察器测试通过\n")
    return True


def test_fake_quantize():
    """测试伪量化层"""
    print("=" * 60)
    print("测试伪量化层")
    print("=" * 60)
    
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")
    
    # 创建伪量化层
    fake_quant = FakeQuantize(config)
    
    # 测试数据
    x = torch.randn(32, 3, 224, 224)
    
    # 前向传播（训练模式）
    fake_quant.train()
    y = fake_quant(x)
    
    # 检查输出形状
    assert y.shape == x.shape
    print(f"✓ 输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 检查量化误差
    error = torch.abs(y - x).mean()
    print(f"✓ 平均量化误差: {error:.6f}")
    
    # 测试禁用伪量化
    fake_quant.enable_fake_quant(False)
    y_no_quant = fake_quant(x)
    assert torch.allclose(y_no_quant, x)
    print("✓ 禁用伪量化后输出与输入相同")
    
    print("伪量化层测试通过\n")
    return True


def test_fake_quantized_linear():
    """测试伪量化线性层"""
    print("=" * 60)
    print("测试伪量化线性层")
    print("=" * 60)
    
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")
    
    # 创建伪量化线性层
    linear = FakeQuantizedLinear(784, 128, bias=True, config=config)
    
    # 测试数据
    x = torch.randn(32, 784)
    
    # 前向传播
    y = linear(x)
    
    # 检查输出形状
    assert y.shape == (32, 128)
    print(f"✓ 输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 检查是否有伪量化器
    assert hasattr(linear, 'activation_fake_quant')
    assert hasattr(linear, 'weight_fake_quant')
    print("✓ 伪量化器已添加")
    
    print("伪量化线性层测试通过\n")
    return True


def test_qat_preparation():
    """测试 QAT 模型准备"""
    print("=" * 60)
    print("测试 QAT 模型准备")
    print("=" * 60)
    
    # 创建模型
    model = SimpleModel()
    
    # 统计原始模型的可量化层
    original_layers = get_quantizable_layers(model)
    print(f"✓ 原始模型可量化层: {len(original_layers)} 个")
    for name in original_layers:
        print(f"  - {name}")
    
    # 准备 QAT
    config = QuantizationConfig(
        enabled=True,
        bits=8,
        scheme="symmetric",
        fuse_modules=False  # 简单模型不需要融合
    )
    
    qat = QuantizationAwareTraining(model, config)
    prepared_model = qat.prepare()
    
    # 检查模型是否已修改
    assert hasattr(prepared_model.fc1, 'activation_fake_quant')
    assert hasattr(prepared_model.fc2, 'activation_fake_quant')
    print("✓ 伪量化层已插入")
    
    # 测试前向传播
    x = torch.randn(32, 1, 28, 28)
    y = prepared_model(x)
    assert y.shape == (32, 10)
    print(f"✓ 前向传播成功，输出形状: {y.shape}")
    
    print("QAT 模型准备测试通过\n")
    return True


def test_qat_training():
    """测试 QAT 训练流程"""
    print("=" * 60)
    print("测试 QAT 训练流程")
    print("=" * 60)
    
    # 创建模型
    model = SimpleModel()
    
    # 准备 QAT
    config = QuantizationConfig(
        enabled=True,
        bits=8,
        scheme="symmetric",
        fuse_modules=False
    )
    
    qat = QuantizationAwareTraining(model, config)
    prepared_model = qat.prepare()
    
    # 模拟训练
    optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    prepared_model.train()
    
    # 模拟几个训练步骤
    for step in range(5):
        x = torch.randn(32, 1, 28, 28)
        target = torch.randint(0, 10, (32,))
        
        optimizer.zero_grad()
        output = prepared_model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}, Loss: {loss.item():.4f}")
    
    print("✓ QAT 训练成功")
    
    # 冻结观察器
    qat.freeze_observer()
    print("✓ 观察器已冻结")
    
    # 获取量化参数
    quant_params = qat.get_quantization_params()
    print(f"✓ 量化参数数量: {len(quant_params)}")
    
    print("QAT 训练流程测试通过\n")
    return True


def test_quantize_dequantize():
    """测试量化和反量化"""
    print("=" * 60)
    print("测试量化和反量化")
    print("=" * 60)
    
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")
    
    # 测试数据
    x = torch.randn(32, 3, 224, 224)
    
    # 计算量化参数
    min_val = torch.min(x)
    max_val = torch.max(x)
    scale, zero_point = calculate_scale_zero_point(min_val, max_val, config)
    
    print(f"✓ scale: {scale:.6f}, zero_point: {zero_point:.6f}")
    
    # 量化
    x_quant = quantize_tensor(x, scale, zero_point, config)
    print(f"✓ 量化后范围: [{torch.min(x_quant):.0f}, {torch.max(x_quant):.0f}]")
    
    # 反量化
    x_dequant = dequantize_tensor(x_quant, scale, zero_point)
    print(f"✓ 反量化后形状: {x_dequant.shape}")
    
    # 计算误差
    error = torch.abs(x - x_dequant).mean()
    print(f"✓ 平均误差: {error:.6f}")
    
    print("量化和反量化测试通过\n")
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("ParaScale 量化训练测试套件")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    all_passed &= test_quantization_config()
    all_passed &= test_observer()
    all_passed &= test_fake_quantize()
    all_passed &= test_fake_quantized_linear()
    all_passed &= test_qat_preparation()
    all_passed &= test_qat_training()
    all_passed &= test_quantize_dequantize()
    
    print("=" * 60)
    if all_passed:
        print("所有量化测试通过！")
    else:
        print("部分测试失败！")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
