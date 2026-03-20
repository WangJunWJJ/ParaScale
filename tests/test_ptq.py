# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : Jun Wang
# @File    : test_ptq.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale PTQ（训练后量化）测试模块

本模块测试 PTQ 功能，包括校准、权重量化和模型转换。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.quantization import (
    QuantizationConfig,
    PostTrainingQuantization,
    ptq_quantize,
    load_quantized_model,
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


class CNNModel(nn.Module):
    """CNN 测试模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        return self.fc(x)


def create_calibration_data(num_samples=1000, input_shape=(784,)):
    """创建校准数据"""
    X = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader


def create_cnn_calibration_data(num_samples=1000, input_shape=(3, 32, 32)):
    """创建 CNN 校准数据"""
    X = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader


def test_ptq_basic():
    """测试 PTQ 基本流程"""
    print("=" * 60)
    print("测试 PTQ 基本流程")
    print("=" * 60)
    
    # 创建模型
    model = SimpleModel()
    
    # 配置 PTQ
    config = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="symmetric",
        per_channel=True,
        calib_batches=10,
        fuse_modules=False
    )
    
    # 创建校准数据
    calib_loader = create_calibration_data(num_samples=320)
    
    # 创建 PTQ 量化器
    ptq = PostTrainingQuantization(model, config)
    
    # 准备模型
    print("准备模型...")
    ptq.prepare()
    print("✓ 模型准备完成")
    
    # 校准
    print("开始校准...")
    ptq.calibrate(calib_loader)
    print("✓ 校准完成")
    
    # 转换模型
    print("转换模型...")
    quantized_model = ptq.convert()
    print("✓ 模型转换完成")
    
    # 测试前向传播
    x = torch.randn(32, 784)
    with torch.no_grad():
        output = quantized_model(x)
    assert output.shape == (32, 10)
    print("✓ 前向传播测试通过")
    
    print("PTQ 基本流程测试通过\n")
    return True


def test_ptq_with_module_fusion():
    """测试带模块融合的 PTQ"""
    print("=" * 60)
    print("测试带模块融合的 PTQ")
    print("=" * 60)
    
    # 创建 CNN 模型（适合融合）
    model = CNNModel()
    
    # 配置 PTQ（启用模块融合）
    config = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="symmetric",
        per_channel=True,
        calib_batches=10,
        fuse_modules=True
    )
    
    # 创建校准数据
    calib_loader = create_cnn_calibration_data(num_samples=320)
    
    # 创建 PTQ 量化器
    ptq = PostTrainingQuantization(model, config)
    
    # 准备模型
    print("准备模型（带模块融合）...")
    ptq.prepare()
    print("✓ 模型准备完成")
    
    # 校准
    print("开始校准...")
    ptq.calibrate(calib_loader)
    print("✓ 校准完成")
    
    # 转换模型
    print("转换模型...")
    quantized_model = ptq.convert()
    print("✓ 模型转换完成")
    
    # 测试前向传播
    x = torch.randn(32, 3, 32, 32)
    with torch.no_grad():
        output = quantized_model(x)
    assert output.shape == (32, 10)
    print("✓ 前向传播测试通过")
    
    print("带模块融合的 PTQ 测试通过\n")
    return True


def test_ptq_int4():
    """测试 INT4 量化"""
    print("=" * 60)
    print("测试 INT4 量化")
    print("=" * 60)
    
    model = SimpleModel()
    
    # 配置 INT4 量化
    config = QuantizationConfig(
        mode="ptq",
        bits=4,
        scheme="symmetric",
        per_channel=True,
        calib_batches=10,
        fuse_modules=False
    )
    
    calib_loader = create_calibration_data(num_samples=320)
    
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader)
    quantized_model = ptq.convert()
    
    # 测试前向传播
    x = torch.randn(32, 784)
    with torch.no_grad():
        output = quantized_model(x)
    assert output.shape == (32, 10)
    print("✓ INT4 量化测试通过")
    
    print("INT4 量化测试通过\n")
    return True


def test_ptq_asymmetric():
    """测试非对称量化"""
    print("=" * 60)
    print("测试非对称量化")
    print("=" * 60)
    
    model = SimpleModel()
    
    # 配置非对称量化
    config = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="asymmetric",
        per_channel=True,
        calib_batches=10,
        fuse_modules=False
    )
    
    calib_loader = create_calibration_data(num_samples=320)
    
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader)
    quantized_model = ptq.convert()
    
    # 测试前向传播
    x = torch.randn(32, 784)
    with torch.no_grad():
        output = quantized_model(x)
    assert output.shape == (32, 10)
    print("✓ 非对称量化测试通过")
    
    print("非对称量化测试通过\n")
    return True


def test_ptq_quantization_params():
    """测试获取量化参数"""
    print("=" * 60)
    print("测试获取量化参数")
    print("=" * 60)
    
    model = SimpleModel()
    config = QuantizationConfig(mode="ptq", bits=8, calib_batches=10)
    calib_loader = create_calibration_data(num_samples=320)
    
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader)
    
    # 获取量化参数
    quant_params = ptq.get_quantization_params()
    
    print(f"✓ 获取到 {len(quant_params)} 个量化参数")
    
    # 验证参数格式
    for layer_name, params in quant_params.items():
        assert 'scale' in params
        assert 'zero_point' in params
        print(f"  - {layer_name}: scale={params['scale'][:3] if isinstance(params['scale'], list) else params['scale']:.4f}")
    
    print("量化参数测试通过\n")
    return True


def test_ptq_export_load():
    """测试导出和加载量化模型"""
    print("=" * 60)
    print("测试导出和加载量化模型")
    print("=" * 60)
    
    import tempfile
    import os
    
    model = SimpleModel()
    config = QuantizationConfig(mode="ptq", bits=8, calib_batches=10)
    calib_loader = create_calibration_data(num_samples=320)
    
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader)
    quantized_model = ptq.convert()
    
    # 导出模型
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        ptq.export(tmp_path)
        print(f"✓ 模型已导出到：{tmp_path}")
        
        # 加载模型
        loaded_model, loaded_config, loaded_params = load_quantized_model(tmp_path, model=SimpleModel())
        print("✓ 模型加载成功")
        
        # 验证加载的模型
        x = torch.randn(32, 784)
        with torch.no_grad():
            output1 = quantized_model(x)
            output2 = loaded_model(x)
        
        # 输出应该相同
        assert torch.allclose(output1, output2, rtol=1e-4)
        print("✓ 加载的模型输出与原始模型一致")
        
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    print("导出和加载测试通过\n")
    return True


def test_ptq_evaluate():
    """测试评估量化模型"""
    print("=" * 60)
    print("测试评估量化模型")
    print("=" * 60)
    
    model = SimpleModel()
    config = QuantizationConfig(mode="ptq", bits=8, calib_batches=10)
    calib_loader = create_calibration_data(num_samples=320)
    test_loader = create_calibration_data(num_samples=200)
    
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader)
    ptq.convert()
    
    # 评估（使用CPU避免设备不匹配）
    criterion = nn.CrossEntropyLoss()
    loss, accuracy = ptq.evaluate(test_loader, criterion, device=torch.device('cpu'))
    
    print(f"✓ 评估结果 - Loss: {loss:.4f}")
    print(f"✓ 评估结果 - Accuracy: {accuracy:.2f}%")
    
    print("评估测试通过\n")
    return True


def test_ptq_convenience_function():
    """测试便捷函数 ptq_quantize"""
    print("=" * 60)
    print("测试便捷函数 ptq_quantize")
    print("=" * 60)
    
    model = SimpleModel()
    config = QuantizationConfig(mode="ptq", bits=8, calib_batches=10)
    calib_loader = create_calibration_data(num_samples=320)
    
    # 使用便捷函数
    quantized_model = ptq_quantize(model, config, calib_loader)
    
    # 测试前向传播
    x = torch.randn(32, 784)
    with torch.no_grad():
        output = quantized_model(x)
    assert output.shape == (32, 10)
    print("✓ 便捷函数测试通过")
    
    print("便捷函数测试通过\n")
    return True


def test_ptq_quantization_info():
    """测试获取量化信息"""
    print("=" * 60)
    print("测试获取量化信息")
    print("=" * 60)
    
    model = SimpleModel()
    config = QuantizationConfig(mode="ptq", bits=8, calib_batches=10)
    calib_loader = create_calibration_data(num_samples=320)
    
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader)
    ptq.convert()
    
    # 获取量化信息
    info = ptq.get_quantization_info()
    
    print(f"✓ 总参数：{info['total_params']:,}")
    print(f"✓ 已量化参数：{info['quantized_params']:,}")
    print(f"✓ 量化比例：{info['quantization_ratio']*100:.2f}%")
    print(f"✓ 量化位数：{info['bits']} bit")
    
    # 打印详细信息
    ptq.print_quantization_info()
    
    print("量化信息测试通过\n")
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("ParaScale PTQ（训练后量化）测试套件")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    try:
        all_passed &= test_ptq_basic()
        all_passed &= test_ptq_with_module_fusion()
        all_passed &= test_ptq_int4()
        all_passed &= test_ptq_asymmetric()
        all_passed &= test_ptq_quantization_params()
        all_passed &= test_ptq_export_load()
        all_passed &= test_ptq_evaluate()
        all_passed &= test_ptq_convenience_function()
        all_passed &= test_ptq_quantization_info()
        
        print("=" * 60)
        if all_passed:
            print("✅ 所有 PTQ 测试通过！")
        else:
            print("❌ 部分测试失败！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
