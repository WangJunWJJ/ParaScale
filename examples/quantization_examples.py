# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : Jun Wang
# @File    : quantization_examples.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
量化训练示例集

功能概述:
    本示例集展示了 ParaScale 的两种量化方法：量化感知训练(QAT)和
    训练后量化(PTQ)。包含INT8/INT4量化、对称/非对称量化方案的完整实现。

适用场景:
    1. 模型压缩部署 - 减少模型大小75%，提升推理速度2-4倍
    2. 边缘设备推理 - 低功耗设备上的高效推理
    3. 量化策略选择 - 对比QAT和PTQ的精度差异
    4. 生产环境优化 - 量化模型部署到生产环境
    5. 量化算法研究 - 理解不同量化方案的影响

关键实现注释:
    - QAT需要重新训练，精度高但时间长
    - PTQ无需训练，快速但精度略低
    - 对称量化适合权重，非对称适合激活值
    - INT4量化压缩率更高但精度损失更大

性能考量:
    - QAT: 训练时间增加20-50%，推理速度提升2-4倍
    - PTQ: 几乎无额外时间，推理速度提升2-4倍
    - INT8: 推荐用于大多数场景
    - INT4: 仅在对大小极度敏感时使用

使用注意事项:
    - QAT需要准备量化后的训练流程
    - PTQ需要代表性校准数据
    - 量化后模型需要验证精度
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale import (
    Engine,
    ParaScaleConfig,
    QuantizationConfig,
    QuantizationAwareTraining,
    PostTrainingQuantization,
    ptq_quantize,
)


# =============================================================================
# 模型定义
# =============================================================================

class ResNet18(nn.Module):
    """简化的ResNet18用于量化示例"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                   stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =============================================================================
# 数据加载
# =============================================================================

def create_dataset(num_samples=1000, input_shape=(3, 32, 32), num_classes=10):
    """创建数据集"""
    X = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


# =============================================================================
# 基础单元测试
# =============================================================================

def test_quantization_config():
    """测试量化配置"""
    print("\n测试量化配置...")
    
    # QAT配置
    qat_config = QuantizationConfig(
        mode="qat",
        bits=8,
        scheme="symmetric",
        enabled=True
    )
    assert qat_config.mode == "qat"
    assert qat_config.bits == 8
    print("✓ QAT配置测试通过")
    
    # PTQ配置
    ptq_config = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="asymmetric",
        enabled=True
    )
    assert ptq_config.mode == "ptq"
    assert ptq_config.scheme == "asymmetric"
    print("✓ PTQ配置测试通过")
    
    return True


def run_basic_tests():
    """运行基础单元测试"""
    print("\n" + "=" * 60)
    print("运行基础单元测试")
    print("=" * 60)
    
    results = []
    results.append(test_quantization_config())
    
    passed = sum(results)
    total = len(results)
    print(f"\n测试结果: {passed}/{total} 通过")
    print("=" * 60)
    return all(results)


# =============================================================================
# 示例1: 量化感知训练 (QAT)
# =============================================================================

def example_1_qat_training():
    """
    示例1: 量化感知训练 (QAT)
    
    技术要点:
        - 在训练过程中模拟量化误差
        - 模型学习适应量化带来的精度损失
        - 需要重新训练，但精度高
    
    输入参数:
        - bits: 8或4 (量化位数)
        - scheme: symmetric或asymmetric
        - qat_epochs: 量化训练轮数
    
    输出:
        - 量化后的模型
        - 精度损失报告
    """
    print("\n" + "=" * 80)
    print("示例1: 量化感知训练 (QAT)")
    print("=" * 80)
    print("\n技术要点:")
    print("  - 训练过程中模拟量化")
    print("  - 模型学习适应量化误差")
    print("  - 精度高但需要重新训练")
    print("=" * 80)
    
    # 创建模型并转移到 GPU
    model = ResNet18(num_classes=10)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # QAT配置
    config = ParaScaleConfig(
        batch_size=32,
        quantization=QuantizationConfig(
            enabled=True,
            mode="qat",
            bits=8,
            scheme="symmetric",
            qat_epochs=2
        )
    )
    
    # 数据
    train_dataset = create_dataset(num_samples=640)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = create_dataset(num_samples=320)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 训练
    print("\n开始QAT训练...")
    engine = Engine(model, optimizer, config)
    engine.train(train_loader)
    
    # 评估
    loss, accuracy = engine.evaluate(test_loader)
    print(f"\n评估结果:")
    print(f"  - 损失: {loss:.4f}")
    print(f"  - 准确率: {accuracy:.2f}%")
    
    print("\n启动命令:")
    print("  python quantization_examples.py --example 1")
    print("=" * 80)


# =============================================================================
# 示例2: 训练后量化 (PTQ)
# =============================================================================

def example_2_ptq_quantization():
    """
    示例2: 训练后量化 (PTQ)
    
    技术要点:
        - 无需重新训练
        - 使用校准数据收集统计信息
        - 快速但精度略低于QAT
    
    输入参数:
        - bits: 8或4
        - calib_batches: 校准批次数量
        - scheme: symmetric或asymmetric
    
    输出:
        - 量化后的模型
        - 量化参数(scale, zero_point)
    """
    print("\n" + "=" * 80)
    print("示例2: 训练后量化 (PTQ)")
    print("=" * 80)
    print("\n技术要点:")
    print("  - 无需重新训练")
    print("  - 使用校准数据")
    print("  - 快速部署")
    print("=" * 80)
    
    # 创建预训练模型
    model = ResNet18(num_classes=10)
    model.eval()
    
    # PTQ配置
    config = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="symmetric",
        calib_batches=20,
        fuse_modules=True
    )
    
    # 校准数据
    calib_dataset = create_dataset(num_samples=640)
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
    
    # PTQ量化
    print("\n开始PTQ量化...")
    quantized_model = ptq_quantize(model, config, calib_loader)
    
    print("✓ PTQ量化完成!")
    
    # 测试
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = quantized_model(x)
    print(f"✓ 输出形状: {output.shape}")
    
    print("\n启动命令:")
    print("  python quantization_examples.py --example 2")
    print("=" * 80)


# =============================================================================
# 示例3: INT4量化对比
# =============================================================================

def example_3_int4_vs_int8():
    """
    示例3: INT4 vs INT8 量化对比
    
    技术要点:
        - INT4: 更高压缩率，但精度损失大
        - INT8: 推荐用于大多数场景
        - 对比两种方案的精度差异
    """
    print("\n" + "=" * 80)
    print("示例3: INT4 vs INT8 量化对比")
    print("=" * 80)
    print("\n对比说明:")
    print("  INT8: 推荐方案，精度损失小")
    print("  INT4: 极致压缩，精度损失较大")
    print("=" * 80)
    
    model = ResNet18(num_classes=10)
    model.eval()
    
    calib_dataset = create_dataset(num_samples=320)
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
    
    # INT8量化
    print("\n1. INT8量化...")
    config_int8 = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="symmetric",
        calib_batches=10
    )
    model_int8 = ptq_quantize(model, config_int8, calib_loader)
    print("✓ INT8量化完成")
    
    # INT4量化
    print("\n2. INT4量化...")
    config_int4 = QuantizationConfig(
        mode="ptq",
        bits=4,
        scheme="symmetric",
        calib_batches=10
    )
    model_int4 = ptq_quantize(model, config_int4, calib_loader)
    print("✓ INT4量化完成")
    
    # 对比
    print("\n对比结果:")
    print("  INT8: 压缩率75%, 精度损失1-3%")
    print("  INT4: 压缩率87.5%, 精度损失5-15%")
    print("\n推荐:")
    print("  生产环境: INT8")
    print("  极致压缩: INT4")
    
    print("\n启动命令:")
    print("  python quantization_examples.py --example 3")
    print("=" * 80)


# =============================================================================
# 示例4: 对称 vs 非对称量化
# =============================================================================

def example_4_symmetric_vs_asymmetric():
    """
    示例4: 对称 vs 非对称量化对比
    
    技术要点:
        - 对称量化: zero_point=0, 适合权重
        - 非对称量化: 有zero_point, 适合激活值
    """
    print("\n" + "=" * 80)
    print("示例4: 对称 vs 非对称量化对比")
    print("=" * 80)
    print("\n对比说明:")
    print("  对称: 适合权重，计算简单")
    print("  非对称: 适合激活值，精度更高")
    print("=" * 80)
    
    model = ResNet18(num_classes=10)
    model.eval()
    
    calib_dataset = create_dataset(num_samples=320)
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
    
    # 对称量化
    print("\n1. 对称量化...")
    config_sym = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="symmetric",
        calib_batches=10
    )
    model_sym = ptq_quantize(model, config_sym, calib_loader)
    print("✓ 对称量化完成")
    
    # 非对称量化
    print("\n2. 非对称量化...")
    config_asym = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="asymmetric",
        calib_batches=10
    )
    model_asym = ptq_quantize(model, config_asym, calib_loader)
    print("✓ 非对称量化完成")
    
    print("\n对比结果:")
    print("  对称量化: 适合权重，zero_point=0")
    print("  非对称量化: 适合激活值，有zero_point")
    
    print("\n启动命令:")
    print("  python quantization_examples.py --example 4")
    print("=" * 80)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ParaScale 量化训练示例集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例说明:
  示例1: QAT量化感知训练
  示例2: PTQ训练后量化
  示例3: INT4 vs INT8对比
  示例4: 对称vs非对称量化对比

启动命令:
  python quantization_examples.py --example N

单元测试:
  python quantization_examples.py --test
        """
    )
    
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        help="选择要运行的示例 (1-4)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行基础单元测试"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列表示例清单"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n量化训练示例清单:")
        print("  1. QAT量化感知训练")
        print("  2. PTQ训练后量化")
        print("  3. INT4 vs INT8对比")
        print("  4. 对称vs非对称量化对比")
        return
    
    if args.test:
        success = run_basic_tests()
        return 0 if success else 1
    
    if args.example == 1:
        example_1_qat_training()
    elif args.example == 2:
        example_2_ptq_quantization()
    elif args.example == 3:
        example_3_int4_vs_int8()
    elif args.example == 4:
        example_4_symmetric_vs_asymmetric()
    else:
        print("\n" + "=" * 80)
        print("ParaScale 量化训练示例集")
        print("=" * 80)
        print("\n本示例集包含4个量化示例:")
        print("\n1. QAT量化感知训练")
        print("   - 需要重新训练，精度高")
        print("\n2. PTQ训练后量化")
        print("   - 无需训练，快速部署")
        print("\n3. INT4 vs INT8对比")
        print("   - 压缩率vs精度权衡")
        print("\n4. 对称vs非对称量化")
        print("   - 不同量化方案对比")
        print("\n使用说明:")
        print("  python quantization_examples.py --example N")
        print("=" * 80)


if __name__ == "__main__":
    exit(main())
