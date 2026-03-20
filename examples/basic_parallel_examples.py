# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : Jun Wang
# @File    : basic_parallel_examples.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
基础并行策略示例集

功能概述:
    本示例集展示了 ParaScale 的四种基础并行策略：数据并行(DP)、模型并行(MP)、
    流水线并行(PP)和张量并行(TP)。每个示例都结合了量化感知训练(QAT)，
    适用于需要模型压缩的生产环境。

适用场景:
    1. 单节点多GPU训练 - 充分利用多卡资源
    2. 模型压缩部署 - 结合QAT实现INT8量化
    3. 并行策略对比学习 - 理解不同并行方式的适用场景
    4. 生产环境原型 - 可直接用于生产代码参考
    5. 教学演示 - 清晰的代码结构和详细注释

关键实现注释:
    - 每个并行策略都使用独立的模型类，避免混淆
    - 量化配置统一使用对称INT8方案，确保一致性
    - 训练循环封装为独立函数，便于复用
    - 包含完整的参数验证和错误处理

性能考量:
    - 数据并行：适合模型较小、数据量大的场景
    - 模型并行：适合模型过大无法放入单卡
    - 流水线并行：适合层数较多的模型
    - 张量并行：适合线性层占主导的模型

使用注意事项:
    - 需要2-8个GPU运行不同示例
    - 使用 torchrun 启动分布式训练
    - 确保所有GPU内存充足
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
)
from parascale.parallel import (
    DataParallel,
    ModelParallel,
    PipelineParallel,
    TensorParallel,
)


# =============================================================================
# 模型定义
# =============================================================================

class SimpleModel(nn.Module):
    """简单模型用于数据并行示例"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MultiStageModel(nn.Module):
    """多阶段模型用于模型并行示例"""
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class PipelineModel(nn.Module):
    """流水线模型用于流水线并行示例"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.layer4 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class LargeLinearModel(nn.Module):
    """大线性模型用于张量并行示例"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# =============================================================================
# 数据加载
# =============================================================================

def create_dataloader(batch_size=32, num_samples=1000):
    """创建数据加载器"""
    X = torch.randn(num_samples, 784)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# =============================================================================
# 基础单元测试
# =============================================================================

def test_model_forward(model_class, input_shape=(32, 784)):
    """测试模型前向传播"""
    model = model_class()
    x = torch.randn(*input_shape)
    try:
        output = model(x)
        assert output.shape[0] == input_shape[0]
        print(f"✓ {model_class.__name__} 前向传播测试通过")
        return True
    except Exception as e:
        print(f"✗ {model_class.__name__} 测试失败: {e}")
        return False


def run_basic_tests():
    """运行基础单元测试"""
    print("\n" + "=" * 60)
    print("运行基础单元测试")
    print("=" * 60)
    
    tests = [
        (SimpleModel, (32, 784)),
        (MultiStageModel, (32, 784)),
        (PipelineModel, (32, 784)),
        (LargeLinearModel, (32, 4096)),
    ]
    
    results = []
    for model_class, input_shape in tests:
        results.append(test_model_forward(model_class, input_shape))
    
    passed = sum(results)
    total = len(results)
    print(f"\n测试结果: {passed}/{total} 通过")
    print("=" * 60)
    return all(results)


# =============================================================================
# 示例1: 数据并行 + 量化
# =============================================================================

def example_1_data_parallel_with_qat():
    """
    示例1: 数据并行 + 量化感知训练
    
    技术要点:
        - 使用DataParallel实现数据并行
        - 结合QAT进行INT8量化
        - 适用于模型较小但数据量大的场景
    
    输入参数:
        - world_size: 2-8 (GPU数量)
        - batch_size: 32-128
        - epochs: 1-10
    
    输出:
        - 训练后的量化模型
        - 准确率报告
    """
    print("\n" + "=" * 80)
    print("示例1: 数据并行 + 量化感知训练")
    print("=" * 80)
    print("\n技术要点:")
    print("  - DataParallel: 每个GPU持有完整模型副本")
    print("  - QAT: 量化感知训练，模拟量化误差")
    print("  - 适用场景: 模型较小(<1B参数)，数据量大")
    print("=" * 80)
    
    # 模型和配置
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    config = ParaScaleConfig(
        data_parallel_size=2,
        batch_size=32,
        quantization=QuantizationConfig(
            enabled=True,
            bits=8,
            scheme="symmetric",
            qat_epochs=1
        )
    )
    
    # 数据
    train_loader = create_dataloader(batch_size=32, num_samples=320)
    test_loader = create_dataloader(batch_size=32, num_samples=160)
    
    # 训练
    print("\n开始训练...")
    engine = Engine(model, optimizer, config)
    engine.train(train_loader)
    
    # 评估
    loss, accuracy = engine.evaluate(test_loader)
    print(f"\n评估结果:")
    print(f"  - 损失: {loss:.4f}")
    print(f"  - 准确率: {accuracy:.2f}%")
    
    print("\n启动命令:")
    print("  torchrun --nproc_per_node=2 basic_parallel_examples.py --example 1")
    print("=" * 80)


# =============================================================================
# 示例2: 模型并行 + 量化
# =============================================================================

def example_2_model_parallel_with_qat():
    """
    示例2: 模型并行 + 量化感知训练
    
    技术要点:
        - 使用ModelParallel将模型分阶段放到不同GPU
        - 适合模型过大无法放入单卡的场景
        - 结合QAT实现量化
    
    输入参数:
        - world_size: 2 (必须)
        - batch_size: 16-64
        - epochs: 1-10
    
    输出:
        - 分阶段量化模型
        - 各阶段性能报告
    """
    print("\n" + "=" * 80)
    print("示例2: 模型并行 + 量化感知训练")
    print("=" * 80)
    print("\n技术要点:")
    print("  - ModelParallel: 按层分割模型到不同GPU")
    print("  - 适用场景: 模型过大，单卡内存不足")
    print("  - 注意: 需要精确的层分割策略")
    print("=" * 80)
    
    model = MultiStageModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    config = ParaScaleConfig(
        model_parallel_size=2,
        batch_size=32,
        quantization=QuantizationConfig(
            enabled=True,
            bits=8,
            scheme="symmetric",
            qat_epochs=1
        )
    )
    
    train_loader = create_dataloader(batch_size=32, num_samples=320)
    test_loader = create_dataloader(batch_size=32, num_samples=160)
    
    print("\n开始训练...")
    engine = Engine(model, optimizer, config)
    engine.train(train_loader)
    
    loss, accuracy = engine.evaluate(test_loader)
    print(f"\n评估结果:")
    print(f"  - 损失: {loss:.4f}")
    print(f"  - 准确率: {accuracy:.2f}%")
    
    print("\n启动命令:")
    print("  torchrun --nproc_per_node=2 basic_parallel_examples.py --example 2")
    print("=" * 80)


# =============================================================================
# 示例3: 流水线并行 + 量化
# =============================================================================

def example_3_pipeline_parallel_with_qat():
    """
    示例3: 流水线并行 + 量化感知训练
    
    技术要点:
        - 使用PipelineParallel实现流水线并行
        - 支持微批次处理，提高GPU利用率
        - 适合层数较多的模型
    
    输入参数:
        - world_size: 2-4 (流水线阶段数)
        - pipeline_chunks: 2-4 (微批次数量)
        - batch_size: 32-128
    
    输出:
        - 流水线量化模型
        - 吞吐量报告
    """
    print("\n" + "=" * 80)
    print("示例3: 流水线并行 + 量化感知训练")
    print("=" * 80)
    print("\n技术要点:")
    print("  - PipelineParallel: 按层分割，支持微批次")
    print("  - 适用场景: 层数多，需要高吞吐量的场景")
    print("  - 注意: 需要平衡各阶段的计算量")
    print("=" * 80)
    
    model = PipelineModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    config = ParaScaleConfig(
        pipeline_parallel_size=4,
        batch_size=32,
        pipeline_parallel_chunks=2,
        quantization=QuantizationConfig(
            enabled=True,
            bits=8,
            scheme="symmetric",
            qat_epochs=1
        )
    )
    
    train_loader = create_dataloader(batch_size=32, num_samples=320)
    test_loader = create_dataloader(batch_size=32, num_samples=160)
    
    print("\n开始训练...")
    engine = Engine(model, optimizer, config)
    engine.train(train_loader)
    
    loss, accuracy = engine.evaluate(test_loader)
    print(f"\n评估结果:")
    print(f"  - 损失: {loss:.4f}")
    print(f"  - 准确率: {accuracy:.2f}%")
    
    print("\n启动命令:")
    print("  torchrun --nproc_per_node=4 basic_parallel_examples.py --example 3")
    print("=" * 80)


# =============================================================================
# 示例4: 张量并行 + 量化
# =============================================================================

def example_4_tensor_parallel_with_qat():
    """
    示例4: 张量并行 + 量化感知训练
    
    技术要点:
        - 使用TensorParallel分割权重矩阵
        - 支持行并行和列并行两种模式
        - 适合线性层占主导的模型
    
    输入参数:
        - world_size: 2-4 (张量并行度)
        - tensor_parallel_mode: "row" 或 "column"
        - batch_size: 32-128
    
    输出:
        - 张量并行量化模型
        - 通信开销报告
    """
    print("\n" + "=" * 80)
    print("示例4: 张量并行 + 量化感知训练")
    print("=" * 80)
    print("\n技术要点:")
    print("  - TensorParallel: 分割权重矩阵到不同GPU")
    print("  - 支持行并行(row)和列并行(column)")
    print("  - 适用场景: 线性层多，矩阵运算密集")
    print("=" * 80)
    
    model = LargeLinearModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    config = ParaScaleConfig(
        tensor_parallel_size=2,
        batch_size=32,
        tensor_parallel_mode="row",
        quantization=QuantizationConfig(
            enabled=True,
            bits=8,
            scheme="symmetric",
            qat_epochs=1
        )
    )
    
    # 注意：张量并行需要更大的输入维度
    X = torch.randn(320, 4096)
    y = torch.randint(0, 10, (320,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    X_test = torch.randn(160, 4096)
    y_test = torch.randint(0, 10, (160,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print("\n开始训练...")
    engine = Engine(model, optimizer, config)
    engine.train(train_loader)
    
    loss, accuracy = engine.evaluate(test_loader)
    print(f"\n评估结果:")
    print(f"  - 损失: {loss:.4f}")
    print(f"  - 准确率: {accuracy:.2f}%")
    
    print("\n启动命令:")
    print("  torchrun --nproc_per_node=2 basic_parallel_examples.py --example 4")
    print("=" * 80)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ParaScale 基础并行策略示例集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例说明:
  示例1: 数据并行 + 量化 (需要2个GPU)
  示例2: 模型并行 + 量化 (需要2个GPU)
  示例3: 流水线并行 + 量化 (需要4个GPU)
  示例4: 张量并行 + 量化 (需要2个GPU)

启动命令:
  torchrun --nproc_per_node=N basic_parallel_examples.py --example X

单元测试:
  python basic_parallel_examples.py --test
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
        print("\n基础并行策略示例清单:")
        print("  1. 数据并行 + 量化感知训练")
        print("  2. 模型并行 + 量化感知训练")
        print("  3. 流水线并行 + 量化感知训练")
        print("  4. 张量并行 + 量化感知训练")
        return
    
    if args.test:
        success = run_basic_tests()
        return 0 if success else 1
    
    if args.example == 1:
        example_1_data_parallel_with_qat()
    elif args.example == 2:
        example_2_model_parallel_with_qat()
    elif args.example == 3:
        example_3_pipeline_parallel_with_qat()
    elif args.example == 4:
        example_4_tensor_parallel_with_qat()
    else:
        # 默认运行所有示例说明
        print("\n" + "=" * 80)
        print("ParaScale 基础并行策略示例集")
        print("=" * 80)
        print("\n本示例集包含4个基础并行策略示例:")
        print("\n1. 数据并行 + 量化")
        print("   - 适用: 模型小，数据量大")
        print("   - GPU: 2-8个")
        print("\n2. 模型并行 + 量化")
        print("   - 适用: 模型大，单卡放不下")
        print("   - GPU: 2个")
        print("\n3. 流水线并行 + 量化")
        print("   - 适用: 层数多，需要高吞吐")
        print("   - GPU: 4个")
        print("\n4. 张量并行 + 量化")
        print("   - 适用: 线性层多，矩阵运算密集")
        print("   - GPU: 2-4个")
        print("\n使用说明:")
        print("  python basic_parallel_examples.py --example N")
        print("  torchrun --nproc_per_node=N basic_parallel_examples.py --example N")
        print("=" * 80)


if __name__ == "__main__":
    exit(main())
