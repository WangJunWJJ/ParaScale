# -*- coding: utf-8 -*-
# @Time    : 2026/3/12
# @Author  : Jun Wang
# @File    : para_engine_example.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaEngine 使用示例

本示例展示了如何使用 ParaEngine 进行自动并行策略选择的分布式训练。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale import ParaEngine, ParaScaleConfig


class TransformerModel(nn.Module):
    """简单的 Transformer 模型用于示例"""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model
    
    def forward(self, x):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class CNNModel(nn.Module):
    """卷积神经网络模型用于示例"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class MLPModel(nn.Module):
    """多层感知机模型用于示例"""
    def __init__(self, input_size=784, hidden_size=512, num_layers=4, num_classes=10):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def create_dummy_data(num_samples=1000, input_shape=(784,), num_classes=10, input_dtype=torch.float32):
    """创建虚拟数据用于示例"""
    if input_dtype == torch.long:
        # 对于 Embedding 层，输入是整数索引
        X = torch.randint(0, num_classes, (num_samples, *input_shape))
    elif len(input_shape) == 3:
        X = torch.randn(num_samples, *input_shape, dtype=input_dtype)
    else:
        X = torch.randn(num_samples, *input_shape, dtype=input_dtype)

    if num_classes > 0:
        y = torch.randint(0, num_classes, (num_samples,))
    else:
        y = torch.randn(num_samples, 1)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


def example_1_auto_parallel_small_model():
    """
    示例 1: 小模型的自动并行配置
    
    小模型（< 1B 参数）会自动选择纯数据并行策略
    """
    print("\n" + "="*80)
    print("示例 1: 小模型的自动并行配置")
    print("="*80)
    
    model = MLPModel(input_size=784, hidden_size=256, num_layers=3, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    engine = ParaEngine(
        model=model,
        optimizer=optimizer,
        auto_parallel=True,
        auto_init_distributed=False
    )
    
    strategy = engine.get_strategy()
    print(f"\n自动选择的策略:")
    print(f"  策略类型：{strategy.strategy_type}")
    print(f"  数据并行大小 (DP): {strategy.dp_size}")
    print(f"  张量并行大小 (TP): {strategy.tp_size}")
    print(f"  流水线并行大小 (PP): {strategy.pp_size}")
    print(f"  选择原因：{strategy.reason}")
    print(f"  预估加速比：{strategy.estimated_speedup:.2f}x")
    print("="*80 + "\n")


def example_2_auto_parallel_medium_model():
    """
    示例 2: 中等模型的自动并行配置
    
    中等模型（1B-10B 参数）会根据内存约束选择 DP+TP 混合策略
    """
    print("\n" + "="*80)
    print("示例 2: 中等模型的自动并行配置")
    print("="*80)
    
    model = TransformerModel(
        vocab_size=30522,
        d_model=768,
        nhead=12,
        num_layers=12,
        num_classes=10
    )
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    engine = ParaEngine(
        model=model,
        optimizer=optimizer,
        auto_parallel=True,
        auto_init_distributed=False
    )
    
    strategy = engine.get_strategy()
    print(f"\n自动选择的策略:")
    print(f"  策略类型：{strategy.strategy_type}")
    print(f"  数据并行大小 (DP): {strategy.dp_size}")
    print(f"  张量并行大小 (TP): {strategy.tp_size}")
    print(f"  流水线并行大小 (PP): {strategy.pp_size}")
    print(f"  选择原因：{strategy.reason}")
    print(f"  预估加速比：{strategy.estimated_speedup:.2f}x")
    print("="*80 + "\n")


def example_3_auto_parallel_large_model():
    """
    示例 3: 大模型的自动并行配置
    
    大模型（> 10B 参数）会自动选择 3D 混合并行策略（DP+TP+PP）
    """
    print("\n" + "="*80)
    print("示例 3: 大模型的自动并行配置（模拟）")
    print("="*80)
    
    model = TransformerModel(
        vocab_size=50257,
        d_model=2048,
        nhead=32,
        num_layers=24,
        num_classes=100
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    engine = ParaEngine(
        model=model,
        optimizer=optimizer,
        auto_parallel=True,
        auto_init_distributed=False
    )
    
    strategy = engine.get_strategy()
    print(f"\n自动选择的策略:")
    print(f"  策略类型：{strategy.strategy_type}")
    print(f"  数据并行大小 (DP): {strategy.dp_size}")
    print(f"  张量并行大小 (TP): {strategy.tp_size}")
    print(f"  流水线并行大小 (PP): {strategy.pp_size}")
    print(f"  选择原因：{strategy.reason}")
    print(f"  预估加速比：{strategy.estimated_speedup:.2f}x")
    print("="*80 + "\n")


def example_4_manual_config():
    """
    示例 4: 手动配置并行策略
    
    用户也可以手动指定并行配置，不使用自动选择
    """
    print("\n" + "="*80)
    print("示例 4: 手动配置并行策略")
    print("="*80)
    
    model = CNNModel(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 手动配置并行策略（在单GPU环境下使用DP=1, TP=1, PP=1）
    # 在多GPU环境下可以调整这些值
    config = ParaScaleConfig(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_mode="row",
        pipeline_parallel_chunks=2
    )
    
    engine = ParaEngine(
        model=model,
        optimizer=optimizer,
        config=config,
        auto_parallel=False,
        auto_init_distributed=False
    )
    
    strategy = engine.get_strategy()
    print(f"\n手动配置的策略:")
    print(f"  策略类型：hybrid (手动配置)")
    if strategy:
        print(f"  数据并行大小 (DP): {strategy.dp_size}")
        print(f"  张量并行大小 (TP): {strategy.tp_size}")
        print(f"  流水线并行大小 (PP): {strategy.pp_size}")
    else:
        print(f"  数据并行大小 (DP): {config.data_parallel_size}")
        print(f"  张量并行大小 (TP): {config.tensor_parallel_size}")
        print(f"  流水线并行大小 (PP): {config.pipeline_parallel_size}")
    print("="*80 + "\n")


def example_5_training_workflow():
    """
    示例 5: 完整的训练流程
    
    展示使用 ParaEngine 进行完整训练的流程
    """
    print("\n" + "="*80)
    print("示例 5: 完整的训练流程")
    print("="*80)
    
    model = MLPModel(input_size=784, hidden_size=128, num_layers=2, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = create_dummy_data(num_samples=500, input_shape=(784,), num_classes=10)
    test_loader = create_dummy_data(num_samples=100, input_shape=(784,), num_classes=10)
    
    engine = ParaEngine(
        model=model,
        optimizer=optimizer,
        auto_parallel=True,
        auto_init_distributed=False
    )
    
    strategy = engine.get_strategy()
    print(f"\n自动选择的策略：{strategy.strategy_type}")
    print(f"并行配置：DP={strategy.dp_size}, TP={strategy.tp_size}, PP={strategy.pp_size}")
    
    print("\n开始训练...")
    engine.train(train_loader, epochs=2)
    
    print("\n开始评估...")
    loss, accuracy = engine.evaluate(test_loader)
    print(f"评估结果 - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    parallel_info = engine.get_parallel_info()
    if parallel_info:
        print(f"\n并行信息:")
        for key, value in parallel_info.items():
            print(f"  {key}: {value}")
    
    print("="*80 + "\n")


def example_6_mlp_classification():
    """
    示例 6: MLP 分类

    展示使用 ParaEngine 训练 MLP 进行分类
    """
    print("\n" + "="*80)
    print("示例 6: MLP 分类")
    print("="*80)

    model = MLPModel(input_size=784, hidden_size=256, num_layers=3, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_loader = create_dummy_data(num_samples=1000, input_shape=(784,), num_classes=10)

    engine = ParaEngine(
        model=model,
        optimizer=optimizer,
        auto_parallel=True,
        auto_init_distributed=False
    )

    strategy = engine.get_strategy()
    print(f"\n自动选择的策略：{strategy.strategy_type}")
    print(f"并行配置：DP={strategy.dp_size}, TP={strategy.tp_size}, PP={strategy.pp_size}")

    print("\n开始训练 MLP...")
    engine.train(train_loader, epochs=1)

    print("MLP 训练完成!")
    print("="*80 + "\n")


def example_7_simple_sequence_model():
    """
    示例 7: 简单序列分类模型

    展示使用 ParaEngine 训练简单的序列分类模型
    """
    print("\n" + "="*80)
    print("示例 7: 简单序列分类模型")
    print("="*80)

    # 使用 MLP 模型处理展平的序列
    model = MLPModel(input_size=32*32, hidden_size=256, num_layers=3, num_classes=10)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    num_samples = 200
    train_loader = create_dummy_data(num_samples=num_samples, input_shape=(32*32,), num_classes=10)

    engine = ParaEngine(
        model=model,
        optimizer=optimizer,
        auto_parallel=True,
        auto_init_distributed=False
    )

    strategy = engine.get_strategy()
    print(f"\n自动选择的策略：{strategy.strategy_type}")
    print(f"并行配置：DP={strategy.dp_size}, TP={strategy.tp_size}, PP={strategy.pp_size}")

    print("\n开始训练序列分类模型...")
    engine.train(train_loader, epochs=1)

    print("序列分类模型训练完成!")
    print("="*80 + "\n")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("ParaScale ParaEngine 使用示例")
    print("="*80)
    print("\nParaEngine 是一个支持自动并行策略选择的智能训练引擎。")
    print("它能够根据模型规模、硬件状态自适应决策最优的并行策略组合。")
    print("="*80)
    
    example_1_auto_parallel_small_model()
    example_2_auto_parallel_medium_model()
    example_3_auto_parallel_large_model()
    example_4_manual_config()
    example_5_training_workflow()
    example_6_mlp_classification()
    example_7_simple_sequence_model()
    
    print("\n" + "="*80)
    print("所有示例运行完成!")
    print("="*80)
    print("\n提示:")
    print("- 在实际使用中，需要使用 torchrun 启动分布式训练")
    print("- 例如：torchrun --nproc_per_node=8 para_engine_example.py")
    print("- ParaEngine 会自动检测 GPU 数量并选择最优并行策略")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
