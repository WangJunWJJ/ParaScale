# -*- coding: utf-8 -*-
# @Time    : 2026/3/19
# @Author  : Jun Wang
# @File    : fourbit_optimizer_example.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

import os
import sys

# 添加项目根目录到路径（必须在其他导入之前）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from parascale.optimizers import FourBitAdamW, FourBitSGD

"""
4bit 优化器使用示例

本示例展示如何在实际训练中使用 4bit 量化优化器来节省内存，
同时保持训练效果。

适用场景：
- 大模型训练，内存受限
- 需要训练更大模型的场景
- 边缘设备训练
"""


# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TransformerModel(nn.Module):
    """Transformer 模型示例"""
    def __init__(self,
                vocab_size=10000,
                d_model=512,
                nhead=8,
                num_layers=6,
                dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                        num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def forward(self, src):
        # Embedding
        x = self.embedding(src) * (self.d_model ** 0.5)
        # Transformer
        x = self.transformer(x)
        # Output projection
        output = self.fc(x)
        return output


class SimpleClassifier(nn.Module):
    """简单分类器"""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def example_1_basic_usage():
    """示例 1: 基本使用"""
    print("\n" + "=" * 60)
    print("示例 1: 4bit AdamW 基本使用")
    print("=" * 60)
    
    # 创建模型
    model = SimpleClassifier(input_dim=784, hidden_dim=256, num_classes=10)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 创建 4bit AdamW 优化器
    optimizer = FourBitAdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        group_size=128,  # 分组大小，影响精度和内存
        compensate_quant_error=True  # 启用误差补偿
    )
    
    print(f"优化器类型: {type(optimizer).__name__}")
    print(f"分组大小: {optimizer.group_size}")
    
    # 模拟训练
    criterion = nn.CrossEntropyLoss()
    
    print("\n训练 10 步...")
    for step in range(10):
        # 模拟数据
        inputs = torch.randn(32, 784)
        labels = torch.randint(0, 10, (32,))
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step + 1}, Loss: {loss.item():.4f}")
    
    # 打印内存统计
    print()
    optimizer.print_memory_stats()


def example_2_memory_comparison():
    """示例 2: 内存使用对比"""
    print("\n" + "=" * 60)
    print("示例 2: 内存使用对比")
    print("=" * 60)
    
    # 创建一个较大的模型
    model = TransformerModel(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数数量: {total_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 标准 AdamW 内存估算
    standard_adamw_memory = total_params * 12 / 1024 / 1024
    print(f"\n标准 AdamW 内存需求:")
    print(f"  - 参数: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  - Momentum: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  - Variance: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  - 总计: {standard_adamw_memory:.2f} MB")
    
    # 4bit AdamW
    optimizer = FourBitAdamW(model.parameters(), lr=1e-3)
    
    # 执行一步以初始化状态
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randint(0, 10000, (4, 32))
    labels = torch.randint(0, 10000, (4, 32))
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, 10000), labels.view(-1))
    loss.backward()
    optimizer.step()
    
    stats = optimizer.get_memory_stats()

    # 计算参数内存（FP32）
    fp32_params_bytes = stats['total_params'] * 4

    print(f"\n4bit AdamW 内存需求:")
    print(f"  - 参数: {fp32_params_bytes / 1024 / 1024:.2f} MB")
    print(f"  - 4bit 状态: {stats['quantized_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  - 总计: {(fp32_params_bytes + stats['quantized_bytes']) / 1024 / 1024:.2f} MB")

    print(f"\n内存节省: {stats['savings_percent']:.1f}%")
    savings_bytes = stats['standard_bytes'] - stats['quantized_bytes']
    print(f"可节省内存: {savings_bytes / 1024 / 1024:.2f} MB")


def example_3_training_comparison():
    """示例 3: 训练效果对比"""
    print("\n" + "=" * 60)
    print("示例 3: 训练效果对比（4bit AdamW vs 标准 AdamW）")
    print("=" * 60)
    
    # 创建两个相同初始化的模型
    torch.manual_seed(42)
    model_4bit = SimpleClassifier(input_dim=784,
                hidden_dim=128,
                num_classes=10)
    
    torch.manual_seed(42)
    model_standard = SimpleClassifier(input_dim=784,
                hidden_dim=128,
                num_classes=10)
    
    # 确保权重相同
    model_standard.load_state_dict(model_4bit.state_dict())
    
    # 创建优化器
    optimizer_4bit = FourBitAdamW(model_4bit.parameters(), lr=1e-3)
    optimizer_standard = torch.optim.AdamW(model_standard.parameters(),
                lr=1e-3)
    
    # 训练数据
    torch.manual_seed(123)
    X = torch.randn(200, 784)
    y = torch.randint(0, 10, (200,))
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    losses_4bit = []
    losses_standard = []
    
    print("\n训练 50 轮...")
    for epoch in range(50):
        # 4bit AdamW
        optimizer_4bit.zero_grad()
        output_4bit = model_4bit(X)
        loss_4bit = criterion(output_4bit, y)
        loss_4bit.backward()
        optimizer_4bit.step()
        losses_4bit.append(loss_4bit.item())
        
        # 标准 AdamW
        optimizer_standard.zero_grad()
        output_standard = model_standard(X)
        loss_standard = criterion(output_standard, y)
        loss_standard.backward()
        optimizer_standard.step()
        losses_standard.append(loss_standard.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: 4bit Loss = {loss_4bit.item():.4f}, "
                  f"Standard Loss = {loss_standard.item():.4f}")
    
    # 对比结果
    print("\n训练结果对比:")
    print(f"初始损失 - 4bit: {losses_4bit[0]:.4f}, 标准: {losses_standard[0]:.4f}")
    print(f"最终损失 - 4bit: {losses_4bit[-1]:.4f}, 标准: {losses_standard[-1]:.4f}")
    print(f"损失下降 - 4bit: {losses_4bit[0] - losses_4bit[-1]:.4f}, "
          f"标准: {losses_standard[0] - losses_standard[-1]:.4f}")
    
    # 计算最终差异
    diff = abs(losses_4bit[-1] - losses_standard[-1])
    print(f"\n最终损失差异: {diff:.4f}")
    
    if diff < 0.1:
        print("✓ 4bit AdamW 与标准 AdamW 训练效果相近！")
    else:
        print("! 训练效果有一定差异，但仍在可接受范围")


def example_4_sgd_momentum():
    """示例 4: 4bit SGD with Momentum"""
    print("\n" + "=" * 60)
    print("示例 4: 4bit SGD with Momentum")
    print("=" * 60)
    
    model = SimpleClassifier(input_dim=784, hidden_dim=256, num_classes=10)
    
    # 4bit SGD
    optimizer = FourBitSGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        group_size=128,
        compensate_quant_error=True
    )
    
    print(f"优化器类型: {type(optimizer).__name__}")
    print(f"学习率: {optimizer.param_groups[0]['lr']}")
    print(f"动量: {optimizer.param_groups[0]['momentum']}")
    
    # 模拟训练
    criterion = nn.CrossEntropyLoss()
    
    print("\n训练 10 步...")
    for step in range(10):
        inputs = torch.randn(32, 784)
        labels = torch.randint(0, 10, (32,))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step + 1}, Loss: {loss.item():.4f}")
    
    print()
    optimizer.print_memory_stats()


def example_5_different_group_sizes():
    """示例 5: 不同分组大小的影响"""
    print("\n" + "=" * 60)
    print("示例 5: 不同分组大小的影响")
    print("=" * 60)
    
    print("\n分组大小影响:")
    print("- 较小的 group_size: 更高的精度，但更多的内存开销")
    print("- 较大的 group_size: 更低的内存开销，但可能降低精度")
    print()
    
    model_size = 100000  # 100K 参数
    
    for group_size in [64, 128, 256, 512]:
        # 计算内存使用
        num_groups = (model_size + group_size - 1) // group_size
        
        # 量化数据: 每个参数 0.5 字节（4bit）
        quantized_data_bytes = model_size // 2
        
        # scale 和 zero_point: 每个组 2 个 FP32 值
        scale_zp_bytes = num_groups * 2 * 4
        
        total_bytes = quantized_data_bytes + scale_zp_bytes
        
        print(f"Group Size = {group_size}:")
        print(f"  组数: {num_groups}")
        print(f"  量化数据: {quantized_data_bytes / 1024:.2f} KB")
        print(f"  Scale/ZP: {scale_zp_bytes / 1024:.2f} KB")
        print(f"  总计: {total_bytes / 1024:.2f} KB")
        print(f"  相比 FP32 节省: {(1 - total_bytes / (model_size * 4)) * 100:.1f}%")
        print()


def example_6_save_load_checkpoint():
    """示例 6: 保存和加载检查点"""
    print("\n" + "=" * 60)
    print("示例 6: 保存和加载检查点")
    print("=" * 60)
    
    # 创建模型和优化器
    model = SimpleClassifier(input_dim=784, hidden_dim=128, num_classes=10)
    optimizer = FourBitAdamW(model.parameters(), lr=1e-3)
    
    # 训练几步
    criterion = nn.CrossEntropyLoss()
    for _ in range(5):
        inputs = torch.randn(32, 784)
        labels = torch.randint(0, 10, (32,))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 保存检查点
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    checkpoint_path = '/tmp/4bit_optimizer_checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存到: {checkpoint_path}")
    
    # 创建新模型和优化器
    new_model = SimpleClassifier(input_dim=784, hidden_dim=128, num_classes=10)
    new_optimizer = FourBitAdamW(new_model.parameters(), lr=1e-3)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("检查点已加载")
    
    # 验证状态
    print(f"\n原始优化器学习率: {optimizer.param_groups[0]['lr']}")
    print(f"加载后优化器学习率: {new_optimizer.param_groups[0]['lr']}")
    
    # 清理
    os.remove(checkpoint_path)
    print("\n✓ 检查点保存和加载成功！")


def example_7_best_practices():
    """示例 7: 最佳实践"""
    print("\n" + "=" * 60)
    print("示例 7: 4bit 优化器最佳实践")
    print("=" * 60)
    
    print("""
1. 选择合适的 group_size:
   - 小模型 (< 1M 参数): 使用 64 或 128
   - 中等模型 (1M - 100M 参数): 使用 128 或 256
   - 大模型 (> 100M 参数): 使用 256 或 512

2. 启用误差补偿:
   - compensate_quant_error=True（默认）
   - 可以显著减少量化误差累积
   - 轻微增加计算开销

3. 学习率调整:
   - 4bit 优化器通常可以使用与标准优化器相同的学习率
   - 如果遇到训练不稳定，可以尝试略微降低学习率

4. 监控训练:
   - 使用 print_memory_stats() 监控内存使用
   - 对比标准优化器验证训练效果

5. 适用场景:
   - ✓ 大模型训练，内存受限
   - ✓ 需要训练更大模型的场景
   - ✓ 边缘设备训练
   - ✗ 对精度要求极高的任务（可能需要更小的 group_size）
""")
    
    # 展示最佳实践示例
    model = SimpleClassifier(input_dim=784, hidden_dim=256, num_classes=10)
    
    optimizer = FourBitAdamW(
        model.parameters(),
        lr=1e-3,  # 标准学习率
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        group_size=128,  # 推荐的分组大小
        compensate_quant_error=True  # 启用误差补偿
    )
    
    print("\n最佳实践配置示例:")
    print(f"  优化器: FourBitAdamW")
    print(f"  学习率: {optimizer.param_groups[0]['lr']}")
    print(f"  分组大小: {optimizer.group_size}")
    print(f"  误差补偿: {optimizer.compensate_quant_error}")


if __name__ == '__main__':
    # 运行所有示例
    example_1_basic_usage()
    example_2_memory_comparison()
    example_3_training_comparison()
    example_4_sgd_momentum()
    example_5_different_group_sizes()
    example_6_save_load_checkpoint()
    example_7_best_practices()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
