# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : zero_optimizer_example.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale ZeRO优化器使用示例

本示例展示如何使用ZeRO (Zero Redundancy Optimizer) 优化器
来减少大规模模型训练中的内存使用。

ZeRO的三个阶段:
- Stage 1: 分片优化器状态 (4x内存节省)
- Stage 2: 分片优化器状态 + 梯度 (8x内存节省)
- Stage 3: 分片优化器状态 + 梯度 + 参数 (与数据并行度线性相关)

使用方法:
    python zero_optimizer_example.py
    torchrun --nproc_per_node=2 zero_optimizer_example.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.optimizers.zero_optimizer import ZeroOptimizer, ZeroAdamW, ZeroSGD, ZeroStage


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, hidden_size=1000):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def example_1_stage0_no_zero():
    """
    示例 1: 不使用ZeRO (Stage 0)
    
    作为基线对比，展示标准优化器的内存使用。
    """
    print("\n" + "="*60)
    print("示例 1: 不使用ZeRO (Stage 0)")
    print("="*60)
    
    model = SimpleModel()
    
    # 创建ZeRO优化器 (Stage 0 = 禁用ZeRO)
    optimizer = ZeroAdamW(
        model,
        lr=1e-3,
        stage=ZeroStage.DISABLED
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"ZeRO阶段: {optimizer.stage}")
    
    # 打印内存统计
    optimizer.print_memory_stats()
    
    # 模拟训练步骤
    x = torch.randn(4, 1000)
    target = torch.randint(0, 10, (4,))
    
    output = model(x)
    loss = nn.functional.cross_entropy(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"训练步骤完成，Loss: {loss.item():.4f}")


def example_2_stage1_optimizer_states():
    """
    示例 2: ZeRO Stage 1 - 分片优化器状态
    
    只分片优化器状态（如Adam的momentum和variance），
    可以节省约4倍内存（与数据并行度成正比）。
    
    注意: 需要分布式环境才能实际分片
    """
    print("\n" + "="*60)
    print("示例 2: ZeRO Stage 1 - 分片优化器状态")
    print("="*60)
    
    model = SimpleModel()
    
    try:
        # 尝试创建Stage 1优化器
        optimizer = ZeroAdamW(
            model,
            lr=1e-3,
            stage=ZeroStage.OPTIMIZER_STATES
        )
        print("ZeRO Stage 1 优化器创建成功")
        optimizer.print_memory_stats()
    except RuntimeError as e:
        print(f"注意: {e}")
        print("在非分布式环境下，使用Stage 0作为回退")
        
        optimizer = ZeroAdamW(
            model,
            lr=1e-3,
            stage=ZeroStage.DISABLED
        )
        optimizer.print_memory_stats()


def example_3_stage2_gradients():
    """
    示例 3: ZeRO Stage 2 - 分片优化器状态 + 梯度
    
    在Stage 1基础上增加梯度分片，可以节省约8倍内存。
    """
    print("\n" + "="*60)
    print("示例 3: ZeRO Stage 2 - 分片优化器状态 + 梯度")
    print("="*60)
    
    model = SimpleModel()
    
    try:
        optimizer = ZeroAdamW(
            model,
            lr=1e-3,
            stage=ZeroStage.GRADIENTS
        )
        print("ZeRO Stage 2 优化器创建成功")
        optimizer.print_memory_stats()
    except RuntimeError as e:
        print(f"注意: {e}")
        print("在非分布式环境下，使用Stage 0作为回退")


def example_4_stage3_parameters():
    """
    示例 4: ZeRO Stage 3 - 分片优化器状态 + 梯度 + 参数
    
    完全分片，每个rank只存储部分参数。
    内存节省与数据并行度线性相关。
    """
    print("\n" + "="*60)
    print("示例 4: ZeRO Stage 3 - 完全分片")
    print("="*60)
    
    model = SimpleModel()
    
    try:
        optimizer = ZeroAdamW(
            model,
            lr=1e-3,
            stage=ZeroStage.PARAMETERS
        )
        print("ZeRO Stage 3 优化器创建成功")
        optimizer.print_memory_stats()
    except RuntimeError as e:
        print(f"注意: {e}")
        print("在非分布式环境下，使用Stage 0作为回退")


def example_5_with_offload():
    """
    示例 5: 使用CPU Offload
    
    将优化器状态卸载到CPU，进一步减少GPU内存使用。
    """
    print("\n" + "="*60)
    print("示例 5: 使用CPU Offload")
    print("="*60)
    
    model = SimpleModel()
    
    try:
        optimizer = ZeroAdamW(
            model,
            lr=1e-3,
            stage=ZeroStage.OPTIMIZER_STATES,
            offload_optimizer=True  # 启用CPU Offload
        )
        print("ZeRO + CPU Offload 优化器创建成功")
        print(f"Offload优化器状态: {optimizer.offload_optimizer}")
        optimizer.print_memory_stats()
    except RuntimeError as e:
        print(f"注意: {e}")


def example_6_zero_sgd():
    """
    示例 6: 使用ZeRO SGD
    
    ZeRO也可以与SGD优化器结合使用。
    """
    print("\n" + "="*60)
    print("示例 6: 使用ZeRO SGD")
    print("="*60)
    
    model = SimpleModel()
    
    try:
        optimizer = ZeroSGD(
            model,
            lr=0.01,
            momentum=0.9,
            stage=ZeroStage.GRADIENTS
        )
        print("ZeRO SGD优化器创建成功")
        optimizer.print_memory_stats()
        
        # 训练步骤
        x = torch.randn(4, 1000)
        target = torch.randint(0, 10, (4,))
        
        for i in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 2 == 0:
                print(f"Step {i+1}, Loss: {loss.item():.4f}")
        
    except RuntimeError as e:
        print(f"注意: {e}")


def example_7_memory_comparison():
    """
    示例 7: 内存使用对比
    
    对比不同ZeRO阶段的理论内存使用。
    """
    print("\n" + "="*60)
    print("示例 7: 不同ZeRO阶段的内存对比")
    print("="*60)
    
    model = SimpleModel(hidden_size=2000)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n模型参数量: {total_params:,}")
    print(f"数据并行度: 4 (假设)\n")
    
    # 标准AdamW内存
    param_memory = total_params * 4 / (1024**2)  # MB
    grad_memory = total_params * 4 / (1024**2)
    optimizer_memory = total_params * 4 * 2 / (1024**2)  # momentum + variance
    total_standard = param_memory + grad_memory + optimizer_memory
    
    print("标准 AdamW (无ZeRO):")
    print(f"  参数内存:     {param_memory:8.2f} MB")
    print(f"  梯度内存:     {grad_memory:8.2f} MB")
    print(f"  优化器状态:   {optimizer_memory:8.2f} MB")
    print(f"  总计:         {total_standard:8.2f} MB")
    
    print("\nZeRO Stage 1 (分片优化器状态):")
    print(f"  参数内存:     {param_memory:8.2f} MB")
    print(f"  梯度内存:     {grad_memory:8.2f} MB")
    print(f"  优化器状态:   {optimizer_memory/4:8.2f} MB (分片到4个rank)")
    print(f"  总计:         {param_memory + grad_memory + optimizer_memory/4:8.2f} MB")
    print(f"  节省:         {total_standard / (param_memory + grad_memory + optimizer_memory/4):8.2f}x")
    
    print("\nZeRO Stage 2 (分片优化器状态 + 梯度):")
    print(f"  参数内存:     {param_memory:8.2f} MB")
    print(f"  梯度内存:     {grad_memory/4:8.2f} MB (分片到4个rank)")
    print(f"  优化器状态:   {optimizer_memory/4:8.2f} MB (分片到4个rank)")
    print(f"  总计:         {param_memory + grad_memory/4 + optimizer_memory/4:8.2f} MB")
    print(f"  节省:         {total_standard / (param_memory + grad_memory/4 + optimizer_memory/4):8.2f}x")
    
    print("\nZeRO Stage 3 (完全分片):")
    print(f"  参数内存:     {param_memory/4:8.2f} MB (分片到4个rank)")
    print(f"  梯度内存:     {grad_memory/4:8.2f} MB (分片到4个rank)")
    print(f"  优化器状态:   {optimizer_memory/4:8.2f} MB (分片到4个rank)")
    print(f"  总计:         {(param_memory + grad_memory + optimizer_memory)/4:8.2f} MB")
    print(f"  节省:         4.00x")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("ParaScale ZeRO优化器示例集")
    print("="*60)
    print("""
本示例集展示ZeRO (Zero Redundancy Optimizer) 的使用方法。

ZeRO通过分片技术减少分布式训练中的内存冗余：
- Stage 1: 分片优化器状态
- Stage 2: 分片优化器状态 + 梯度
- Stage 3: 分片优化器状态 + 梯度 + 参数

使用方法:
  python zero_optimizer_example.py
  torchrun --nproc_per_node=4 zero_optimizer_example.py
""")
    
    # 运行所有示例
    example_1_stage0_no_zero()
    example_2_stage1_optimizer_states()
    example_3_stage2_gradients()
    example_4_stage3_parameters()
    example_5_with_offload()
    example_6_zero_sgd()
    example_7_memory_comparison()
    
    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60)
    print("""
提示:
1. 在非分布式环境下，ZeRO Stage 1+ 会回退到 Stage 0
2. 使用 torchrun 启动可以体验真正的ZeRO分片效果
3. Stage 3 需要配合分布式数据并行使用

示例命令:
  torchrun --nproc_per_node=4 zero_optimizer_example.py
""")


if __name__ == '__main__':
    main()
