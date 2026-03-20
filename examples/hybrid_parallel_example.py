# -*- coding: utf-8 -*-
# @Time    : 2026/3/9 上午11:00
# @Author  : Jun Wang
# @File    : hybrid_parallel_example.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 3D混合并行示例

本示例展示了如何使用3D混合并行（数据并行+张量并行+流水线并行）
来训练一个深度学习模型。

3D并行配置示例:
- DP=2, TP=2, PP=2: 总共使用 2×2×2=8 个GPU
  - 数据并行: 2组
  - 张量并行: 每组2个GPU
  - 流水线并行: 每个张量并行组分为2个阶段

运行方式:
    # 使用8个GPU
    torchrun --nproc_per_node=8 hybrid_parallel_example.py
    
    # 使用4个GPU (DP=2, TP=2, PP=1)
    torchrun --nproc_per_node=4 hybrid_parallel_example.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.parallel.hybrid_parallel import HybridParallel
from parascale.utils.utils import print_rank_0, setup_logging


class Example3DModel(nn.Module):
    """
    用于3D并行示例的模型
    
    包含多个线性层和激活函数，适合展示流水线并行
    """
    def __init__(self, input_dim=3072, hidden_dim=512, output_dim=10, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        
        # 隐藏层
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


def create_dummy_dataloader(batch_size=32, num_batches=10):
    """
    创建模拟数据加载器
    
    Args:
        batch_size: 批次大小
        num_batches: 批次数量
    
    Returns:
        数据列表
    """
    data = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, 3, 32, 32)
        targets = torch.randint(0, 10, (batch_size,))
        data.append((inputs, targets))
    return data


def train_with_3d_parallel(
    rank,
    world_size,
    dp_size=2,
    tp_size=2,
    pp_size=2,
    tensor_parallel_mode="row",
    pipeline_chunks=2,
    epochs=2
):
    """
    使用3D混合并行训练模型
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        dp_size: 数据并行大小
        tp_size: 张量并行大小
        pp_size: 流水线并行大小
        tensor_parallel_mode: 张量并行模式
        pipeline_chunks: 流水线微批次数量
        epochs: 训练轮数
    """
    # 初始化分布式环境
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
    
    # 创建模型
    model = Example3DModel(num_layers=8)
    
    # 创建3D混合并行实例
    print_rank_0(f"\n初始化3D混合并行: DP={dp_size}, TP={tp_size}, PP={pp_size}")
    
    hp = HybridParallel(
        model=model,
        rank=rank,
        world_size=world_size,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        tensor_parallel_mode=tensor_parallel_mode,
        pipeline_chunks=pipeline_chunks
    )
    
    # 打印并行信息
    info = hp.get_parallel_info()
    print(f"[Rank {rank}] 并行信息:")
    print(f"  - Global Rank: {info['global_rank']}")
    print(f"  - DP Rank: {info['dp_rank']}, TP Rank: {info['tp_rank']}, PP Rank: {info['pp_rank']}")
    print(f"  - Is First Stage: {info['is_first_stage']}, Is Last Stage: {info['is_last_stage']}")
    
    # 创建优化器
    optimizer = optim.AdamW(hp.stage_layers.parameters(), lr=1e-3)
    
    # 创建模拟数据
    train_data = create_dummy_dataloader(batch_size=32, num_batches=10)
    
    # 训练循环
    print_rank_0(f"\n开始训练 {epochs} 轮...")
    
    for epoch in range(epochs):
        print_rank_0(f"\nEpoch {epoch + 1}/{epochs}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_data):
            # 将数据移动到GPU
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            # 前向传播
            outputs = hp.forward(inputs)
            
            # 计算损失和反向传播（仅在最后一个流水线阶段）
            if hp.is_last_stage and outputs is not None:
                loss = nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1
                
                if batch_idx % 2 == 0:
                    print(f"[Rank {rank}] Batch {batch_idx}, Loss: {batch_loss:.4f}")
            
            # 收集梯度（数据并行）
            hp.gather_gradients()
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
        
        # 打印epoch平均损失
        if hp.is_last_stage and num_batches > 0:
            avg_loss = epoch_loss / num_batches
            print_rank_0(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
    
    print_rank_0("\n训练完成!")
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


def demo_2d_parallel(rank, world_size):
    """
    演示2D并行配置
    
    DP+TP配置: DP=2, TP=2, PP=1
    """
    print_rank_0("\n" + "=" * 60)
    print_rank_0("演示2D并行: 数据并行 + 张量并行")
    print_rank_0("=" * 60)
    
    train_with_3d_parallel(
        rank=rank,
        world_size=world_size,
        dp_size=2,
        tp_size=2,
        pp_size=1,
        tensor_parallel_mode="row",
        epochs=1
    )


def demo_3d_parallel(rank, world_size):
    """
    演示3D并行配置
    
    DP+TP+PP配置: DP=2, TP=2, PP=2
    """
    print_rank_0("\n" + "=" * 60)
    print_rank_0("演示3D并行: 数据并行 + 张量并行 + 流水线并行")
    print_rank_0("=" * 60)
    
    train_with_3d_parallel(
        rank=rank,
        world_size=world_size,
        dp_size=2,
        tp_size=2,
        pp_size=2,
        tensor_parallel_mode="row",
        pipeline_chunks=2,
        epochs=1
    )


def demo_column_parallel(rank, world_size):
    """
    演示列并行模式
    """
    print_rank_0("\n" + "=" * 60)
    print_rank_0("演示列并行模式")
    print_rank_0("=" * 60)
    
    train_with_3d_parallel(
        rank=rank,
        world_size=world_size,
        dp_size=1,
        tp_size=2,
        pp_size=2,
        tensor_parallel_mode="column",
        pipeline_chunks=1,
        epochs=1
    )


def main():
    """主函数"""
    setup_logging(level=20)
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print_rank_0(f"\n{'=' * 60}")
    print_rank_0(f"ParaScale 3D混合并行示例")
    print_rank_0(f"{'=' * 60}")
    print_rank_0(f"总进程数: {world_size}")
    
    # 根据可用的GPU数量选择演示
    if world_size >= 8:
        # 演示3D并行
        demo_3d_parallel(rank, world_size)
        
        # 演示2D并行
        if dist.is_initialized():
            dist.destroy_process_group()
        demo_2d_parallel(rank, world_size)
        
    elif world_size >= 4:
        # 演示2D并行
        demo_2d_parallel(rank, world_size)
        
        # 演示列并行
        if dist.is_initialized():
            dist.destroy_process_group()
        demo_column_parallel(rank, world_size)
        
    elif world_size >= 2:
        # 演示基本的张量并行
        print_rank_0("\n" + "=" * 60)
        print_rank_0("演示基本张量并行")
        print_rank_0("=" * 60)
        
        train_with_3d_parallel(
            rank=rank,
            world_size=world_size,
            dp_size=1,
            tp_size=world_size,
            pp_size=1,
            tensor_parallel_mode="row",
            epochs=1
        )
    else:
        print_rank_0("\n警告: 只有1个进程，无法进行并行演示")
        print_rank_0("请使用 torchrun --nproc_per_node=N 运行，其中N >= 2")
    
    print_rank_0("\n" + "=" * 60)
    print_rank_0("示例完成!")
    print_rank_0("=" * 60)


if __name__ == '__main__':
    main()
