# -*- coding: utf-8 -*-
# @Time    : 2026/3/23
# @Author  : Jun Wang
# @File    : sequence_parallel_example.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 序列并行使用示例

本示例展示如何在ParaScale中使用序列并行（Sequence Parallelism）来训练大模型。

序列并行优势:
    - 减少LayerNorm、Dropout等层的激活内存占用
    - 与张量并行结合使用，进一步优化内存效率
    - 支持超长序列训练（结合Ulysses Attention）

运行方式:
    # 单GPU测试
    python examples/sequence_parallel_example.py
    
    # 多GPU测试（4卡）
    torchrun --nproc_per_node=4 examples/sequence_parallel_example.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parascale.parallel import (
    SequenceParallel,
    SequenceParallelConfig,
    SequenceParallelMode,
    enable_sequence_parallel,
)
from parascale.utils import initialize_distributed, get_rank, get_world_size


class TransformerModel(nn.Module):
    """示例Transformer模型"""
    
    def __init__(self, vocab_size=10000, hidden_size=512, num_layers=6, num_heads=8, max_seq_len=2048):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 位置编码
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        
        # 嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 输出
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


class TransformerLayer(nn.Module):
    """单个Transformer层"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # QKV投影
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 输出投影
        return self.out_proj(out)


def create_dummy_data(batch_size, seq_len, vocab_size, device):
    """创建虚拟训练数据"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, labels


def train_with_sequence_parallel():
    """使用序列并行训练"""
    
    # 初始化分布式环境
    rank, world_size, local_rank = initialize_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Rank {rank}/{world_size} 开始训练")
    print(f"{'='*60}")
    
    # 模型配置
    vocab_size = 10000
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    max_seq_len = 2048
    
    # 训练配置
    batch_size = 4
    seq_len = 512
    learning_rate = 1e-4
    num_steps = 10
    
    print(f"\n模型配置:")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Num Layers: {num_layers}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Max Seq Len: {max_seq_len}")
    
    print(f"\n训练配置:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Seq Len: {seq_len}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Num Steps: {num_steps}")
    
    # 创建模型
    print(f"\n[Rank {rank}] 创建模型...")
    model = TransformerModel(vocab_size, hidden_size, num_layers, num_heads, max_seq_len)
    model = model.to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 配置序列并行
    # 在4卡环境下，使用 sp_size=2, tp_size=2
    sp_size = min(2, world_size)
    tp_size = world_size // sp_size if world_size > 1 else 1
    
    print(f"\n[Rank {rank}] 配置序列并行...")
    print(f"  SP Size: {sp_size}")
    print(f"  TP Size: {tp_size}")
    
    config = SequenceParallelConfig(
        sp_size=sp_size,
        tp_size=tp_size,
        mode=SequenceParallelMode.STANDARD,
        scatter_input=True,
        gather_output=True,
        enable_for_layernorm=True,
        enable_for_dropout=True,
    )
    
    # 创建序列并行包装器
    sp_model = SequenceParallel(
        model,
        rank=rank,
        world_size=world_size,
        config=config,
    )
    
    # 获取内存优化统计
    memory_stats = sp_model.get_memory_stats()
    if rank == 0:
        print(f"\n内存优化统计:")
        for key, value in memory_stats.items():
            print(f"  {key}: {value}x")
    
    # 创建优化器
    optimizer = optim.AdamW(sp_model.model.parameters(), lr=learning_rate)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print(f"\n[Rank {rank}] 开始训练...")
    sp_model.model.train()
    
    for step in range(num_steps):
        # 创建虚拟数据
        input_ids, labels = create_dummy_data(batch_size, seq_len, vocab_size, device)
        
        # 前向传播
        outputs = sp_model(input_ids)
        
        # 计算损失
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 收集所有rank的损失（仅用于显示）
        if world_size > 1:
            loss_tensor = torch.tensor([loss.item()], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_value = loss_tensor.item()
        else:
            loss_value = loss.item()
        
        if rank == 0 and (step + 1) % 2 == 0:
            print(f"  Step {step+1}/{num_steps}, Loss: {loss_value:.4f}")
    
    print(f"\n[Rank {rank}] 训练完成!")
    
    # 获取并行信息
    parallel_info = sp_model.get_parallel_info()
    if rank == 0:
        print(f"\n并行信息:")
        for key, value in parallel_info.items():
            print(f"  {key}: {value}")
    
    return sp_model


def demo_memory_saving():
    """演示序列并行的内存节省效果"""
    
    print("\n" + "="*60)
    print("序列并行内存节省演示")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型配置
    hidden_size = 1024
    seq_len = 2048
    batch_size = 4
    
    print(f"\n配置:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Seq Len: {seq_len}")
    print(f"  Hidden Size: {hidden_size}")
    
    # 计算LayerNorm的激活内存
    layernorm_activation = batch_size * seq_len * hidden_size * 4  # float32
    
    print(f"\n标准训练 (无序列并行):")
    print(f"  LayerNorm激活内存: {layernorm_activation / (1024**2):.2f} MB")
    
    # 不同序列并行大小下的内存
    sp_sizes = [2, 4, 8]
    
    print(f"\n使用序列并行:")
    for sp_size in sp_sizes:
        reduced_memory = layernorm_activation / sp_size
        print(f"  SP Size={sp_size}: {reduced_memory / (1024**2):.2f} MB ({sp_size}x 节省)")
    
    # Dropout的内存节省
    dropout_mask = batch_size * seq_len * hidden_size  # 1 byte per element (bool)
    
    print(f"\nDropout掩码内存:")
    print(f"  标准: {dropout_mask / (1024**2):.2f} MB")
    for sp_size in sp_sizes:
        reduced = dropout_mask / sp_size
        print(f"  SP Size={sp_size}: {reduced / (1024**2):.2f} MB")


def compare_with_without_sp():
    """对比使用和不使用序列并行的效果"""
    
    print("\n" + "="*60)
    print("序列并行效果对比")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简单模型
    model = nn.Sequential(
        nn.LayerNorm(512),
        nn.Linear(512, 512),
        nn.Dropout(0.1),
        nn.LayerNorm(512),
        nn.Linear(512, 512),
    ).to(device)
    
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    print("\n1. 标准模型（无序列并行）:")
    print(f"   输入形状: {x.shape}")
    
    # 标准前向传播
    model.eval()
    with torch.no_grad():
        out_standard = model(x)
    print(f"   输出形状: {out_standard.shape}")
    
    # 统计LayerNorm和Dropout数量
    ln_count = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    dropout_count = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    print(f"   LayerNorm数量: {ln_count}")
    print(f"   Dropout数量: {dropout_count}")
    
    print("\n2. 序列并行模型（SP Size=2，单GPU模拟）:")
    
    # 创建序列并行版本（强制sp_size=2以演示转换效果）
    from parascale.parallel.sequence_parallel import SequenceParallelConverter, SequenceParallelConfig
    
    sp_config = SequenceParallelConfig(
        sp_size=2,  # 设置为2以触发模型转换
        enable_for_layernorm=True,
        enable_for_dropout=True,
    )
    
    # 先转换模型
    converted_model = SequenceParallelConverter.convert_model(model, sp_config, sp_group=None)
    
    sp_model = SequenceParallel(
        converted_model,
        rank=0,
        world_size=1,
        sp_size=1,  # 单GPU运行时仍使用1
        tp_size=1,
    )
    
    sp_model.model.eval()
    with torch.no_grad():
        out_sp = sp_model(x)
    
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out_sp.shape}")
    print(f"   输出匹配: {torch.allclose(out_standard, out_sp, atol=1e-6)}")
    
    # 统计替换后的层
    from parascale.parallel.sequence_parallel import SequenceParallelLayerNorm, SequenceParallelDropout
    sp_ln_count = sum(1 for m in sp_model.model.modules() if isinstance(m, SequenceParallelLayerNorm))
    sp_dropout_count = sum(1 for m in sp_model.model.modules() if isinstance(m, SequenceParallelDropout))
    print(f"   序列并行LayerNorm数量: {sp_ln_count}")
    print(f"   序列并行Dropout数量: {sp_dropout_count}")


def main():
    """主函数"""
    
    print("\n" + "="*60)
    print("ParaScale 序列并行示例")
    print("="*60)
    
    # 演示1: 内存节省效果
    demo_memory_saving()
    
    # 演示2: 对比效果
    compare_with_without_sp()
    
    # 演示3: 训练（如果有GPU）
    if torch.cuda.is_available():
        try:
            train_with_sequence_parallel()
        except Exception as e:
            print(f"\n训练演示出错: {e}")
            print("这可能是因为分布式环境未正确初始化")
    else:
        print("\n跳过训练演示（需要GPU）")
    
    print("\n" + "="*60)
    print("示例完成!")
    print("="*60)


if __name__ == '__main__':
    main()
