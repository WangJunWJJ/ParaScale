# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_refactored_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 重构后并行策略测试脚本

本脚本用于验证重构后的 TensorParallel 和 HybridParallel 实现。

运行方式:
    # 单进程测试
    python tests/test_refactored_parallel.py
    
    # 多进程测试 (2 GPUs)
    torchrun --nproc_per_node=2 tests/test_refactored_parallel.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import os
import sys
import time
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.parallel import (
    TensorParallel,
    TensorParallelConfig,
    HybridParallel,
    HybridParallelConfig,
    ParallelStrategy,
    PipelineSchedule,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    ParallelSelfAttention,
    ParallelMLP,
)


# =============================================================================
# 测试模型
# =============================================================================

class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.fc2(torch.relu(self.fc1(x)))
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, vocab_size=1000, hidden_size=512, num_layers=4, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


# =============================================================================
# 测试函数
# =============================================================================

def test_imports():
    """测试导入"""
    print("\n" + "="*80)
    print("测试1: 模块导入")
    print("="*80)
    
    try:
        # 测试所有导入
        assert TensorParallel is not None
        assert TensorParallelConfig is not None
        assert HybridParallel is not None
        assert HybridParallelConfig is not None
        assert ParallelStrategy is not None
        assert PipelineSchedule is not None
        assert ColumnParallelLinear is not None
        assert RowParallelLinear is not None
        assert VocabParallelEmbedding is not None
        assert ParallelSelfAttention is not None
        assert ParallelMLP is not None
        
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_tensor_parallel_config():
    """测试 TensorParallel 配置"""
    print("\n" + "="*80)
    print("测试2: TensorParallel 配置")
    print("="*80)
    
    try:
        # 测试默认配置
        config = TensorParallelConfig()
        assert config.tp_size == 1
        assert config.strategy == ParallelStrategy.AUTO
        assert config.auto_detect == True
        print("✓ 默认配置创建成功")
        
        # 测试自定义配置
        config = TensorParallelConfig(
            tp_size=2,
            strategy=ParallelStrategy.TRANSFORMER,
            layer_config={"fc1": "column", "fc2": "row"},
            auto_detect=False,
        )
        assert config.tp_size == 2
        assert config.strategy == ParallelStrategy.TRANSFORMER
        print("✓ 自定义配置创建成功")
        
        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False


def test_hybrid_parallel_config():
    """测试 HybridParallel 配置"""
    print("\n" + "="*80)
    print("测试3: HybridParallel 配置")
    print("="*80)
    
    try:
        # 测试默认配置
        config = HybridParallelConfig()
        assert config.dp_size == 1
        assert config.tp_size == 1
        assert config.pp_size == 1
        assert config.schedule == PipelineSchedule.ONE_FORWARD_ONE_BACKWARD
        print("✓ 默认配置创建成功")
        
        # 测试自定义配置
        config = HybridParallelConfig(
            dp_size=2,
            tp_size=2,
            pp_size=2,
            num_micro_batches=8,
        )
        assert config.dp_size == 2
        assert config.tp_size == 2
        assert config.pp_size == 2
        assert config.num_micro_batches == 8
        print("✓ 自定义配置创建成功")
        
        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False


def test_parallel_layers():
    """测试并行层组件"""
    print("\n" + "="*80)
    print("测试4: 并行层组件")
    print("="*80)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 测试 ColumnParallelLinear
        col_linear = ColumnParallelLinear(512, 1024).to(device)
        x = torch.randn(2, 512, device=device)
        out = col_linear(x)
        assert out.shape == (2, 1024)
        print("✓ ColumnParallelLinear 测试通过")
        
        # 测试 RowParallelLinear
        row_linear = RowParallelLinear(1024, 512).to(device)
        x = torch.randn(2, 1024, device=device)
        out = row_linear(x)
        assert out.shape == (2, 512)
        print("✓ RowParallelLinear 测试通过")
        
        # 测试 VocabParallelEmbedding
        vocab_embed = VocabParallelEmbedding(1000, 512).to(device)
        x = torch.randint(0, 1000, (2, 10), device=device)
        out = vocab_embed(x)
        assert out.shape == (2, 10, 512)
        print("✓ VocabParallelEmbedding 测试通过")
        
        # 测试 ParallelSelfAttention
        attn = ParallelSelfAttention(512, 8).to(device)
        x = torch.randn(2, 10, 512, device=device)
        out = attn(x)
        assert out.shape == (2, 10, 512)
        print("✓ ParallelSelfAttention 测试通过")
        
        # 测试 ParallelMLP
        mlp = ParallelMLP(512, 2048).to(device)
        x = torch.randn(2, 10, 512, device=device)
        out = mlp(x)
        assert out.shape == (2, 10, 512)
        print("✓ ParallelMLP 测试通过")
        
        return True
    except Exception as e:
        print(f"✗ 并行层测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tensor_parallel_single_gpu():
    """测试单 GPU TensorParallel"""
    print("\n" + "="*80)
    print("测试5: TensorParallel 单GPU测试 (tp_size=1)")
    print("="*80)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = SimpleModel().to(device)
        tp = TensorParallel(
            model,
            rank=0,
            world_size=1,
            tp_size=1,  # 单GPU，不使用张量并行
        )
        
        # 测试前向传播
        x = torch.randn(4, 784, device=device)
        out = tp(x)
        assert out.shape == (4, 10)
        print("✓ TensorParallel 前向传播测试通过")
        
        # 测试反向传播
        loss = out.sum()
        loss.backward()
        print("✓ TensorParallel 反向传播测试通过")
        
        # 测试获取信息
        info = tp.get_parallel_info()
        assert info["tp_size"] == 1
        print(f"✓ 并行信息: {info}")
        
        return True
    except Exception as e:
        print(f"✗ TensorParallel 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_parallel_single_gpu():
    """测试单 GPU HybridParallel"""
    print("\n" + "="*80)
    print("测试6: HybridParallel 单GPU测试 (dp=tp=pp=1)")
    print("="*80)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = SimpleModel().to(device)
        hp = HybridParallel(
            model,
            rank=0,
            world_size=1,
            dp_size=1,
            tp_size=1,
            pp_size=1,
        )
        
        # 测试前向传播
        x = torch.randn(4, 784, device=device)
        out = hp(x)
        assert out.shape == (4, 10)
        print("✓ HybridParallel 前向传播测试通过")
        
        # 测试反向传播
        loss = out.sum()
        loss.backward()
        print("✓ HybridParallel 反向传播测试通过")
        
        # 测试获取信息
        info = hp.get_parallel_info()
        assert info["dp_size"] == 1
        assert info["tp_size"] == 1
        assert info["pp_size"] == 1
        print(f"✓ 并行信息: {info}")
        
        return True
    except Exception as e:
        print(f"✗ HybridParallel 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_model():
    """测试 Transformer 模型"""
    print("\n" + "="*80)
    print("测试7: Transformer 模型测试")
    print("="*80)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = TransformerModel(vocab_size=1000, hidden_size=512, num_layers=4).to(device)
        
        # 测试 TensorParallel
        tp = TensorParallel(
            model,
            rank=0,
            world_size=1,
            tp_size=1,
        )
        
        x = torch.randint(0, 1000, (2, 20), device=device)
        out = tp(x)
        assert out.shape == (2, 20, 1000)
        print("✓ Transformer + TensorParallel 测试通过")
        
        # 测试 HybridParallel
        model = TransformerModel(vocab_size=1000, hidden_size=512, num_layers=4).to(device)
        hp = HybridParallel(
            model,
            rank=0,
            world_size=1,
            dp_size=1,
            tp_size=1,
            pp_size=1,
        )
        
        out = hp(x)
        assert out.shape == (2, 20, 1000)
        print("✓ Transformer + HybridParallel 测试通过")
        
        return True
    except Exception as e:
        print(f"✗ Transformer 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distributed():
    """测试分布式环境"""
    if not dist.is_initialized():
        print("\n" + "="*80)
        print("测试8: 分布式测试 (跳过 - 非分布式环境)")
        print("="*80)
        return True
    
    print("\n" + "="*80)
    print("测试8: 分布式测试")
    print("="*80)
    
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        print(f"Rank {rank}/{world_size} 开始测试")
        
        # 测试 TensorParallel
        if world_size >= 2:
            model = SimpleModel().to(device)
            tp = TensorParallel(
                model,
                rank=rank,
                world_size=world_size,
                tp_size=world_size,
            )
            
            x = torch.randn(4, 784, device=device)
            out = tp(x)
            print(f"✓ Rank {rank} TensorParallel 测试通过")
            
            dist.barrier()
        
        # 测试 HybridParallel
        if world_size >= 2 and world_size % 2 == 0:
            model = SimpleModel().to(device)
            hp = HybridParallel(
                model,
                rank=rank,
                world_size=world_size,
                dp_size=2,
                tp_size=world_size // 2,
                pp_size=1,
            )
            
            x = torch.randn(4, 784, device=device)
            out = hp(x)
            print(f"✓ Rank {rank} HybridParallel 测试通过")
            
            dist.barrier()
        
        return True
    except Exception as e:
        print(f"✗ 分布式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("\n" + "="*80)
    print("ParaScale 重构后并行策略测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 记录测试结果
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("TensorParallel 配置", test_tensor_parallel_config()))
    results.append(("HybridParallel 配置", test_hybrid_parallel_config()))
    results.append(("并行层组件", test_parallel_layers()))
    results.append(("TensorParallel 单GPU", test_tensor_parallel_single_gpu()))
    results.append(("HybridParallel 单GPU", test_hybrid_parallel_single_gpu()))
    results.append(("Transformer 模型", test_transformer_model()))
    results.append(("分布式测试", test_distributed()))
    
    # 打印测试报告
    print("\n" + "="*80)
    print("测试报告")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print("\n" + "="*80)
    print(f"总计: {len(results)} 个测试")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"通过率: {passed/len(results)*100:.1f}%")
    print("="*80)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
