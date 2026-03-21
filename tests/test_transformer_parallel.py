# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_transformer_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale Transformer模型并行测试

使用Transformer模型测试TensorParallel和HybridParallel功能。

运行方式:
    torchrun --nproc_per_node=2 tests/test_transformer_parallel.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import math
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.parallel import TensorParallel, HybridParallel


# =============================================================================
# Transformer组件
# =============================================================================
class MultiHeadAttention(nn.Module):
    """多头注意力机制 - 支持张量并行"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Q, K, V投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到V
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """前馈网络 - 支持张量并行"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    """Transformer层"""
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 自注意力子层
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerModel(nn.Module):
    """完整的Transformer模型 - 适合张量并行"""
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        num_classes: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        
        # 词嵌入 + 位置编码
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 输出
        x = self.norm(x)
        x = self.fc(x)
        
        return x


# =============================================================================
# 测试结果记录
# =============================================================================
class TestResult:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.status = "pending"
        self.duration = 0.0
        self.error = None
        self.metrics = {}
        self.rank = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category,
            'status': self.status,
            'duration': self.duration,
            'error': self.error,
            'metrics': self.metrics,
            'rank': self.rank
        }


# =============================================================================
# 测试函数
# =============================================================================
def test_tensor_parallel_transformer():
    """使用Transformer测试张量并行"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    result = TestResult("TensorParallel (Transformer)", "张量并行")
    result.rank = rank
    
    try:
        # 创建Transformer模型
        model = TransformerModel(
            vocab_size=1000,
            d_model=512,
            nhead=8,
            num_layers=4,
            d_ff=2048,
            num_classes=10
        ).cuda(rank)
        
        # 创建张量并行包装器
        tp = TensorParallel(
            model,
            rank=rank,
            world_size=world_size,
            mode="column"  # 列并行更适合Transformer
        )
        
        # 创建输入数据 (batch_size=8, seq_len=32)
        inputs = torch.randint(0, 1000, (8, 32)).cuda(rank)
        
        # 前向传播
        start = time.time()
        output = tp.forward(inputs)
        forward_time = time.time() - start
        
        if output is not None:
            # 反向传播
            loss = output.sum()
            loss.backward()
        
        result.status = "passed"
        result.duration = forward_time
        result.metrics = {
            'forward_time': forward_time,
            'output_shape': list(output.shape) if output is not None else None,
            'mode': 'column',
            'model_params': sum(p.numel() for p in model.parameters())
        }
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_hybrid_parallel_transformer():
    """使用Transformer测试3D混合并行"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    result = TestResult("HybridParallel (Transformer, DP=2)", "3D混合并行")
    result.rank = rank
    
    try:
        # 创建Transformer模型
        model = TransformerModel(
            vocab_size=1000,
            d_model=512,
            nhead=8,
            num_layers=4,
            d_ff=2048,
            num_classes=10
        ).cuda(rank)
        
        # 创建3D混合并行 (DP=2, TP=1, PP=1)
        hp = HybridParallel(
            model,
            rank=rank,
            world_size=world_size,
            dp_size=2,
            tp_size=1,
            pp_size=1,
            tensor_parallel_mode="column",
            pipeline_chunks=1
        )
        
        # 创建输入数据
        inputs = torch.randint(0, 1000, (8, 32)).cuda(rank)
        
        # 前向传播
        start = time.time()
        output = hp.forward(inputs)
        forward_time = time.time() - start
        
        if output is not None:
            # 反向传播
            loss = output.sum()
            loss.backward()
            hp.gather_gradients()
        
        parallel_info = hp.get_parallel_info()
        
        result.status = "passed"
        result.duration = forward_time
        result.metrics = {
            'forward_time': forward_time,
            'dp_size': parallel_info.get('dp_size'),
            'tp_size': parallel_info.get('tp_size'),
            'pp_size': parallel_info.get('pp_size'),
            'has_output': output is not None,
            'model_params': sum(p.numel() for p in model.parameters())
        }
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def test_hybrid_parallel_3d():
    """测试完整的3D并行配置"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    result = TestResult("HybridParallel (3D: DP=1, TP=2, PP=1)", "3D混合并行")
    result.rank = rank
    
    try:
        # 对于2 GPU，使用TP=2进行张量并行
        model = TransformerModel(
            vocab_size=1000,
            d_model=512,
            nhead=8,
            num_layers=4,
            d_ff=2048,
            num_classes=10
        ).cuda(rank)
        
        # 3D配置: DP=1, TP=2, PP=1
        hp = HybridParallel(
            model,
            rank=rank,
            world_size=world_size,
            dp_size=1,
            tp_size=2,
            pp_size=1,
            tensor_parallel_mode="column",
            pipeline_chunks=1
        )
        
        inputs = torch.randint(0, 1000, (8, 32)).cuda(rank)
        
        start = time.time()
        output = hp.forward(inputs)
        forward_time = time.time() - start
        
        if output is not None:
            loss = output.sum()
            loss.backward()
            hp.gather_gradients()
        
        parallel_info = hp.get_parallel_info()
        
        result.status = "passed"
        result.duration = forward_time
        result.metrics = {
            'forward_time': forward_time,
            'dp_size': parallel_info.get('dp_size'),
            'tp_size': parallel_info.get('tp_size'),
            'pp_size': parallel_info.get('pp_size'),
            'tp_rank': parallel_info.get('tp_rank'),
            'has_output': output is not None
        }
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


# =============================================================================
# 主函数
# =============================================================================
def main():
    """主测试函数"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("=" * 80)
        print("ParaScale Transformer模型并行测试")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"World Size: {world_size}")
        print("=" * 80)
    
    dist.barrier()
    
    all_results = []
    
    # 1. 测试TensorParallel with Transformer
    if rank == 0:
        print("\n【测试1: TensorParallel (Transformer模型)】")
    dist.barrier()
    result = test_tensor_parallel_transformer()
    all_results.append(result.to_dict())
    
    # 2. 测试HybridParallel with Transformer (DP=2)
    if rank == 0:
        print("\n【测试2: HybridParallel (Transformer, DP=2)】")
    dist.barrier()
    result = test_hybrid_parallel_transformer()
    all_results.append(result.to_dict())
    
    # 3. 测试3D并行 (TP=2)
    if rank == 0:
        print("\n【测试3: HybridParallel (3D: DP=1, TP=2, PP=1)】")
    dist.barrier()
    result = test_hybrid_parallel_3d()
    all_results.append(result.to_dict())
    
    dist.barrier()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("生成测试报告...")
        print("=" * 80)
        
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r['status'] == "passed")
        failed_tests = sum(1 for r in all_results if r['status'] == "failed")
        
        categories = {}
        for r in all_results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        print(f"\n总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✓")
        print(f"失败: {failed_tests} ✗")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")
        print()
        
        for cat, results in sorted(categories.items()):
            print(f"\n【{cat}】")
            for r in results:
                status = "✓" if r['status'] == "passed" else "✗"
                print(f"  {status} {r['name']}: {r['status']} ({r['duration']:.3f}s)")
                if r['metrics']:
                    for key, value in r['metrics'].items():
                        if isinstance(value, float):
                            print(f"      {key}: {value:.4f}")
                        else:
                            print(f"      {key}: {value}")
                if r['error']:
                    print(f"      Error: {r['error']}")
        
        print("\n" + "=" * 80)
        print("测试报告结束")
        print("=" * 80)
        
        report_file = f"/wangjun/ParaScale/transformer_parallel_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'world_size': world_size,
                'results': all_results
            }, f, indent=2)
        print(f"\n报告已保存: {report_file}")
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
