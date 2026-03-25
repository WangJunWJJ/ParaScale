# -*- coding: utf-8 -*-
# @Time    : 2026/3/23
# @Author  : Jun Wang
# @File    : test_sequence_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 序列并行测试模块

本模块包含序列并行的单元测试和集成测试，验证：
1. 序列并行层（LayerNorm、Dropout）的正确性
2. 张量切分和收集的功能
3. 与张量并行的集成
4. 内存节省效果

测试环境:
    - 单GPU: 测试基础功能
    - 多GPU: 测试分布式功能（需要 torchrun 启动）

使用方法:
    # 单GPU测试
    python tests/test_sequence_parallel.py
    
    # 多GPU测试（4卡）
    torchrun --nproc_per_node=4 tests/test_sequence_parallel.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import unittest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parascale.parallel.sequence_parallel import (
    SequenceParallel,
    SequenceParallelConfig,
    SequenceParallelLayerNorm,
    SequenceParallelDropout,
    SequenceParallelLinear,
    SequenceParallelMode,
    _split_tensor_along_dim,
    _gather_tensor_along_dim,
    _ScatterToSequenceParallelRegion,
    _GatherFromSequenceParallelRegion,
    enable_sequence_parallel,
)
from parascale.parallel import TensorParallel


class SimpleTransformerBlock(nn.Module):
    """简单的Transformer块用于测试"""
    
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        
        # LayerNorm
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # 简化的Attention
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Attention with residual
        residual = x
        x = self.ln1(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 简化attention计算
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = self.out_proj(attn_out)
        x = residual + self.dropout(attn_out)
        
        # MLP with residual
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class TestSequenceParallelLayers(unittest.TestCase):
    """测试序列并行层"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.seq_len = 128
        self.hidden_size = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_sequence_parallel_layernorm(self):
        """测试序列并行LayerNorm"""
        print("\n测试序列并行LayerNorm...")
        
        # 创建标准LayerNorm
        standard_ln = nn.LayerNorm(self.hidden_size).to(self.device)
        
        # 创建序列并行LayerNorm
        sp_ln = SequenceParallelLayerNorm(self.hidden_size).to(self.device)
        
        # 复制权重
        sp_ln.weight.data = standard_ln.weight.data.clone()
        sp_ln.bias.data = standard_ln.bias.data.clone()
        
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # 前向传播
        standard_out = standard_ln(x)
        sp_out = sp_ln(x)
        
        # 验证输出相同
        self.assertTrue(torch.allclose(standard_out, sp_out, atol=1e-6))
        print("✓ LayerNorm输出匹配")
        
        # 验证梯度
        loss_standard = standard_out.sum()
        loss_sp = sp_out.sum()
        
        loss_standard.backward()
        loss_sp.backward()
        
        self.assertTrue(torch.allclose(standard_ln.weight.grad, sp_ln.weight.grad, atol=1e-6))
        print("✓ LayerNorm梯度匹配")
    
    def test_sequence_parallel_dropout(self):
        """测试序列并行Dropout"""
        print("\n测试序列并行Dropout...")
        
        # 创建Dropout
        sp_dropout = SequenceParallelDropout(p=0.1).to(self.device)
        sp_dropout.eval()  # 评估模式，dropout不生效
        
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # 前向传播（eval模式应返回相同结果）
        out = sp_dropout(x)
        
        # 验证输出相同（eval模式）
        self.assertTrue(torch.allclose(x, out, atol=1e-6))
        print("✓ Dropout在eval模式下输出匹配")
    
    def test_sequence_parallel_linear(self):
        """测试序列并行Linear"""
        print("\n测试序列并行Linear...")
        
        in_features = self.hidden_size
        out_features = 256
        
        # 创建标准Linear
        standard_linear = nn.Linear(in_features, out_features).to(self.device)
        
        # 创建序列并行Linear（不收集输出）
        sp_linear = SequenceParallelLinear(
            in_features, out_features, gather_output=False
        ).to(self.device)
        
        # 复制权重
        sp_linear.weight.data = standard_linear.weight.data.clone()
        sp_linear.bias.data = standard_linear.bias.data.clone()
        
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, in_features, device=self.device)
        
        # 前向传播
        standard_out = standard_linear(x)
        sp_out = sp_linear(x)
        
        # 验证输出相同
        self.assertTrue(torch.allclose(standard_out, sp_out, atol=1e-6))
        print("✓ Linear输出匹配")
    
    def test_tensor_split_and_gather(self):
        """测试张量切分和收集"""
        print("\n测试张量切分和收集...")
        
        # 创建测试张量
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        # 模拟4路序列并行
        sp_size = 4
        dim = 1  # 序列维度
        
        # 切分张量
        chunks = []
        for sp_rank in range(sp_size):
            chunk = _split_tensor_along_dim(x, dim, sp_size, sp_rank)
            chunks.append(chunk)
            
            # 验证切分大小
            expected_size = self.seq_len // sp_size
            self.assertEqual(chunk.size(dim), expected_size)
        
        print(f"✓ 张量切分成功，每块大小: {chunks[0].shape}")
        
        # 收集张量（模拟，非分布式环境）
        gathered = torch.cat(chunks, dim=dim)
        
        # 验证收集后恢复原状
        self.assertTrue(torch.allclose(x, gathered, atol=1e-6))
        print("✓ 张量收集后恢复原状")


class TestSequenceParallelIntegration(unittest.TestCase):
    """测试序列并行集成"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.seq_len = 128
        self.hidden_size = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_sequence_parallel_single_gpu(self):
        """测试单GPU序列并行（sp_size=1）"""
        print("\n测试单GPU序列并行...")
        
        # 创建简单模型
        model = SimpleTransformerBlock(self.hidden_size).to(self.device)
        
        # 创建序列并行（sp_size=1，不实际切分）
        sp = SequenceParallel(
            model,
            rank=0,
            world_size=1,
            sp_size=1,
            tp_size=1,
        )
        
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # 前向传播
        out = sp(x)
        
        # 验证输出形状
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        print(f"✓ 单GPU序列并行输出形状正确: {out.shape}")
        
        # 验证梯度
        loss = out.sum()
        loss.backward()
        
        # 检查梯度是否存在
        has_grad = any(p.grad is not None for p in sp.model.parameters())
        self.assertTrue(has_grad)
        print("✓ 梯度计算正常")
    
    def test_sequence_parallel_config(self):
        """测试序列并行配置"""
        print("\n测试序列并行配置...")
        
        config = SequenceParallelConfig(
            sp_size=4,
            tp_size=2,
            mode=SequenceParallelMode.STANDARD,
            scatter_input=True,
            gather_output=True,
            enable_for_layernorm=True,
            enable_for_dropout=True,
        )
        
        self.assertEqual(config.sp_size, 4)
        self.assertEqual(config.tp_size, 2)
        self.assertEqual(config.mode, SequenceParallelMode.STANDARD)
        print("✓ 配置对象创建成功")
    
    def test_model_conversion(self):
        """测试模型转换"""
        print("\n测试模型转换...")
        
        # 创建模型
        model = SimpleTransformerBlock(self.hidden_size).to(self.device)
        
        # 记录原始LayerNorm数量
        original_ln_count = sum(1 for _ in model.modules() if isinstance(_, nn.LayerNorm))
        
        # 创建配置
        config = SequenceParallelConfig(
            sp_size=4,
            enable_for_layernorm=True,
            enable_for_dropout=True,
        )
        
        # 转换模型
        from parascale.parallel.sequence_parallel import SequenceParallelConverter
        converted_model = SequenceParallelConverter.convert_model(model, config, sp_group=None)
        
        # 验证LayerNorm被替换
        sp_ln_count = sum(1 for _ in converted_model.modules() if isinstance(_, SequenceParallelLayerNorm))
        self.assertEqual(sp_ln_count, original_ln_count)
        print(f"✓ 模型转换成功，替换了 {sp_ln_count} 个LayerNorm")
        
        # 验证Dropout被替换
        sp_dropout_count = sum(1 for _ in converted_model.modules() if isinstance(_, SequenceParallelDropout))
        self.assertGreater(sp_dropout_count, 0)
        print(f"✓ 替换了 {sp_dropout_count} 个Dropout")


class TestSequenceParallelWithTensorParallel(unittest.TestCase):
    """测试序列并行与张量并行结合"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 2
        self.seq_len = 128
        self.hidden_size = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_sp_tp_combined_single_gpu(self):
        """测试SP+TP组合（单GPU模拟）"""
        print("\n测试SP+TP组合...")
        
        # 创建模型
        model = SimpleTransformerBlock(self.hidden_size).to(self.device)
        
        # 先应用序列并行
        sp_config = SequenceParallelConfig(sp_size=1, tp_size=1)
        sp = SequenceParallel(
            model,
            rank=0,
            world_size=1,
            config=sp_config,
        )
        
        # 创建输入
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        # 前向传播
        out = sp(x)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        print(f"✓ SP+TP组合输出形状正确: {out.shape}")


class TestSequenceParallelMemory(unittest.TestCase):
    """测试序列并行的内存优化效果"""
    
    def setUp(self):
        """测试前设置"""
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_size = 1024
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_memory_reduction_calculation(self):
        """测试内存减少计算"""
        print("\n测试内存减少计算...")
        
        # 创建模型
        model = SimpleTransformerBlock(self.hidden_size).to(self.device)
        
        # 创建序列并行
        sp = SequenceParallel(
            model,
            rank=0,
            world_size=4,
            sp_size=4,
            tp_size=1,
        )
        
        # 获取内存统计
        memory_stats = sp.get_memory_stats()
        
        self.assertIn('activation_memory_reduction', memory_stats)
        self.assertIn('layernorm_memory_reduction', memory_stats)
        self.assertEqual(memory_stats['activation_memory_reduction'], 4)
        self.assertEqual(memory_stats['layernorm_memory_reduction'], 4)
        print(f"✓ 内存减少统计: {memory_stats}")


class TestSequenceParallelDistributed(unittest.TestCase):
    """测试分布式序列并行（需要多GPU环境）"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置"""
        cls.distributed_available = dist.is_available() and dist.is_initialized()
        if cls.distributed_available:
            cls.rank = dist.get_rank()
            cls.world_size = dist.get_world_size()
        else:
            cls.rank = 0
            cls.world_size = 1
    
    def test_distributed_sp_setup(self):
        """测试分布式SP设置"""
        if not self.distributed_available:
            self.skipTest("分布式环境不可用")
        
        print(f"\n测试分布式SP设置 (rank={self.rank}, world_size={self.world_size})...")
        
        # 创建模型
        model = SimpleTransformerBlock(512)
        
        # 创建序列并行
        sp = SequenceParallel(
            model,
            rank=self.rank,
            world_size=self.world_size,
            sp_size=min(2, self.world_size),
            tp_size=1,
        )
        
        # 验证初始化
        self.assertIsNotNone(sp)
        print(f"✓ Rank {self.rank} 序列并行初始化成功")
    
    def test_distributed_forward_backward(self):
        """测试分布式前向/反向传播"""
        if not self.distributed_available or self.world_size < 2:
            self.skipTest("需要至少2个GPU")
        
        print(f"\n测试分布式前向/反向传播 (rank={self.rank})...")
        
        device = torch.device(f'cuda:{self.rank}')
        model = SimpleTransformerBlock(512).to(device)
        
        sp = SequenceParallel(
            model,
            rank=self.rank,
            world_size=self.world_size,
            sp_size=2,
            tp_size=1,
        )
        
        # 创建输入
        batch_size = 2
        seq_len = 128
        hidden_size = 512
        
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # 前向传播
        out = sp(x)
        
        # 在分布式环境下，输出可能被切分
        if sp.sp_size > 1 and not sp.config.gather_output:
            expected_seq_len = seq_len // sp.sp_size
            self.assertEqual(out.shape[1], expected_seq_len)
        
        # 反向传播
        loss = out.sum()
        loss.backward()
        
        print(f"✓ Rank {self.rank} 前向/反向传播成功")


def run_single_gpu_tests():
    """运行单GPU测试"""
    print("=" * 60)
    print("运行单GPU序列并行测试")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加单GPU测试
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceParallelLayers))
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceParallelIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceParallelWithTensorParallel))
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceParallelMemory))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_distributed_tests():
    """运行分布式测试"""
    print("=" * 60)
    print("运行分布式序列并行测试")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加分布式测试
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceParallelDistributed))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def demo_sequence_parallel():
    """演示序列并行功能"""
    print("\n" + "=" * 60)
    print("序列并行功能演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleTransformerBlock(hidden_size=512).to(device)
    
    print("\n1. 创建序列并行配置...")
    config = SequenceParallelConfig(
        sp_size=1,  # 单GPU测试
        tp_size=1,
        enable_for_layernorm=True,
        enable_for_dropout=True,
    )
    print(f"   配置: sp_size={config.sp_size}, tp_size={config.tp_size}")
    
    print("\n2. 初始化序列并行...")
    sp = SequenceParallel(
        model,
        rank=0,
        world_size=1,
        config=config,
    )
    print(f"   序列并行初始化完成")
    
    print("\n3. 执行前向传播...")
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    print(f"   输入形状: {x.shape}")
    
    out = sp(x)
    print(f"   输出形状: {out.shape}")
    
    print("\n4. 执行反向传播...")
    loss = out.sum()
    loss.backward()
    print("   反向传播完成")
    
    print("\n5. 内存优化统计...")
    memory_stats = sp.get_memory_stats()
    for key, value in memory_stats.items():
        print(f"   {key}: {value}x")
    
    print("\n6. 并行信息...")
    info = sp.get_parallel_info()
    print(f"   sp_size: {info['sp_size']}")
    print(f"   tp_size: {info['tp_size']}")
    print(f"   sequence_dim: {info['sequence_dim']}")
    print(f"   mode: {info['mode']}")
    
    print("\n✓ 序列并行演示完成!")


if __name__ == '__main__':
    # 检查是否是分布式运行
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 分布式环境
        import torch.distributed as dist
        
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        if not dist.is_initialized():
            dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
        
        torch.cuda.set_device(rank)
        
        success = run_distributed_tests()
        
        dist.destroy_process_group()
    else:
        # 单GPU环境
        success = run_single_gpu_tests()
        
        # 运行演示
        if success:
            demo_sequence_parallel()
    
    sys.exit(0 if success else 1)
