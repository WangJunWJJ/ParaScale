# -*- coding: utf-8 -*-
# @Time    : 2026/3/12
# @Author  : Jun Wang
# @File    : para_engine.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale ParaEngine - 自动并行策略选择训练引擎

本模块实现了智能的自动并行策略选择引擎，能够根据模型规模、硬件状态
自适应决策最优的并行策略组合（数据并行、张量并行、流水线并行）。

核心功能:
- 模型规模分析：自动分析模型的参数量、层结构、内存需求
- 硬件状态监控：实时监测 GPU 内存、计算能力、通信带宽
- 智能策略决策：基于启发式规则和性能模型选择最优并行组合
- 动态调整：支持在训练过程中根据资源变化调整并行策略
- 自动配置：无需手动指定并行参数，自动推荐最优配置

Example:
    >>> from parascale import ParaEngine, SimpleModel
    >>> model = SimpleModel()
    >>> engine = ParaEngine(model, auto_parallel=True)
    >>> engine.train(dataloader, epochs=10)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import logging

from ..config import ParaScaleConfig
from ..parallel import (
    DataParallel, 
    ModelParallel, 
    TensorParallel, 
    PipelineParallel,
    HybridParallel
)
from ..utils.utils import (
    print_rank_0, get_rank, get_world_size, get_local_rank,
    get_node_rank, get_num_nodes, setup_logging
)
from ..utils.distributed_utils import initialize_distributed, print_distributed_info
from ..utils.hardware_monitor import create_hardware_monitor

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """
    模型配置文件

    存储模型分析结果，用于并行策略决策。

    Attributes:
        total_params: 模型总参数量
        total_memory: 模型总内存占用（字节）
        num_layers: 模型层数
        max_layer_memory: 最大层的内存占用
        layer_types: 模型中各层类型的计数字典
        embedding_size: 嵌入层大小（如果有）
        hidden_size: 隐藏层维度
        vocab_size: 词表大小
        model_type: 模型类型（'transformer', 'rnn', 'cnn', 'mlp', 'unknown'）
    """
    total_params: int = 0
    total_memory: int = 0
    num_layers: int = 0
    max_layer_memory: int = 0
    layer_types: Dict[str, int] = None
    embedding_size: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    model_type: str = 'unknown'

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_params': self.total_params,
            'total_memory_mb': self.total_memory / (1024 ** 2),
            'num_layers': self.num_layers,
            'max_layer_memory_mb': self.max_layer_memory / (1024 ** 2),
            'layer_types': self.layer_types,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
        }


@dataclass
class HardwareProfile:
    """
    硬件配置文件
    
    存储硬件资源分析结果，用于并行策略决策。
    
    Attributes:
        num_gpus: GPU 数量
        gpu_memory: 每个 GPU 的内存（字节）
        gpu_compute_capability: GPU 计算能力
        available_memory: 可用 GPU 内存（字节）
        communication_bandwidth: 通信带宽（GB/s）
        num_nodes: 节点数
        gpus_per_node: 每个节点的 GPU 数
    """
    num_gpus: int = 0
    gpu_memory: int = 0
    gpu_compute_capability: float = 0.0
    available_memory: int = 0
    communication_bandwidth: float = 0.0
    num_nodes: int = 1
    gpus_per_node: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'num_gpus': self.num_gpus,
            'gpu_memory_gb': self.gpu_memory / (1024 ** 3),
            'gpu_compute_capability': self.gpu_compute_capability,
            'available_memory_gb': self.available_memory / (1024 ** 3),
            'communication_bandwidth_gbps': self.communication_bandwidth,
            'num_nodes': self.num_nodes,
            'gpus_per_node': self.gpus_per_node,
        }


@dataclass
class ParallelStrategy:
    """
    并行策略配置
    
    存储推荐的并行策略组合。
    
    Attributes:
        dp_size: 数据并行大小
        tp_size: 张量并行大小
        pp_size: 流水线并行大小
        strategy_type: 策略类型（'data', 'tensor', 'pipeline', 'hybrid'）
        reason: 选择该策略的原因
        estimated_memory_saving: 预估的内存节省比例
        estimated_speedup: 预估的加速比
    """
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    strategy_type: str = 'data'
    reason: str = ''
    estimated_memory_saving: float = 0.0
    estimated_speedup: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'dp_size': self.dp_size,
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
            'strategy_type': self.strategy_type,
            'reason': self.reason,
            'estimated_memory_saving': self.estimated_memory_saving,
            'estimated_speedup': self.estimated_speedup,
        }
    
    def validate(self, world_size: int) -> bool:
        """
        验证并行配置是否有效
        
        Args:
            world_size: 总进程数
        
        Returns:
            如果配置有效返回 True
        """
        return self.dp_size * self.tp_size * self.pp_size == world_size


class ModelAnalyzer:
    """
    模型分析器
    
    分析模型结构、参数量、内存需求等特征，
    为并行策略决策提供依据。
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化模型分析器
        
        Args:
            model: PyTorch 模型实例
        """
        self.model = model
        self.profile = ModelProfile()
    
    def analyze(self) -> ModelProfile:
        """
        分析模型并生成配置文件
        
        Returns:
            模型配置文件
        """
        self._count_parameters()
        self._analyze_layers()
        self._estimate_memory()
        self._detect_model_type()
        return self.profile
    
    def _count_parameters(self) -> None:
        """统计模型参数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.profile.total_params = total_params
        self.profile.total_memory = total_params * 4  # 假设 float32，4 字节/参数
    
    def _analyze_layers(self) -> None:
        """分析模型层结构"""
        layer_types = {}
        max_layer_memory = 0
        num_layers = 0

        for name, module in self.model.named_modules():
            # 跳过模型本身（空名称）
            if name == '':
                continue

            module_type = module.__class__.__name__
            layer_types[module_type] = layer_types.get(module_type, 0) + 1

            module_params = sum(p.numel() for p in module.parameters())
            module_memory = module_params * 4
            max_layer_memory = max(max_layer_memory, module_memory)

            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.TransformerEncoderLayer)):
                num_layers += 1

        self.profile.layer_types = layer_types
        self.profile.num_layers = num_layers
        self.profile.max_layer_memory = max_layer_memory
    
    def _estimate_memory(self) -> None:
        """估算模型内存需求"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                self.profile.embedding_size = module.num_embeddings * module.embedding_dim
            elif isinstance(module, nn.Linear):
                self.profile.hidden_size = max(self.profile.hidden_size, module.out_features)
    
    def _detect_model_type(self) -> None:
        """检测模型类型"""
        layer_types = self.profile.layer_types

        if not layer_types:
            self.profile.model_type = 'unknown'
            return

        if 'TransformerEncoderLayer' in layer_types or 'MultiheadAttention' in layer_types:
            self.profile.model_type = 'transformer'
        elif 'LSTM' in layer_types or 'GRU' in layer_types:
            self.profile.model_type = 'rnn'
        elif 'Conv2d' in layer_types or 'Conv1d' in layer_types or 'Conv3d' in layer_types:
            self.profile.model_type = 'cnn'
        elif 'Linear' in layer_types:
            # 只要有 Linear 层就认为是 MLP（包括简单的多层感知机）
            linear_count = layer_types.get('Linear', 0)
            # 检查是否主要是 Linear 层（允许有激活函数等辅助层）
            mlp_related_layers = ['Linear', 'ReLU', 'GELU', 'Sigmoid', 'Tanh', 'Dropout', 'BatchNorm1d', 'LayerNorm']
            mlp_layer_count = sum(layer_types.get(k, 0) for k in mlp_related_layers)
            total_layers = sum(layer_types.values())
            # 如果 MLP 相关层占总层数的大部分，则认为是 MLP
            if linear_count >= 1 and mlp_layer_count >= total_layers * 0.5:
                self.profile.model_type = 'mlp'
            else:
                self.profile.model_type = 'unknown'
        else:
            self.profile.model_type = 'unknown'


class HardwareMonitor:
    """
    硬件监控器
    
    监控 GPU 资源、内存、通信带宽等硬件状态，
    为并行策略决策提供硬件信息。
    """
    
    def __init__(self):
        """初始化硬件监控器"""
        self.profile = HardwareProfile()
    
    def monitor(self) -> HardwareProfile:
        """
        监控硬件资源并生成配置文件
        
        Returns:
            硬件配置文件
        """
        self._detect_gpus()
        self._measure_memory()
        self._detect_compute_capability()
        self._estimate_bandwidth()
        return self.profile
    
    def _detect_gpus(self) -> None:
        """检测 GPU 信息"""
        if dist.is_initialized():
            # 分布式环境下使用 world_size
            self.profile.num_nodes = get_num_nodes()
            self.profile.num_gpus = get_world_size()
            self.profile.gpus_per_node = get_world_size() // self.profile.num_nodes
        elif torch.cuda.is_available():
            # 非分布式环境但 CUDA 可用，默认使用单 GPU
            self.profile.num_gpus = 1
            self.profile.gpus_per_node = 1
        else:
            # CPU 环境
            self.profile.num_gpus = 1
            self.profile.gpus_per_node = 1
    
    def _measure_memory(self) -> None:
        """测量 GPU 内存"""
        if torch.cuda.is_available():
            local_rank = get_local_rank()
            total_memory = torch.cuda.get_device_properties(local_rank).total_memory
            allocated_memory = torch.cuda.memory_allocated(local_rank)
            self.profile.gpu_memory = total_memory
            self.profile.available_memory = total_memory - allocated_memory
        else:
            self.profile.gpu_memory = 8 * 1024 ** 3  # 默认 8GB
            self.profile.available_memory = self.profile.gpu_memory
    
    def _detect_compute_capability(self) -> None:
        """检测 GPU 计算能力"""
        if torch.cuda.is_available():
            local_rank = get_local_rank()
            major = torch.cuda.get_device_capability(local_rank)[0]
            minor = torch.cuda.get_device_capability(local_rank)[1]
            self.profile.gpu_compute_capability = major + minor * 0.1
        else:
            self.profile.gpu_compute_capability = 7.0
    
    def _estimate_bandwidth(self) -> None:
        """估算通信带宽"""
        if dist.is_initialized():
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{get_local_rank()}')
                tensor = torch.randn(100 * 1024 * 1024, device=device)
                
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                dist.all_reduce(tensor)
                end_time.record()
                torch.cuda.synchronize()
                
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0
                data_size = 2 * tensor.element_size() * tensor.numel()
                self.profile.communication_bandwidth = data_size / elapsed_time / (1024 ** 3)
            else:
                self.profile.communication_bandwidth = 10.0
        else:
            self.profile.communication_bandwidth = 0.0


class StrategyDecider:
    """
    并行策略决策器
    
    根据模型分析和硬件监控结果，
    使用启发式规则和性能模型决策最优并行策略。
    """
    
    def __init__(self, model_profile: ModelProfile, hardware_profile: HardwareProfile):
        """
        初始化策略决策器
        
        Args:
            model_profile: 模型配置文件
            hardware_profile: 硬件配置文件
        """
        self.model_profile = model_profile
        self.hardware_profile = hardware_profile
    
    def decide(self) -> ParallelStrategy:
        """
        决策最优并行策略
        
        Returns:
            推荐的并行策略配置
        """
        world_size = self.hardware_profile.num_gpus
        
        if world_size == 1:
            return self._single_gpu_strategy()
        
        if self.model_profile.total_params > 10 * 1024 ** 3:
            return self._large_model_strategy(world_size)
        elif self.model_profile.total_params > 1 * 1024 ** 3:
            return self._medium_model_strategy(world_size)
        else:
            return self._small_model_strategy(world_size)
    
    def _single_gpu_strategy(self) -> ParallelStrategy:
        """单 GPU 策略"""
        return ParallelStrategy(
            dp_size=1,
            tp_size=1,
            pp_size=1,
            strategy_type='single',
            reason='Single GPU detected',
            estimated_memory_saving=0.0,
            estimated_speedup=1.0
        )
    
    def _small_model_strategy(self, world_size: int) -> ParallelStrategy:
        """
        小模型策略（< 1B 参数）
        
        优先使用数据并行，最大化数据并行度
        """
        return ParallelStrategy(
            dp_size=world_size,
            tp_size=1,
            pp_size=1,
            strategy_type='data',
            reason=f'Small model ({self.model_profile.total_params/1e9:.2f}B params), use pure data parallel',
            estimated_memory_saving=1.0 - 1.0/world_size,
            estimated_speedup=world_size * 0.9
        )
    
    def _medium_model_strategy(self, world_size: int) -> ParallelStrategy:
        """
        中等模型策略（1B-10B 参数）
        
        使用数据并行 + 张量并行的混合策略
        """
        available_memory = self.hardware_profile.available_memory
        model_memory = self.model_profile.total_memory
        
        if model_memory > available_memory * 0.5:
            tp_size = min(4, world_size)
            if world_size % tp_size == 0:
                dp_size = world_size // tp_size
                return ParallelStrategy(
                    dp_size=dp_size,
                    tp_size=tp_size,
                    pp_size=1,
                    strategy_type='hybrid',
                    reason=f'Medium model with memory constraints, use DP={dp_size} × TP={tp_size}',
                    estimated_memory_saving=1.0 - 1.0/(dp_size * tp_size),
                    estimated_speedup=dp_size * tp_size * 0.85
                )
        
        dp_size = world_size
        return ParallelStrategy(
            dp_size=dp_size,
            tp_size=1,
            pp_size=1,
            strategy_type='data',
            reason=f'Medium model ({self.model_profile.total_params/1e9:.2f}B params), use data parallel',
            estimated_memory_saving=1.0 - 1.0/dp_size,
            estimated_speedup=dp_size * 0.9
        )
    
    def _large_model_strategy(self, world_size: int) -> ParallelStrategy:
        """
        大模型策略（> 10B 参数）
        
        使用 3D 混合并行：数据并行 + 张量并行 + 流水线并行
        """
        available_memory = self.hardware_profile.available_memory
        model_memory = self.model_profile.total_memory
        
        if model_memory > available_memory:
            pp_size = min(8, world_size)
            remaining = world_size // pp_size
            
            tp_size = min(4, remaining)
            while remaining % tp_size != 0 and tp_size > 1:
                tp_size -= 1
            
            dp_size = remaining // tp_size
            
            return ParallelStrategy(
                dp_size=dp_size,
                tp_size=tp_size,
                pp_size=pp_size,
                strategy_type='hybrid',
                reason=f'Large model ({self.model_profile.total_params/1e9:.2f}B params), use 3D parallel: DP={dp_size} × TP={tp_size} × PP={pp_size}',
                estimated_memory_saving=1.0 - 1.0/(dp_size * tp_size * pp_size),
                estimated_speedup=dp_size * tp_size * pp_size * 0.75
            )
        
        if world_size >= 8:
            pp_size = 4
            tp_size = 2
            dp_size = world_size // (pp_size * tp_size)
            
            return ParallelStrategy(
                dp_size=dp_size,
                tp_size=tp_size,
                pp_size=pp_size,
                strategy_type='hybrid',
                reason=f'Large model with sufficient resources, use 3D parallel: DP={dp_size} × TP={tp_size} × PP={pp_size}',
                estimated_memory_saving=1.0 - 1.0/(dp_size * tp_size * pp_size),
                estimated_speedup=dp_size * tp_size * pp_size * 0.8
            )
        
        dp_size = world_size // 2
        pp_size = 2
        return ParallelStrategy(
            dp_size=dp_size,
            tp_size=1,
            pp_size=pp_size,
            strategy_type='hybrid',
            reason=f'Large model, use DP+PP: DP={dp_size} × PP={pp_size}',
            estimated_memory_saving=1.0 - 1.0/(dp_size * pp_size),
            estimated_speedup=dp_size * pp_size * 0.8
        )


class ParaEngine:
    """
    ParaScale 自动并行训练引擎
    
    支持根据模型规模、硬件状态自适应决策并行优化策略组合。
    自动选择最优的并行配置（数据并行、张量并行、流水线并行）。
    
    Attributes:
        model: PyTorch 模型实例
        optimizer: 优化器实例
        config: ParaScaleConfig 配置实例
        rank: 当前进程的 rank
        world_size: 世界大小（进程总数）
        local_rank: 本地 rank（当前节点内的 GPU 编号）
        parallel_strategy: 自动选择的并行策略
        hybrid_parallel: HybridParallel 实例
        global_step: 全局训练步数
    
    Example:
        >>> model = TransformerModel()
        >>> optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        >>> engine = ParaEngine(model, optimizer, auto_parallel=True)
        >>> print(f"Selected strategy: {engine.parallel_strategy}")
        >>> engine.train(dataloader, epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        config: Optional[ParaScaleConfig] = None,
        auto_parallel: bool = True,
        auto_init_distributed: bool = True
    ):
        """
        初始化 ParaEngine
        
        Args:
            model: PyTorch 模型实例
            optimizer: PyTorch 优化器实例（可选）
            config: ParaScaleConfig 配置实例
            auto_parallel: 是否启用自动并行策略选择
            auto_init_distributed: 是否自动初始化分布式环境
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or ParaScaleConfig()
        self.auto_parallel = auto_parallel
        
        if auto_init_distributed and not dist.is_initialized():
            initialize_distributed()
        
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()
        
        self.parallel_strategy: Optional[ParallelStrategy] = None
        self.hybrid_parallel: Optional[HybridParallel] = None
        self.global_step = 0
        
        self.hardware_monitor = None
        self.monitor_interval = 100  # 每 100 个 batch 监控一次
        
        if auto_parallel:
            self._auto_configure_parallel()
        else:
            self._manual_configure_parallel()
    
    def _auto_configure_parallel(self) -> None:
        """自动配置并行策略"""
        print_rank_0("\n" + "="*80)
        print_rank_0("ParaEngine Auto-Parallel Configuration")
        print_rank_0("="*80)
        
        model_analyzer = ModelAnalyzer(self.model)
        model_profile = model_analyzer.analyze()
        
        print_rank_0("\nModel Profile:")
        for key, value in model_profile.to_dict().items():
            print_rank_0(f"  {key}: {value}")
        
        hardware_monitor = HardwareMonitor()
        hardware_profile = hardware_monitor.monitor()
        
        print_rank_0("\nHardware Profile:")
        for key, value in hardware_profile.to_dict().items():
            print_rank_0(f"  {key}: {value}")
        
        strategy_decider = StrategyDecider(model_profile, hardware_profile)
        self.parallel_strategy = strategy_decider.decide()
        
        print_rank_0("\nRecommended Parallel Strategy:")
        for key, value in self.parallel_strategy.to_dict().items():
            print_rank_0(f"  {key}: {value}")
        
        print_rank_0("="*80 + "\n")
        
        self._apply_parallel_strategy()
    
    def _apply_parallel_strategy(self) -> None:
        """应用并行策略"""
        if not self.parallel_strategy:
            return

        dp = self.parallel_strategy.dp_size
        tp = self.parallel_strategy.tp_size
        pp = self.parallel_strategy.pp_size

        # 确保所有并行尺寸至少为1
        dp = max(1, dp)
        tp = max(1, tp)
        pp = max(1, pp)

        # 验证并行配置是否有效，如果无效则进行调整
        if dp * tp * pp != self.world_size:
            print_rank_0(f"Warning: Strategy {dp}×{tp}×{pp} != world_size {self.world_size}, adjusting...")
            # 重新计算合理的并行配置
            if self.world_size == 1:
                dp, tp, pp = 1, 1, 1
            else:
                # 优先保持数据并行，其次张量并行，最后流水线并行
                dp = min(dp, self.world_size)
                remaining = self.world_size // dp
                tp = min(tp, remaining)
                remaining = remaining // tp
                pp = max(1, remaining)
        
        tensor_parallel_mode = "row" if self.parallel_strategy.tp_size <= 2 else "column"
        
        self.hybrid_parallel = HybridParallel(
            model=self.model,
            rank=self.rank,
            world_size=self.world_size,
            dp_size=dp,
            tp_size=tp,
            pp_size=pp,
            tensor_parallel_mode=tensor_parallel_mode,
            pipeline_chunks=self.config.pipeline_parallel_chunks
        )
        
        self.config.data_parallel_size = dp
        self.config.tensor_parallel_size = tp
        self.config.pipeline_parallel_size = pp
    
    def _manual_configure_parallel(self) -> None:
        """手动配置并行策略"""
        dp = self.config.data_parallel_size
        tp = self.config.tensor_parallel_size
        pp = self.config.pipeline_parallel_size
        
        if dp * tp * pp != self.world_size:
            print_rank_0(f"Warning: Manual config {dp}×{tp}×{pp} != world_size {self.world_size}")
        
        self.hybrid_parallel = HybridParallel(
            model=self.model,
            rank=self.rank,
            world_size=self.world_size,
            dp_size=dp,
            tp_size=tp,
            pp_size=pp,
            tensor_parallel_mode=self.config.tensor_parallel_mode,
            pipeline_chunks=self.config.pipeline_parallel_chunks
        )
    
    def train(self, dataloader: Any, epochs: int = 1) -> None:
        """
        训练模型
        
        Args:
            dataloader: 数据加载器
            epochs: 训练轮数
        """
        self.model.train()
        
        for epoch in range(epochs):
            print_rank_0(f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if torch.cuda.is_available():
                    targets = targets.to(f"cuda:{self.local_rank}")
                
                outputs = self._forward(inputs)
                
                if outputs is not None:
                    loss = nn.functional.cross_entropy(outputs, targets)
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self._gather_gradients()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        
                        if self.global_step % 100 == 0:
                            print_rank_0(f"Step {self.global_step}, Loss: {loss.item():.4f}")
                        
                        # 实时监控硬件状态
                        if self.global_step % self.monitor_interval == 0:
                            self._monitor_hardware()
    
    def _forward(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """执行前向传播"""
        if self.hybrid_parallel:
            return self.hybrid_parallel.forward(inputs)
        else:
            if torch.cuda.is_available():
                inputs = inputs.to(f"cuda:{self.local_rank}")
            return self.model(inputs)
    
    def _gather_gradients(self) -> None:
        """收集梯度"""
        if self.hybrid_parallel:
            self.hybrid_parallel.gather_gradients()
    
    def _monitor_hardware(self) -> None:
        """
        实时监控硬件状态
        
        在训练过程中定期收集 GPU 内存、计算能力、通信带宽等指标，
        为性能优化和资源管理提供实时数据支持。
        """
        if self.hardware_monitor is None:
            self.hardware_monitor = create_hardware_monitor(
                local_rank=self.local_rank,
                max_history_size=1000,
                bandwidth_test_interval=60.0,
                compute_check_interval=30.0
            )
        
        metrics = self.hardware_monitor.collect_metrics()
        
        if self.global_step % (self.monitor_interval * 10) == 0:
            print_rank_0("\n" + "="*60)
            print_rank_0("硬件监控摘要")
            print_rank_0("="*60)
            self.hardware_monitor.print_metrics_summary()
            print_rank_0("="*60 + "\n")
    
    def evaluate(self, dataloader: Any) -> Tuple[float, float]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            (平均损失，准确率)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                if torch.cuda.is_available():
                    targets = targets.to(f"cuda:{self.local_rank}")
                
                outputs = self._forward(inputs)
                
                if outputs is not None:
                    loss = nn.functional.cross_entropy(outputs, targets, reduction='sum')
                    total_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        if total > 0:
            accuracy = 100. * correct / total
            avg_loss = total_loss / total
            return avg_loss, accuracy
        else:
            return 0.0, 0.0
    
    def get_parallel_info(self) -> Dict[str, Any]:
        """
        获取并行信息
        
        Returns:
            并行信息字典
        """
        if self.hybrid_parallel:
            return self.hybrid_parallel.get_parallel_info()
        return {}
    
    def get_strategy(self) -> Optional[ParallelStrategy]:
        """
        获取当前并行策略
        
        Returns:
            并行策略配置
        """
        return self.parallel_strategy
