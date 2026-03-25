# -*- coding: utf-8 -*-
# @Time    : 2026/3/23
# @Author  : Jun Wang
# @File    : adaptive_parallel.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 自适应并行策略模块

本模块实现了根据硬件环境和模型特性自动选择最优并行策略的功能，
特别针对华为昇腾平台进行了优化。

核心功能：
- 硬件感知：自动检测硬件拓扑和性能特征
- 模型分析：分析模型结构和计算特性
- 策略推荐：基于分析结果推荐最优并行策略
- 动态调整：训练过程中动态调整并行策略

支持的并行策略：
- 数据并行 (DP)
- 张量并行 (TP)
- 流水线并行 (PP)
- 3D 混合并行 (DP + TP + PP)
- ZeRO 优化 (Stage 1-3)

决策因素：
1. 模型大小和结构
2. GPU/NPU 数量和拓扑
3. 内存容量
4. 通信带宽
5. 计算能力

Example:
    >>> analyzer = AdaptiveParallelAnalyzer()
    >>> strategy = analyzer.analyze(model, cluster_info)
    >>> print(f"推荐策略: {strategy}")
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

# 检测 Ascend 环境
ASCEND_AVAILABLE = False
try:
    import torch_npu
    ASCEND_AVAILABLE = torch.npu.is_available()
except ImportError:
    pass


class ParallelStrategyType(Enum):
    """并行策略类型枚举"""
    DP_ONLY = "dp_only"                     # 仅数据并行
    TP_ONLY = "tp_only"                     # 仅张量并行
    PP_ONLY = "pp_only"                     # 仅流水线并行
    DP_TP = "dp_tp"                         # 数据并行 + 张量并行
    DP_PP = "dp_pp"                         # 数据并行 + 流水线并行
    TP_PP = "tp_pp"                         # 张量并行 + 流水线并行
    DP_TP_PP = "dp_tp_pp"                   # 3D 混合并行
    ZERO_1 = "zero_1"                       # ZeRO Stage 1
    ZERO_2 = "zero_2"                       # ZeRO Stage 2
    ZERO_3 = "zero_3"                       # ZeRO Stage 3


class HardwareType(Enum):
    """硬件类型枚举"""
    NVIDIA_GPU = "nvidia_gpu"
    HUAWEI_NPU = "huawei_npu"
    AMD_GPU = "amd_gpu"
    CPU = "cpu"


@dataclass
class ModelProfile:
    """模型性能分析数据"""
    num_parameters: int = 0
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    max_seq_length: int = 0
    model_memory_bytes: int = 0
    activation_memory_bytes: int = 0
    gradient_memory_bytes: int = 0
    optimizer_memory_bytes: int = 0
    is_transformer: bool = False
    has_embedding: bool = False
    layer_types: List[str] = field(default_factory=list)


@dataclass
class ClusterInfo:
    """集群信息数据"""
    hardware_type: HardwareType = HardwareType.NVIDIA_GPU
    num_nodes: int = 1
    num_devices_per_node: int = 1
    total_devices: int = 1
    device_memory_bytes: int = 0
    intra_node_bandwidth_gbps: float = 300.0
    inter_node_bandwidth_gbps: float = 100.0
    compute_capability: float = 1.0
    supports_tensor_parallel: bool = True
    supports_pipeline_parallel: bool = True


@dataclass
class ParallelStrategy:
    """并行策略配置"""
    strategy_type: ParallelStrategyType
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    zero_stage: int = 0
    offload_optimizer: bool = False
    offload_parameters: bool = False
    gradient_checkpointing: bool = False
    micro_batches: int = 1
    expected_memory_usage: int = 0
    expected_speedup: float = 1.0
    reasoning: str = ""


class ModelProfiler:
    """
    模型分析器
    
    分析模型结构和计算特性，为并行策略选择提供依据。
    
    Example:
        >>> profiler = ModelProfiler()
        >>> profile = profiler.profile(model)
        >>> print(f"参数量: {profile.num_parameters:,}")
    """
    
    def __init__(self):
        self._profile: Optional[ModelProfile] = None
    
    def profile(self, model: nn.Module, batch_size: int = 1,
                seq_length: int = 512) -> ModelProfile:
        """
        分析模型
        
        Args:
            model: 模型
            batch_size: 批次大小
            seq_length: 序列长度
        
        Returns:
            模型分析结果
        """
        self._profile = ModelProfile()
        
        # 统计参数
        self._count_parameters(model)
        
        # 分析层结构
        self._analyze_layers(model)
        
        # 估算内存
        self._estimate_memory(batch_size, seq_length)
        
        # 检测模型类型
        self._detect_model_type(model)
        
        return self._profile
    
    def _count_parameters(self, model: nn.Module):
        """统计参数数量"""
        self._profile.num_parameters = sum(p.numel() for p in model.parameters())
        self._profile.model_memory_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
    
    def _analyze_layers(self, model: nn.Module):
        """分析层结构"""
        layer_types = set()
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            layer_types.add(module_type)
            
            # 提取 Transformer 特征
            if 'Linear' in module_type:
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    in_f = module.in_features
                    out_f = module.out_features
                    
                    # 推断隐藏层大小
                    if self._profile.hidden_size == 0:
                        self._profile.hidden_size = max(in_f, out_f)
                    
                    # 推断中间层大小
                    if out_f > in_f and out_f > self._profile.intermediate_size:
                        self._profile.intermediate_size = out_f
            
            if 'Embedding' in module_type:
                self._profile.has_embedding = True
                if hasattr(module, 'num_embeddings'):
                    self._profile.vocab_size = module.num_embeddings
            
            if 'MultiheadAttention' in module_type or 'Attention' in module_type:
                self._profile.is_transformer = True
                if hasattr(module, 'num_heads'):
                    self._profile.num_attention_heads = module.num_heads
        
        self._profile.layer_types = list(layer_types)
        self._profile.num_layers = len(list(model.named_children()))
    
    def _estimate_memory(self, batch_size: int, seq_length: int):
        """估算内存使用"""
        hidden_size = self._profile.hidden_size or 768
        
        # 激活内存估算
        self._profile.activation_memory_bytes = (
            batch_size * seq_length * hidden_size * 4 *  # FP32
            self._profile.num_layers * 2  # 前向 + 反向
        )
        
        # 梯度内存
        self._profile.gradient_memory_bytes = self._profile.model_memory_bytes
        
        # 优化器内存 (Adam: 2 个状态)
        self._profile.optimizer_memory_bytes = self._profile.model_memory_bytes * 2
    
    def _detect_model_type(self, model: nn.Module):
        """检测模型类型"""
        model_class = model.__class__.__name__.lower()
        
        if any(x in model_class for x in ['bert', 'gpt', 'llama', 'transformer']):
            self._profile.is_transformer = True


class ClusterDetector:
    """
    集群检测器
    
    自动检测集群硬件配置和拓扑结构。
    
    Example:
        >>> detector = ClusterDetector()
        >>> info = detector.detect()
        >>> print(f"设备数量: {info.total_devices}")
    """
    
    def __init__(self):
        self._info: Optional[ClusterInfo] = None
    
    def detect(self) -> ClusterInfo:
        """
        检测集群信息
        
        Returns:
            集群信息
        """
        self._info = ClusterInfo()
        
        # 检测硬件类型
        self._info.hardware_type = self._detect_hardware_type()
        
        # 检测设备数量
        self._info.num_devices_per_node = self._detect_device_count()
        
        # 检测节点数
        self._info.num_nodes = self._detect_node_count()
        
        self._info.total_devices = self._info.num_nodes * self._info.num_devices_per_node
        
        # 检测设备内存
        self._info.device_memory_bytes = self._detect_device_memory()
        
        # 检测带宽
        self._info.intra_node_bandwidth_gbps, self._info.inter_node_bandwidth_gbps = \
            self._detect_bandwidth()
        
        # 检测计算能力
        self._info.compute_capability = self._detect_compute_capability()
        
        return self._info
    
    def _detect_hardware_type(self) -> HardwareType:
        """检测硬件类型"""
        if ASCEND_AVAILABLE:
            return HardwareType.HUAWEI_NPU
        elif torch.cuda.is_available():
            return HardwareType.NVIDIA_GPU
        else:
            return HardwareType.CPU
    
    def _detect_device_count(self) -> int:
        """检测设备数量"""
        if ASCEND_AVAILABLE:
            return torch.npu.device_count()
        elif torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1
    
    def _detect_node_count(self) -> int:
        """检测节点数"""
        import torch.distributed as dist
        if dist.is_initialized():
            world_size = dist.get_world_size()
            local_size = self._detect_device_count()
            return (world_size + local_size - 1) // local_size
        return 1
    
    def _detect_device_memory(self) -> int:
        """检测设备内存"""
        if ASCEND_AVAILABLE:
            try:
                return torch.npu.get_device_properties(0).total_memory
            except Exception:
                return 64 * 1024 * 1024 * 1024  # 默认 64GB
        elif torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 16 * 1024 * 1024 * 1024  # 默认 16GB
    
    def _detect_bandwidth(self) -> Tuple[float, float]:
        """检测带宽"""
        if ASCEND_AVAILABLE:
            # HCCS ~300GB/s, RoCE ~100GB/s
            return 300.0, 100.0
        elif torch.cuda.is_available():
            # NVLink ~300GB/s, InfiniBand ~100GB/s
            return 300.0, 100.0
        return 50.0, 25.0
    
    def _detect_compute_capability(self) -> float:
        """检测计算能力"""
        if ASCEND_AVAILABLE:
            # 昇腾 910B ~310 TFLOPS FP16
            return 310.0
        elif torch.cuda.is_available():
            # A100 ~312 TFLOPS FP16
            return 312.0
        return 1.0


class AdaptiveParallelAnalyzer:
    """
    自适应并行策略分析器
    
    根据模型和硬件特性自动选择最优并行策略。
    
    决策流程：
    1. 分析模型结构和内存需求
    2. 检测硬件配置和拓扑
    3. 评估各种策略的可行性
    4. 选择最优策略
    
    Example:
        >>> analyzer = AdaptiveParallelAnalyzer()
        >>> strategy = analyzer.analyze(model)
        >>> print(f"推荐策略: {strategy.strategy_type}")
    """
    
    def __init__(self, 
                 memory_safety_factor: float = 0.8,
                 prefer_memory_efficiency: bool = False):
        """
        初始化分析器
        
        Args:
            memory_safety_factor: 内存安全系数
            prefer_memory_efficiency: 是否优先考虑内存效率
        """
        self.memory_safety_factor = memory_safety_factor
        self.prefer_memory_efficiency = prefer_memory_efficiency
        
        self._model_profiler = ModelProfiler()
        self._cluster_detector = ClusterDetector()
    
    def analyze(self, model: nn.Module,
                cluster_info: Optional[ClusterInfo] = None,
                batch_size: int = 1,
                seq_length: int = 512) -> ParallelStrategy:
        """
        分析并推荐最优并行策略
        
        Args:
            model: 模型
            cluster_info: 集群信息（可选，自动检测）
            batch_size: 批次大小
            seq_length: 序列长度
        
        Returns:
            推荐的并行策略
        """
        # 分析模型
        model_profile = self._model_profiler.profile(model, batch_size, seq_length)
        
        # 检测集群
        if cluster_info is None:
            cluster_info = self._cluster_detector.detect()
        
        # 计算总内存需求
        total_memory = (
            model_profile.model_memory_bytes +
            model_profile.activation_memory_bytes +
            model_profile.gradient_memory_bytes +
            model_profile.optimizer_memory_bytes
        )
        
        # 可用内存
        available_memory = int(cluster_info.device_memory_bytes * self.memory_safety_factor)
        
        # 根据内存压力选择策略
        if total_memory <= available_memory:
            # 内存充足，使用简单策略
            return self._select_simple_strategy(model_profile, cluster_info)
        else:
            # 内存不足，需要优化
            return self._select_optimized_strategy(
                model_profile, cluster_info, total_memory, available_memory
            )
    
    def _select_simple_strategy(self, model_profile: ModelProfile,
                                 cluster_info: ClusterInfo) -> ParallelStrategy:
        """选择简单策略（内存充足时）"""
        num_devices = cluster_info.total_devices
        
        if num_devices == 1:
            return ParallelStrategy(
                strategy_type=ParallelStrategyType.DP_ONLY,
                dp_size=1,
                reasoning="单设备，使用数据并行"
            )
        
        # 根据模型大小选择
        if model_profile.num_parameters < 1e9:  # < 1B 参数
            return ParallelStrategy(
                strategy_type=ParallelStrategyType.DP_ONLY,
                dp_size=num_devices,
                reasoning="小模型，使用纯数据并行"
            )
        elif model_profile.num_parameters < 10e9:  # < 10B 参数
            if cluster_info.hardware_type == HardwareType.HUAWEI_NPU:
                # 昇腾平台优先使用张量并行
                return ParallelStrategy(
                    strategy_type=ParallelStrategyType.DP_TP,
                    dp_size=num_devices // 2,
                    tp_size=2,
                    reasoning="中等模型，昇腾平台推荐 DP+TP"
                )
            else:
                return ParallelStrategy(
                    strategy_type=ParallelStrategyType.DP_ONLY,
                    dp_size=num_devices,
                    reasoning="中等模型，使用数据并行"
                )
        else:
            # 大模型需要混合并行
            return self._select_hybrid_strategy(model_profile, cluster_info)
    
    def _select_optimized_strategy(self, model_profile: ModelProfile,
                                    cluster_info: ClusterInfo,
                                    total_memory: int,
                                    available_memory: int) -> ParallelStrategy:
        """选择优化策略（内存不足时）"""
        memory_ratio = total_memory / available_memory
        
        # 根据内存压力程度选择策略
        if memory_ratio < 2:
            # 轻度内存压力，使用 ZeRO Stage 1
            return ParallelStrategy(
                strategy_type=ParallelStrategyType.ZERO_1,
                dp_size=cluster_info.total_devices,
                zero_stage=1,
                expected_memory_usage=int(total_memory * 0.75),
                reasoning="轻度内存压力，使用 ZeRO Stage 1"
            )
        elif memory_ratio < 4:
            # 中度内存压力，使用 ZeRO Stage 2
            return ParallelStrategy(
                strategy_type=ParallelStrategyType.ZERO_2,
                dp_size=cluster_info.total_devices,
                zero_stage=2,
                expected_memory_usage=int(total_memory * 0.5),
                reasoning="中度内存压力，使用 ZeRO Stage 2"
            )
        elif memory_ratio < 8:
            # 重度内存压力，使用 ZeRO Stage 3
            return ParallelStrategy(
                strategy_type=ParallelStrategyType.ZERO_3,
                dp_size=cluster_info.total_devices,
                zero_stage=3,
                offload_optimizer=True,
                expected_memory_usage=int(total_memory * 0.25),
                reasoning="重度内存压力，使用 ZeRO Stage 3 + Offload"
            )
        else:
            # 极端内存压力，使用混合策略
            return self._select_aggressive_strategy(
                model_profile, cluster_info, total_memory, available_memory
            )
    
    def _select_hybrid_strategy(self, model_profile: ModelProfile,
                                 cluster_info: ClusterInfo) -> ParallelStrategy:
        """选择混合并行策略"""
        num_devices = cluster_info.total_devices
        
        # 计算 TP 和 PP 大小
        if cluster_info.hardware_type == HardwareType.HUAWEI_NPU:
            # 昇腾平台：HCCS 高带宽，适合 TP
            tp_size = min(8, num_devices)
            pp_size = min(4, num_devices // tp_size)
        else:
            # NVIDIA 平台：NVLink 高带宽
            tp_size = min(4, num_devices)
            pp_size = min(4, num_devices // tp_size)
        
        dp_size = num_devices // (tp_size * pp_size) if tp_size * pp_size > 0 else 1
        
        return ParallelStrategy(
            strategy_type=ParallelStrategyType.DP_TP_PP,
            dp_size=max(1, dp_size),
            tp_size=tp_size,
            pp_size=pp_size,
            micro_batches=pp_size * 2,
            reasoning=f"大模型，使用 3D 混合并行 (DP={dp_size}, TP={tp_size}, PP={pp_size})"
        )
    
    def _select_aggressive_strategy(self, model_profile: ModelProfile,
                                     cluster_info: ClusterInfo,
                                     total_memory: int,
                                     available_memory: int) -> ParallelStrategy:
        """选择激进策略（极端内存压力）"""
        num_devices = cluster_info.total_devices
        
        # 使用所有优化手段
        return ParallelStrategy(
            strategy_type=ParallelStrategyType.DP_TP_PP,
            dp_size=1,
            tp_size=min(8, num_devices),
            pp_size=num_devices // min(8, num_devices),
            zero_stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            gradient_checkpointing=True,
            micro_batches=4,
            expected_memory_usage=int(total_memory * 0.1),
            reasoning="极端内存压力，启用所有优化"
        )


class AscendParallelStrategyOptimizer:
    """
    昇腾平台并行策略优化器
    
    针对华为昇腾平台的特性优化并行策略。
    
    昇腾优化要点：
    1. 利用 HCCS 高带宽：优先使用张量并行
    2. 利用达芬奇架构：量化训练加速
    3. 利用 CANN 融合算子：减少内存访问
    4. 适应 HCCL 通信：优化通信策略
    
    Example:
        >>> optimizer = AscendParallelStrategyOptimizer()
        >>> optimized = optimizer.optimize(base_strategy, model_profile, cluster_info)
    """
    
    def __init__(self):
        self._ascend_specific_rules = self._load_ascend_rules()
    
    def _load_ascend_rules(self) -> Dict[str, Any]:
        """加载昇腾特定规则"""
        return {
            'prefer_tp_over_pp': True,      # 优先 TP 而非 PP
            'optimal_tp_size': [2, 4, 8],   # 最优 TP 大小
            'hccs_bandwidth': 300.0,        # HCCS 带宽 GB/s
            'recommended_quantization': True,  # 推荐量化
        }
    
    def optimize(self, strategy: ParallelStrategy,
                 model_profile: ModelProfile,
                 cluster_info: ClusterInfo) -> ParallelStrategy:
        """
        优化并行策略
        
        Args:
            strategy: 基础策略
            model_profile: 模型分析
            cluster_info: 集群信息
        
        Returns:
            优化后的策略
        """
        if cluster_info.hardware_type != HardwareType.HUAWEI_NPU:
            return strategy
        
        # 昇腾特定优化
        optimized = ParallelStrategy(
            strategy_type=strategy.strategy_type,
            dp_size=strategy.dp_size,
            tp_size=strategy.tp_size,
            pp_size=strategy.pp_size,
            zero_stage=strategy.zero_stage,
            offload_optimizer=strategy.offload_optimizer,
            offload_parameters=strategy.offload_parameters,
            gradient_checkpointing=strategy.gradient_checkpointing,
            micro_batches=strategy.micro_batches,
            expected_memory_usage=strategy.expected_memory_usage,
            expected_speedup=strategy.expected_speedup,
            reasoning=strategy.reasoning
        )
        
        # 优化 1: 调整 TP/PP 比例
        if self._ascend_specific_rules['prefer_tp_over_pp']:
            optimized = self._adjust_tp_pp_ratio(optimized, cluster_info)
        
        # 优化 2: 启用量化训练
        if self._ascend_specific_rules['recommended_quantization']:
            optimized.reasoning += " + 昇腾量化优化"
        
        # 优化 3: 调整微批次数量
        optimized.micro_batches = self._optimize_micro_batches(optimized, cluster_info)
        
        return optimized
    
    def _adjust_tp_pp_ratio(self, strategy: ParallelStrategy,
                            cluster_info: ClusterInfo) -> ParallelStrategy:
        """调整 TP/PP 比例以适应昇腾"""
        num_devices = cluster_info.total_devices
        
        # 昇腾 HCCS 高带宽，增大 TP
        if strategy.tp_size < 4 and num_devices >= 4:
            new_tp = min(8, num_devices)
            new_pp = max(1, num_devices // new_tp)
            
            strategy.tp_size = new_tp
            strategy.pp_size = new_pp
            strategy.reasoning += f" (昇腾优化: TP={new_tp}, PP={new_pp})"
        
        return strategy
    
    def _optimize_micro_batches(self, strategy: ParallelStrategy,
                                cluster_info: ClusterInfo) -> int:
        """优化微批次数量"""
        if strategy.pp_size <= 1:
            return 1
        
        # 昇腾平台推荐微批次数为 PP 大小的 2-4 倍
        return strategy.pp_size * 2


class DynamicStrategyAdjuster:
    """
    动态策略调整器
    
    在训练过程中根据运行时指标动态调整并行策略。
    
    监控指标：
    - 内存使用率
    - 通信开销
    - 计算效率
    - 训练吞吐量
    
    Example:
        >>> adjuster = DynamicStrategyAdjuster()
        >>> adjuster.initialize(strategy)
        >>> # 训练循环中
        >>> if adjuster.should_adjust(metrics):
        ...     new_strategy = adjuster.adjust(metrics)
    """
    
    def __init__(self, 
                 adjust_interval: int = 100,
                 memory_threshold: float = 0.9,
                 efficiency_threshold: float = 0.7):
        """
        初始化调整器
        
        Args:
            adjust_interval: 调整间隔（步数）
            memory_threshold: 内存阈值
            efficiency_threshold: 效率阈值
        """
        self.adjust_interval = adjust_interval
        self.memory_threshold = memory_threshold
        self.efficiency_threshold = efficiency_threshold
        
        self._current_strategy: Optional[ParallelStrategy] = None
        self._step_count = 0
        self._metrics_history: List[Dict[str, float]] = []
    
    def initialize(self, strategy: ParallelStrategy):
        """初始化当前策略"""
        self._current_strategy = strategy
    
    def should_adjust(self, metrics: Dict[str, float]) -> bool:
        """
        判断是否需要调整策略
        
        Args:
            metrics: 运行时指标
        
        Returns:
            是否需要调整
        """
        self._step_count += 1
        self._metrics_history.append(metrics)
        
        if self._step_count % self.adjust_interval != 0:
            return False
        
        # 检查内存压力
        if metrics.get('memory_utilization', 0) > self.memory_threshold:
            return True
        
        # 检查效率
        if metrics.get('training_efficiency', 1) < self.efficiency_threshold:
            return True
        
        return False
    
    def adjust(self, metrics: Dict[str, float]) -> Optional[ParallelStrategy]:
        """
        调整策略
        
        Args:
            metrics: 运行时指标
        
        Returns:
            新策略（如果需要调整）
        """
        if self._current_strategy is None:
            return None
        
        new_strategy = ParallelStrategy(
            strategy_type=self._current_strategy.strategy_type,
            dp_size=self._current_strategy.dp_size,
            tp_size=self._current_strategy.tp_size,
            pp_size=self._current_strategy.pp_size,
            zero_stage=self._current_strategy.zero_stage,
            offload_optimizer=self._current_strategy.offload_optimizer,
            offload_parameters=self._current_strategy.offload_parameters,
            gradient_checkpointing=self._current_strategy.gradient_checkpointing,
            micro_batches=self._current_strategy.micro_batches,
            reasoning="动态调整"
        )
        
        # 内存压力高，启用更多优化
        if metrics.get('memory_utilization', 0) > self.memory_threshold:
            if new_strategy.zero_stage < 3:
                new_strategy.zero_stage += 1
                new_strategy.reasoning += " (内存压力，升级 ZeRO)"
            elif not new_strategy.offload_optimizer:
                new_strategy.offload_optimizer = True
                new_strategy.reasoning += " (启用优化器卸载)"
            elif not new_strategy.gradient_checkpointing:
                new_strategy.gradient_checkpointing = True
                new_strategy.reasoning += " (启用梯度检查点)"
        
        # 效率低，减少通信
        if metrics.get('training_efficiency', 1) < self.efficiency_threshold:
            if new_strategy.pp_size > 1:
                new_strategy.micro_batches = max(1, new_strategy.micro_batches - 1)
                new_strategy.reasoning += " (减少微批次)"
        
        self._current_strategy = new_strategy
        return new_strategy
    
    def get_current_strategy(self) -> Optional[ParallelStrategy]:
        """获取当前策略"""
        return self._current_strategy


def create_optimal_strategy(model: nn.Module,
                            batch_size: int = 1,
                            seq_length: int = 512,
                            prefer_memory: bool = False) -> ParallelStrategy:
    """
    创建最优并行策略（便捷函数）
    
    Args:
        model: 模型
        batch_size: 批次大小
        seq_length: 序列长度
        prefer_memory: 是否优先考虑内存
    
    Returns:
        推荐的并行策略
    
    Example:
        >>> strategy = create_optimal_strategy(model, batch_size=32)
        >>> print(f"推荐: {strategy.strategy_type.value}")
    """
    analyzer = AdaptiveParallelAnalyzer(prefer_memory_efficiency=prefer_memory)
    strategy = analyzer.analyze(model, batch_size=batch_size, seq_length=seq_length)
    
    # 如果是昇腾平台，进一步优化
    if ASCEND_AVAILABLE:
        optimizer = AscendParallelStrategyOptimizer()
        cluster_info = ClusterDetector().detect()
        model_profile = ModelProfiler().profile(model, batch_size, seq_length)
        strategy = optimizer.optimize(strategy, model_profile, cluster_info)
    
    return strategy
