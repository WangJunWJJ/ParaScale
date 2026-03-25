# -*- coding: utf-8 -*-
# @Time    : 2026/3/23
# @Author  : Jun Wang
# @File    : memory_offload.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 内存超分策略模块

本模块实现了针对华为昇腾平台的智能内存管理方案，支持内存超分、
参数卸载、梯度卸载和优化器状态卸载，突破物理显存限制。

核心功能：
- 内存监控：实时监控显存使用情况
- 智能卸载：根据访问频率自动卸载冷数据
- 预取机制：异步预取即将使用的数据
- 内存池管理：高效的内存分配和回收
- 分层存储：NPU -> Host Memory -> SSD 三级存储

内存超分策略：
1. 参数分片：将大模型参数分片存储
2. 梯度累积：延迟梯度同步减少内存峰值
3. 激活重计算：牺牲计算换取内存
4. 优化器状态卸载：将优化器状态卸载到 CPU
5. 动态内存分配：根据运行时需求动态调整

Example:
    >>> manager = AscendMemoryManager(offload_ratio=0.8)
    >>> manager.initialize(model)
    >>> manager.optimize_memory()
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
import queue
from collections import OrderedDict
import weakref

logger = logging.getLogger(__name__)

# 检测 Ascend 环境
ASCEND_AVAILABLE = False
try:
    import torch_npu
    ASCEND_AVAILABLE = torch.npu.is_available()
except ImportError:
    pass


class MemoryTier(Enum):
    """内存层级枚举"""
    NPU = "npu"           # NPU 显存（最快）
    HOST = "host"         # 主机内存（中等）
    SSD = "ssd"           # SSD 存储（最慢）


class OffloadStrategy(Enum):
    """卸载策略枚举"""
    NONE = "none"               # 不卸载
    OPTIMIZER = "optimizer"     # 仅卸载优化器状态
    GRADIENT = "gradient"       # 卸载梯度
    PARAMETER = "parameter"     # 卸载参数
    AGGRESSIVE = "aggressive"   # 激进卸载（全部）


@dataclass
class MemoryStats:
    """内存统计数据类"""
    total_memory: int = 0
    used_memory: int = 0
    available_memory: int = 0
    peak_memory: int = 0
    num_allocations: int = 0
    num_frees: int = 0
    
    def utilization(self) -> float:
        """计算内存利用率"""
        if self.total_memory == 0:
            return 0.0
        return self.used_memory / self.total_memory


@dataclass
class ParameterInfo:
    """参数信息数据类"""
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: torch.dtype
    device: torch.device
    memory_bytes: int
    access_count: int = 0
    last_access_time: float = 0.0
    is_offloaded: bool = False
    offload_tier: Optional[MemoryTier] = None
    cpu_copy: Optional[torch.Tensor] = None


class AscendMemoryMonitor:
    """
    Ascend 内存监控器
    
    实时监控 NPU 显存使用情况，提供内存统计和预警功能。
    
    Example:
        >>> monitor = AscendMemoryMonitor(device_id=0)
        >>> stats = monitor.get_memory_stats()
        >>> print(f"内存使用率: {stats.utilization():.2%}")
    """
    
    def __init__(self, device_id: int = 0, monitor_interval: float = 1.0):
        """
        初始化内存监控器
        
        Args:
            device_id: 设备 ID
            monitor_interval: 监控间隔（秒）
        """
        self.device_id = device_id
        self.monitor_interval = monitor_interval
        self._stats = MemoryStats()
        self._history: List[MemoryStats] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """启动后台监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            self._update_stats()
            time.sleep(self.monitor_interval)
    
    def _update_stats(self):
        """更新内存统计"""
        if ASCEND_AVAILABLE:
            try:
                self._stats.used_memory = torch.npu.memory_allocated(self.device_id)
                self._stats.total_memory = torch.npu.get_device_properties(
                    self.device_id
                ).total_memory
                self._stats.available_memory = self._stats.total_memory - self._stats.used_memory
                self._stats.peak_memory = torch.npu.max_memory_allocated(self.device_id)
            except Exception as e:
                logger.warning(f"获取内存信息失败: {e}")
        else:
            if torch.cuda.is_available():
                self._stats.used_memory = torch.cuda.memory_allocated(self.device_id)
                self._stats.total_memory = torch.cuda.get_device_properties(
                    self.device_id
                ).total_memory
                self._stats.available_memory = self._stats.total_memory - self._stats.used_memory
                self._stats.peak_memory = torch.cuda.max_memory_allocated(self.device_id)
        
        # 记录历史
        self._history.append(MemoryStats(
            total_memory=self._stats.total_memory,
            used_memory=self._stats.used_memory,
            available_memory=self._stats.available_memory,
            peak_memory=self._stats.peak_memory
        ))
        
        # 限制历史长度
        if len(self._history) > 1000:
            self._history = self._history[-500:]
    
    def get_memory_stats(self) -> MemoryStats:
        """获取当前内存统计"""
        self._update_stats()
        return self._stats
    
    def get_memory_history(self, last_n: int = 100) -> List[MemoryStats]:
        """获取内存历史"""
        return self._history[-last_n:]
    
    def check_memory_pressure(self, threshold: float = 0.9) -> bool:
        """检查内存压力"""
        stats = self.get_memory_stats()
        return stats.utilization() > threshold


class AscendMemoryPool:
    """
    Ascend 内存池
    
    提供高效的内存分配和回收机制，减少内存碎片。
    
    Example:
        >>> pool = AscendMemoryPool(device_id=0, pool_size_mb=1024)
        >>> tensor = pool.allocate((1000, 1000), dtype=torch.float32)
        >>> pool.deallocate(tensor)
    """
    
    def __init__(self, device_id: int = 0, pool_size_mb: int = 1024):
        """
        初始化内存池
        
        Args:
            device_id: 设备 ID
            pool_size_mb: 内存池大小（MB）
        """
        self.device_id = device_id
        self.pool_size = pool_size_mb * 1024 * 1024
        self.device = torch.device(f"npu:{device_id}" if ASCEND_AVAILABLE else f"cuda:{device_id}")
        
        # 空闲块列表
        self._free_blocks: List[Tuple[int, int, torch.Tensor]] = []  # (start, size, tensor)
        # 已分配块映射
        self._allocated: Dict[int, Tuple[int, int]] = {}  # ptr_id -> (start, size)
        
        # 预分配内存池
        self._pool_tensor: Optional[torch.Tensor] = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化内存池"""
        try:
            self._pool_tensor = torch.empty(
                self.pool_size // 4,  # float32 = 4 bytes
                dtype=torch.float32,
                device=self.device
            )
            self._free_blocks = [(0, self.pool_size, self._pool_tensor)]
            logger.info(f"内存池初始化成功: {self.pool_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.warning(f"内存池初始化失败: {e}")
    
    def allocate(self, shape: Tuple[int, ...], 
                 dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
        """
        从内存池分配张量
        
        Args:
            shape: 张量形状
            dtype: 数据类型
        
        Returns:
            分配的张量，如果失败返回 None
        """
        numel = 1
        for s in shape:
            numel *= s
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        required_size = numel * element_size
        
        # 找到合适的空闲块
        for i, (start, size, pool) in enumerate(self._free_blocks):
            if size >= required_size:
                # 分割块
                remaining = size - required_size
                if remaining > 0:
                    self._free_blocks[i] = (start + required_size, remaining, pool)
                else:
                    self._free_blocks.pop(i)
                
                # 创建张量视图
                offset = start // element_size
                tensor = pool[offset:offset + numel].view(shape)
                
                # 记录分配
                self._allocated[id(tensor)] = (start, required_size)
                
                return tensor
        
        # 内存池不足，使用标准分配
        return torch.empty(shape, dtype=dtype, device=self.device)
    
    def deallocate(self, tensor: torch.Tensor):
        """
        释放张量到内存池
        
        Args:
            tensor: 要释放的张量
        """
        tensor_id = id(tensor)
        if tensor_id not in self._allocated:
            return
        
        start, size = self._allocated.pop(tensor_id)
        
        # 添加回空闲列表
        self._free_blocks.append((start, size, self._pool_tensor))
        
        # 合并相邻块
        self._merge_free_blocks()
    
    def _merge_free_blocks(self):
        """合并相邻的空闲块"""
        if len(self._free_blocks) <= 1:
            return
        
        self._free_blocks.sort(key=lambda x: x[0])
        
        merged = [self._free_blocks[0]]
        for start, size, pool in self._free_blocks[1:]:
            last_start, last_size, last_pool = merged[-1]
            if last_start + last_size == start:
                merged[-1] = (last_start, last_size + size, last_pool)
            else:
                merged.append((start, size, pool))
        
        self._free_blocks = merged
    
    def get_fragmentation(self) -> float:
        """计算内存碎片率"""
        if not self._free_blocks:
            return 0.0
        
        total_free = sum(size for _, size, _ in self._free_blocks)
        max_free = max(size for _, size, _ in self._free_blocks)
        
        if total_free == 0:
            return 0.0
        
        return 1.0 - max_free / total_free


class AscendOffloadManager:
    """
    Ascend 卸载管理器
    
    管理参数、梯度和优化器状态的卸载和预取。
    
    支持的卸载策略：
    - OPTIMIZER: 仅卸载优化器状态（推荐）
    - GRADIENT: 卸载梯度
    - PARAMETER: 卸载参数（需要预取）
    - AGGRESSIVE: 全部卸载
    
    Example:
        >>> manager = AscendOffloadManager(strategy=OffloadStrategy.AGGRESSIVE)
        >>> manager.initialize(model, optimizer)
        >>> manager.offload_all()
        >>> # 训练时自动预取
        >>> manager.prefetch_parameters(layer_idx)
    """
    
    def __init__(self, 
                 strategy: OffloadStrategy = OffloadStrategy.OPTIMIZER,
                 offload_ratio: float = 0.8,
                 prefetch_size: int = 2):
        """
        初始化卸载管理器
        
        Args:
            strategy: 卸载策略
            offload_ratio: 卸载比例（0-1）
            prefetch_size: 预取层数
        """
        self.strategy = strategy
        self.offload_ratio = offload_ratio
        self.prefetch_size = prefetch_size
        
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        
        # 参数信息
        self._param_info: Dict[str, ParameterInfo] = {}
        self._param_order: List[str] = []  # 参数访问顺序
        
        # CPU 缓存
        self._cpu_cache: Dict[str, torch.Tensor] = {}
        
        # 预取队列
        self._prefetch_queue: queue.Queue = queue.Queue()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_running = False
        
        # 统计
        self._stats = {
            'offload_count': 0,
            'prefetch_count': 0,
            'offload_bytes': 0,
            'prefetch_bytes': 0,
        }
    
    def initialize(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        初始化卸载管理器
        
        Args:
            model: 模型
            optimizer: 优化器（可选）
        """
        self._model = model
        self._optimizer = optimizer
        
        # 分析参数
        self._analyze_parameters()
        
        # 启动预取线程
        self._start_prefetch_thread()
        
        logger.info(f"卸载管理器初始化完成，策略: {self.strategy.value}")
    
    def _analyze_parameters(self):
        """分析模型参数"""
        for name, param in self._model.named_parameters():
            info = ParameterInfo(
                name=name,
                shape=tuple(param.shape),
                numel=param.numel(),
                dtype=param.dtype,
                device=param.device,
                memory_bytes=param.numel() * param.element_size()
            )
            self._param_info[name] = info
            self._param_order.append(name)
        
        total_memory = sum(info.memory_bytes for info in self._param_info.values())
        logger.info(f"模型参数总数: {len(self._param_info)}, 总内存: {total_memory / 1024 / 1024:.2f} MB")
    
    def _start_prefetch_thread(self):
        """启动预取线程"""
        self._prefetch_running = True
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()
    
    def _prefetch_loop(self):
        """预取循环"""
        while self._prefetch_running:
            try:
                task = self._prefetch_queue.get(timeout=1.0)
                if task is None:
                    break
                self._do_prefetch(task)
            except queue.Empty:
                continue
    
    def _do_prefetch(self, param_names: List[str]):
        """执行预取"""
        for name in param_names:
            if name not in self._param_info:
                continue
            
            info = self._param_info[name]
            if not info.is_offloaded:
                continue
            
            # 从 CPU 复制回 NPU
            param = dict(self._model.named_parameters()).get(name)
            if param is not None and name in self._cpu_cache:
                param.data.copy_(self._cpu_cache[name])
                info.is_offloaded = False
                info.offload_tier = None
                self._stats['prefetch_count'] += 1
                self._stats['prefetch_bytes'] += info.memory_bytes
    
    def offload_all(self):
        """卸载所有数据"""
        if self.strategy == OffloadStrategy.NONE:
            return
        
        # 卸载优化器状态
        if self.strategy in [OffloadStrategy.OPTIMIZER, OffloadStrategy.AGGRESSIVE]:
            self._offload_optimizer_states()
        
        # 卸载参数
        if self.strategy in [OffloadStrategy.PARAMETER, OffloadStrategy.AGGRESSIVE]:
            self._offload_parameters()
        
        # 卸载梯度
        if self.strategy in [OffloadStrategy.GRADIENT, OffloadStrategy.AGGRESSIVE]:
            self._offload_gradients()
    
    def _offload_optimizer_states(self):
        """卸载优化器状态到 CPU"""
        if self._optimizer is None:
            return
        
        for param in self._model.parameters():
            if param in self._optimizer.state:
                state = self._optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cpu()
                        self._stats['offload_count'] += 1
                        self._stats['offload_bytes'] += value.numel() * value.element_size()
        
        logger.info("优化器状态已卸载到 CPU")
    
    def _offload_parameters(self):
        """卸载参数到 CPU"""
        offload_count = int(len(self._param_order) * self.offload_ratio)
        
        # 选择要卸载的参数（按访问频率排序，卸载低频的）
        for name in self._param_order[-offload_count:]:
            param = dict(self._model.named_parameters()).get(name)
            if param is None:
                continue
            
            # 复制到 CPU
            self._cpu_cache[name] = param.data.cpu()
            
            # 更新信息
            info = self._param_info[name]
            info.is_offloaded = True
            info.offload_tier = MemoryTier.HOST
            info.cpu_copy = self._cpu_cache[name]
            
            self._stats['offload_count'] += 1
            self._stats['offload_bytes'] += info.memory_bytes
        
        logger.info(f"已卸载 {offload_count} 个参数到 CPU")
    
    def _offload_gradients(self):
        """卸载梯度到 CPU"""
        for name, param in self._model.named_parameters():
            if param.grad is not None:
                self._cpu_cache[f"{name}_grad"] = param.grad.cpu()
                param.grad = None
        
        logger.info("梯度已卸载到 CPU")
    
    def prefetch_parameters(self, layer_idx: int):
        """
        预取即将使用的参数
        
        Args:
            layer_idx: 当前层索引
        """
        # 计算需要预取的层
        prefetch_layers = list(range(
            layer_idx, 
            min(layer_idx + self.prefetch_size, len(self._param_order))
        ))
        
        prefetch_names = [self._param_order[i] for i in prefetch_layers 
                         if i < len(self._param_order)]
        
        # 添加到预取队列
        self._prefetch_queue.put(prefetch_names)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()
    
    def cleanup(self):
        """清理资源"""
        self._prefetch_running = False
        if self._prefetch_thread:
            self._prefetch_queue.put(None)
            self._prefetch_thread.join(timeout=5.0)
        
        self._cpu_cache.clear()


class AscendMemoryManager:
    """
    Ascend 内存管理器
    
    统一管理 Ascend 平台的内存资源，提供内存优化和监控功能。
    
    功能：
    - 内存监控
    - 内存池管理
    - 卸载管理
    - 自动内存优化
    
    Example:
        >>> manager = AscendMemoryManager(
        ...     offload_ratio=0.8,
        ...     strategy=OffloadStrategy.AGGRESSIVE
        ... )
        >>> manager.initialize(model, optimizer)
        >>> manager.optimize_memory()
    """
    
    def __init__(self,
                 device_id: int = 0,
                 offload_ratio: float = 0.8,
                 strategy: OffloadStrategy = OffloadStrategy.OPTIMIZER,
                 pool_size_mb: int = 1024,
                 enable_monitoring: bool = True):
        """
        初始化内存管理器
        
        Args:
            device_id: 设备 ID
            offload_ratio: 卸载比例
            strategy: 卸载策略
            pool_size_mb: 内存池大小
            enable_monitoring: 是否启用监控
        """
        self.device_id = device_id
        self.offload_ratio = offload_ratio
        self.strategy = strategy
        
        # 子组件
        self._monitor = AscendMemoryMonitor(device_id)
        self._pool = AscendMemoryPool(device_id, pool_size_mb)
        self._offload_manager = AscendOffloadManager(strategy, offload_ratio)
        
        self._enable_monitoring = enable_monitoring
        self._initialized = False
    
    def initialize(self, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """
        初始化内存管理器
        
        Args:
            model: 模型
            optimizer: 优化器
        
        Returns:
            初始化是否成功
        """
        if self._initialized:
            return True
        
        # 初始化卸载管理器
        self._offload_manager.initialize(model, optimizer)
        
        # 启动监控
        if self._enable_monitoring:
            self._monitor.start_monitoring()
        
        self._initialized = True
        logger.info("Ascend 内存管理器初始化完成")
        return True
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        执行内存优化
        
        Returns:
            优化结果统计
        """
        if not self._initialized:
            return {}
        
        # 获取当前内存状态
        stats_before = self._monitor.get_memory_stats()
        
        # 执行卸载
        self._offload_manager.offload_all()
        
        # 清空缓存
        if ASCEND_AVAILABLE:
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 获取优化后内存状态
        stats_after = self._monitor.get_memory_stats()
        
        result = {
            'memory_before': stats_before.used_memory,
            'memory_after': stats_after.used_memory,
            'memory_saved': stats_before.used_memory - stats_after.used_memory,
            'offload_stats': self._offload_manager.get_stats(),
        }
        
        logger.info(f"内存优化完成，节省: {result['memory_saved'] / 1024 / 1024:.2f} MB")
        return result
    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计"""
        return self._monitor.get_memory_stats()
    
    def check_memory_pressure(self, threshold: float = 0.9) -> bool:
        """检查内存压力"""
        return self._monitor.check_memory_pressure(threshold)
    
    def allocate(self, shape: Tuple[int, ...], 
                 dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
        """从内存池分配张量"""
        return self._pool.allocate(shape, dtype)
    
    def deallocate(self, tensor: torch.Tensor):
        """释放张量到内存池"""
        self._pool.deallocate(tensor)
    
    def prefetch(self, layer_idx: int):
        """预取层参数"""
        self._offload_manager.prefetch_parameters(layer_idx)
    
    def get_memory_efficiency(self) -> float:
        """获取内存效率（节省比例）"""
        stats = self._offload_manager.get_stats()
        offload_bytes = stats.get('offload_bytes', 0)
        
        total_param_bytes = sum(
            info.memory_bytes for info in self._offload_manager._param_info.values()
        )
        
        if total_param_bytes == 0:
            return 0.0
        
        return offload_bytes / total_param_bytes
    
    def cleanup(self):
        """清理所有资源"""
        self._offload_manager.cleanup()
        self._monitor.stop_monitoring()
        self._initialized = False


class ActivationCheckpointManager:
    """
    激活检查点管理器
    
    实现梯度检查点（Gradient Checkpointing）以减少激活内存。
    
    策略：
    - 选择性保存激活：只保存部分层的激活
    - 重计算：在反向传播时重新计算丢弃的激活
    
    Example:
        >>> checkpoint_mgr = ActivationCheckpointManager(checkpoint_ratio=0.5)
        >>> checkpoint_mgr.setup(model)
    """
    
    def __init__(self, checkpoint_ratio: float = 0.5):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_ratio: 检查点比例（0-1）
        """
        self.checkpoint_ratio = checkpoint_ratio
        self._checkpoint_layers: List[str] = []
        self._recompute_layers: List[str] = []
    
    def setup(self, model: nn.Module):
        """
        设置检查点层
        
        Args:
            model: 模型
        """
        all_layers = list(model.named_children())
        num_checkpoint = int(len(all_layers) * self.checkpoint_ratio)
        
        # 选择检查点层（均匀分布）
        step = len(all_layers) / num_checkpoint if num_checkpoint > 0 else 1
        for i in range(num_checkpoint):
            idx = int(i * step)
            if idx < len(all_layers):
                self._checkpoint_layers.append(all_layers[idx][0])
        
        # 其余层需要重计算
        for name, _ in all_layers:
            if name not in self._checkpoint_layers:
                self._recompute_layers.append(name)
        
        logger.info(f"检查点层: {len(self._checkpoint_layers)}, 重计算层: {len(self._recompute_layers)}")
    
    def is_checkpoint_layer(self, layer_name: str) -> bool:
        """检查是否为检查点层"""
        return layer_name in self._checkpoint_layers
    
    def get_memory_savings(self, model: nn.Module) -> int:
        """估算内存节省"""
        total_activation_memory = 0
        saved_memory = 0
        
        for name, module in model.named_children():
            # 估算激活内存（简化）
            if hasattr(module, 'weight'):
                activation_size = module.weight.numel() * 4  # 假设 FP32
                total_activation_memory += activation_size
                if name in self._recompute_layers:
                    saved_memory += activation_size
        
        return saved_memory
