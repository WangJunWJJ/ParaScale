# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : degradation.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 量化模块降级机制

本模块实现量化模块的降级机制，当系统资源不足、核心服务负载过高
或量化计算异常时，能够自动触发降级策略，确保系统在降级状态下
仍能保持核心功能可用。

降级级别：
- Level 0: 正常量化（8-bit / 4-bit）
- Level 1: 降低量化精度（8-bit only）
- Level 2: 部分层量化（跳过敏感层）
- Level 3: 禁用量化（FP32 fallback）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable, Union
from enum import IntEnum
import logging
import warnings
import psutil
import gc

logger = logging.getLogger(__name__)


class DegradationLevel(IntEnum):
    """
    量化降级级别枚举
    
    Attributes:
        NONE: 正常量化，无降级
        REDUCE_PRECISION: 降低量化精度（如从4-bit降级到8-bit）
        PARTIAL_QUANTIZATION: 部分层量化（跳过敏感层）
        DISABLED: 完全禁用量化，使用FP32
    """
    NONE = 0              # 正常量化
    REDUCE_PRECISION = 1  # 降低精度
    PARTIAL_QUANTIZATION = 2  # 部分量化
    DISABLED = 3          # 禁用量化


class QuantizationDegradationError(Exception):
    """
    量化降级错误
    
    当量化过程失败且无法降级时抛出。
    """
    pass


class DegradationTrigger:
    """
    降级触发条件检查器
    
    检查系统资源、负载和量化状态，决定是否触发降级。
    
    Attributes:
        memory_threshold: 内存使用阈值（百分比）
        cpu_threshold: CPU使用阈值（百分比）
        min_available_memory_gb: 最小可用内存（GB）
    
    Example:
        >>> trigger = DegradationTrigger(memory_threshold=85.0)
        >>> if trigger.should_degrade():
        ...     logger.warning("System resource low, triggering degradation")
    """
    
    def __init__(
        self,
        memory_threshold: float = 85.0,
        cpu_threshold: float = 90.0,
        min_available_memory_gb: float = 2.0
    ):
        """
        初始化降级触发器
        
        Args:
            memory_threshold: 内存使用阈值（百分比，默认85%）
            cpu_threshold: CPU使用阈值（百分比，默认90%）
            min_available_memory_gb: 最小可用内存（GB，默认2GB）
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.min_available_memory_gb = min_available_memory_gb
    
    def check_memory(self) -> Dict[str, Any]:
        """
        检查内存状态
        
        Returns:
            内存状态字典
        """
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent,
            'is_critical': memory.percent > self.memory_threshold,
            'is_low': memory.available / (1024**3) < self.min_available_memory_gb
        }
    
    def check_cpu(self) -> Dict[str, Any]:
        """
        检查CPU状态
        
        Returns:
            CPU状态字典
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return {
            'percent_used': cpu_percent,
            'is_critical': cpu_percent > self.cpu_threshold
        }
    
    def should_degrade(self) -> bool:
        """
        检查是否应该触发降级
        
        Returns:
            如果应该降级返回True
        """
        memory_status = self.check_memory()
        cpu_status = self.check_cpu()
        
        # 内存临界或CPU临界时触发降级
        return memory_status['is_critical'] or memory_status['is_low'] or cpu_status['is_critical']
    
    def get_degradation_level(self) -> DegradationLevel:
        """
        根据系统状态确定降级级别
        
        Returns:
            建议的降级级别
        """
        memory_status = self.check_memory()
        cpu_status = self.check_cpu()
        
        # 严重资源不足时完全禁用量化
        if memory_status['percent_used'] > 95 or memory_status['available_gb'] < 0.5:
            return DegradationLevel.DISABLED
        
        # 资源紧张时部分量化
        if memory_status['percent_used'] > 90 or cpu_status['is_critical']:
            return DegradationLevel.PARTIAL_QUANTIZATION
        
        # 资源不足时降低精度
        if memory_status['is_critical'] or memory_status['is_low']:
            return DegradationLevel.REDUCE_PRECISION
        
        return DegradationLevel.NONE
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取完整的系统状态
        
        Returns:
            系统状态字典
        """
        return {
            'memory': self.check_memory(),
            'cpu': self.check_cpu(),
            'should_degrade': self.should_degrade(),
            'recommended_level': self.get_degradation_level().name
        }


class QuantizationWithDegradation:
    """
    带降级机制的量化包装器
    
    包装量化过程，在失败时自动触发降级策略。
    
    Attributes:
        model: 待量化模型
        config: 量化配置
        degradation_level: 当前降级级别
        trigger: 降级触发器
        skipped_layers: 被跳过量化的层名列表
    
    Example:
        >>> from parascale.quantization import QuantizationWithDegradation
        >>> from parascale.config import QuantizationConfig
        >>> 
        >>> config = QuantizationConfig(bits=4)
        >>> wrapper = QuantizationWithDegradation(model, config)
        >>> quantized_model = wrapper.quantize_with_fallback()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,  # QuantizationConfig
        trigger: Optional[DegradationTrigger] = None,
        auto_degrade: bool = True
    ):
        """
        初始化降级量化包装器
        
        Args:
            model: 待量化模型
            config: 量化配置
            trigger: 降级触发器（默认自动创建）
            auto_degrade: 是否自动降级
        """
        self.model = model
        self.original_config = config
        self.config = self._copy_config(config)
        self.trigger = trigger or DegradationTrigger()
        self.auto_degrade = auto_degrade
        
        self.degradation_level = DegradationLevel.NONE
        self.skipped_layers: List[str] = []
        self.degradation_history: List[Dict[str, Any]] = []
    
    def _copy_config(self, config: Any) -> Any:
        """
        复制配置对象
        
        Args:
            config: 原始配置
        
        Returns:
            配置副本
        """
        from parascale.config import QuantizationConfig
        if isinstance(config, QuantizationConfig):
            return QuantizationConfig(**config.to_dict())
        return config
    
    def quantize_with_fallback(self, *args, **kwargs) -> nn.Module:
        """
        执行带降级机制的量化
        
        尝试量化，失败时自动降级并重试。
        
        Args:
            *args: 传递给量化函数的位置参数
            **kwargs: 传递给量化函数的关键字参数
        
        Returns:
            量化后的模型（或降级后的模型）
        
        Raises:
            QuantizationDegradationError: 当所有降级级别都失败时
        """
        # 检查系统状态
        if self.auto_degrade and self.trigger.should_degrade():
            recommended_level = self.trigger.get_degradation_level()
            logger.warning(
                f"System resource low (memory: {self.trigger.check_memory()['percent_used']:.1f}%), "
                f"applying degradation level: {recommended_level.name}"
            )
            self.apply_degradation(recommended_level)
        
        # 尝试量化
        for level in self._get_degradation_chain():
            try:
                self.apply_degradation(level)
                model = self._try_quantize(*args, **kwargs)
                
                logger.info(f"Quantization successful at degradation level: {level.name}")
                self._record_success(level)
                return model
                
            except Exception as e:
                logger.warning(f"Quantization failed at level {level.name}: {e}")
                self._record_failure(level, str(e))
                
                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 如果还有下一个降级级别，继续尝试
                continue
        
        # 所有级别都失败
        raise QuantizationDegradationError(
            "Quantization failed at all degradation levels. "
            "Consider using FP32 model or reducing model size."
        )
    
    def _get_degradation_chain(self) -> List[DegradationLevel]:
        """
        获取降级级别链
        
        Returns:
            按优先级排序的降级级别列表
        """
        return [
            DegradationLevel.NONE,
            DegradationLevel.REDUCE_PRECISION,
            DegradationLevel.PARTIAL_QUANTIZATION,
            DegradationLevel.DISABLED
        ]
    
    def _try_quantize(self, *args, **kwargs) -> nn.Module:
        """
        尝试执行量化
        
        Args:
            *args: 量化函数参数
            **kwargs: 量化函数关键字参数
        
        Returns:
            量化后的模型
        
        Raises:
            Exception: 量化失败时抛出
        """
        from parascale.quantization.qat import QuantizationAwareTraining
        from parascale.quantization.ptq import PostTrainingQuantization
        
        # 根据配置选择量化方式
        if self.config.mode == "qat":
            qat = QuantizationAwareTraining(self.model, self.config)
            return qat.prepare()
        else:
            ptq = PostTrainingQuantization(self.model, self.config)
            return ptq.prepare()
    
    def apply_degradation(self, level: DegradationLevel) -> None:
        """
        应用降级级别
        
        Args:
            level: 降级级别
        """
        if level == self.degradation_level:
            return
        
        logger.info(f"Applying degradation level: {level.name}")
        self.degradation_level = level
        
        if level == DegradationLevel.NONE:
            # 恢复正常配置
            self.config = self._copy_config(self.original_config)
            self.skipped_layers = []
            
        elif level == DegradationLevel.REDUCE_PRECISION:
            # 降低精度（4-bit -> 8-bit）
            if self.config.bits == 4:
                self.config.bits = 8
                logger.info("Degraded: Changed from 4-bit to 8-bit quantization")
            
        elif level == DegradationLevel.PARTIAL_QUANTIZATION:
            # 部分量化：跳过敏感层
            self.config.bits = 8
            self.skipped_layers = self._identify_sensitive_layers()
            logger.info(f"Degraded: Partial quantization, skipping {len(self.skipped_layers)} sensitive layers")
            
        elif level == DegradationLevel.DISABLED:
            # 完全禁用量化
            self.config.enabled = False
            logger.info("Degraded: Quantization disabled, using FP32")
    
    def _identify_sensitive_layers(self) -> List[str]:
        """
        识别需要跳过量化的敏感层
        
        Returns:
            敏感层名称列表
        """
        sensitive_layers = []
        
        for name, module in self.model.named_modules():
            # 跳过第一层和最后一层（通常对精度敏感）
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 检查是否是第一层或最后一层
                if self._is_first_or_last_layer(name):
                    sensitive_layers.append(name)
                    continue
                
                # 检查参数量（大层可能更敏感）
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 1000000:  # 大于100万参数
                    sensitive_layers.append(name)
        
        return sensitive_layers
    
    def _is_first_or_last_layer(self, layer_name: str) -> bool:
        """
        检查是否是第一层或最后一层
        
        Args:
            layer_name: 层名称
        
        Returns:
            如果是第一层或最后一层返回True
        """
        # 简化的启发式判断
        first_layer_keywords = ['conv1', 'fc1', 'input', 'embed']
        last_layer_keywords = ['fc', 'classifier', 'output', 'head']
        
        layer_lower = layer_name.lower()
        
        # 检查是否是第一层
        if any(kw in layer_lower for kw in first_layer_keywords):
            # 检查是否包含数字1或在模型顶层
            if '1' in layer_lower or '.' not in layer_name:
                return True
        
        # 检查是否是最后一层
        if any(kw in layer_lower for kw in last_layer_keywords):
            # 检查是否是最后的层
            if layer_name.count('.') <= 1:
                return True
        
        return False
    
    def _record_success(self, level: DegradationLevel) -> None:
        """
        记录成功
        
        Args:
            level: 成功的降级级别
        """
        self.degradation_history.append({
            'level': level.name,
            'success': True,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        })
    
    def _record_failure(self, level: DegradationLevel, error: str) -> None:
        """
        记录失败
        
        Args:
            level: 失败的降级级别
            error: 错误信息
        """
        self.degradation_history.append({
            'level': level.name,
            'success': False,
            'error': error
        })
    
    def get_degradation_report(self) -> Dict[str, Any]:
        """
        获取降级报告
        
        Returns:
            降级报告字典
        """
        return {
            'current_level': self.degradation_level.name,
            'original_bits': self.original_config.bits if hasattr(self.original_config, 'bits') else None,
            'current_bits': self.config.bits if hasattr(self.config, 'bits') else None,
            'skipped_layers': self.skipped_layers,
            'history': self.degradation_history,
            'system_status': self.trigger.get_system_status()
        }
    
    def print_degradation_report(self) -> None:
        """
        打印降级报告
        """
        report = self.get_degradation_report()
        
        print("=" * 60)
        print("Quantization Degradation Report")
        print("=" * 60)
        print(f"Current Level: {report['current_level']}")
        print(f"Original Bits: {report['original_bits']}")
        print(f"Current Bits: {report['current_bits']}")
        print(f"Skipped Layers: {len(report['skipped_layers'])}")
        if report['skipped_layers']:
            for layer in report['skipped_layers'][:5]:
                print(f"  - {layer}")
            if len(report['skipped_layers']) > 5:
                print(f"  ... and {len(report['skipped_layers']) - 5} more")
        print("-" * 60)
        print("Degradation History:")
        for entry in report['history']:
            status = "✓" if entry['success'] else "✗"
            print(f"  {status} {entry['level']}")
        print("=" * 60)


def quantize_with_fallback(
    model: nn.Module,
    config: Any,
    *args,
    **kwargs
) -> nn.Module:
    """
    带降级机制的量化便捷函数
    
    一站式完成带降级机制的量化。
    
    Args:
        model: 待量化模型
        config: 量化配置
        *args: 量化函数参数
        **kwargs: 额外参数
            - auto_degrade: 是否自动降级（默认True）
            - memory_threshold: 内存阈值（默认85.0）
    
    Returns:
        量化后的模型
    
    Example:
        >>> from parascale.quantization import quantize_with_fallback
        >>> from parascale.config import QuantizationConfig
        >>> 
        >>> config = QuantizationConfig(bits=4)
        >>> quantized_model = quantize_with_fallback(model, config)
    """
    auto_degrade = kwargs.pop('auto_degrade', True)
    memory_threshold = kwargs.pop('memory_threshold', 85.0)
    
    trigger = DegradationTrigger(memory_threshold=memory_threshold)
    wrapper = QuantizationWithDegradation(
        model,
        config,
        trigger=trigger,
        auto_degrade=auto_degrade
    )
    
    return wrapper.quantize_with_fallback(*args, **kwargs)


class AdaptiveQuantization:
    """
    自适应量化管理器
    
    根据系统资源动态调整量化策略。
    
    Example:
        >>> from parascale.quantization import AdaptiveQuantization
        >>> 
        >>> adaptive = AdaptiveQuantization(model)
        >>> quantized_model = adaptive.quantize_if_feasible()
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_config: Optional[Any] = None
    ):
        """
        初始化自适应量化管理器
        
        Args:
            model: 待量化模型
            base_config: 基础量化配置
        """
        self.model = model
        self.base_config = base_config
        self.trigger = DegradationTrigger()
    
    def quantize_if_feasible(
        self,
        min_memory_gb: float = 4.0,
        *args,
        **kwargs
    ) -> Optional[nn.Module]:
        """
        在资源充足时执行量化
        
        Args:
            min_memory_gb: 最小所需内存（GB）
            *args: 量化函数参数
            **kwargs: 量化函数关键字参数
        
        Returns:
            量化后的模型，如果资源不足返回None
        """
        memory_status = self.trigger.check_memory()
        
        if memory_status['available_gb'] < min_memory_gb:
            logger.warning(
                f"Insufficient memory for quantization: "
                f"{memory_status['available_gb']:.2f}GB available, "
                f"{min_memory_gb}GB required"
            )
            return None
        
        if self.base_config is None:
            from parascale.config import QuantizationConfig
            self.base_config = QuantizationConfig()
        
        return quantize_with_fallback(
            self.model,
            self.base_config,
            *args,
            **kwargs
        )
    
    def get_recommendation(self) -> Dict[str, Any]:
        """
        获取量化建议
        
        Returns:
            建议字典
        """
        system_status = self.trigger.get_system_status()
        
        recommendation = {
            'should_quantize': True,
            'recommended_bits': 8,
            'reason': ''
        }
        
        memory = system_status['memory']
        
        if memory['available_gb'] < 1.0:
            recommendation['should_quantize'] = False
            recommendation['reason'] = 'Insufficient memory for quantization'
        elif memory['available_gb'] < 4.0:
            recommendation['recommended_bits'] = 8
            recommendation['reason'] = 'Limited memory, recommend 8-bit quantization'
        else:
            recommendation['recommended_bits'] = 4
            recommendation['reason'] = 'Sufficient memory, can use 4-bit quantization'
        
        return recommendation
