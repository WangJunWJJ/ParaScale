# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : qat.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 量化感知训练（QAT）模块

本模块实现了量化感知训练（Quantization Aware Training, QAT），
在训练过程中模拟量化误差，使模型能够适应量化后的推理环境。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .base import QuantizationConfig
from .fake_quantize import FakeQuantize, FakeQuantizedLinear, FakeQuantizedConv2d
from .utils import fuse_modules, get_quantizable_layers, copy_model_weights


class QuantizationAwareTraining:
    """
    量化感知训练类
    
    管理 QAT 的整个流程，包括：
    1. 准备模型（插入伪量化层）
    2. 冻结观察器（停止收集统计信息）
    3. 转换模型（导出为真正的量化模型）
    
    Attributes:
        model: 原始模型
        config: 量化配置
        prepared_model: 准备好的模型（已插入伪量化层）
    
    Example:
        >>> qat = QuantizationAwareTraining(model, config)
        >>> qat.prepare()
        >>> # 训练模型
        >>> for epoch in range(10):
        ...     train(qat.prepared_model)
        >>> qat.freeze_observer()
        >>> quantized_model = qat.convert()
    """
    
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        """
        初始化 QAT
        
        Args:
            model: 原始模型
            config: 量化配置
        """
        self.model = model
        self.config = config
        self.prepared_model: Optional[nn.Module] = None
    
    def prepare(self) -> nn.Module:
        """
        准备模型进行 QAT
        
        1. 融合模块（Conv + BN + ReLU）
        2. 在可量化层插入伪量化层
        
        Returns:
            准备好的模型
        """
        # 复制模型
        self.prepared_model = self._copy_model(self.model)
        
        # 融合模块
        if self.config.fuse_modules:
            fuse_modules(self.prepared_model)
        
        # 插入伪量化层
        self._insert_fake_quantize(self.prepared_model)
        
        # 复制权重
        copy_model_weights(self.model, self.prepared_model)
        
        return self.prepared_model
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """
        复制模型
        
        Args:
            model: 原始模型
        
        Returns:
            模型副本
        """
        import copy
        return copy.deepcopy(model)
    
    def _insert_fake_quantize(self, model: nn.Module) -> None:
        """
        在模型中插入伪量化层
        
        Args:
            model: 目标模型
        """
        quantizable_layers = get_quantizable_layers(model)
        
        for name in quantizable_layers:
            # 获取父模块和层名
            *parent_path, layer_name = name.split('.')
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            
            # 获取原始层
            original_layer = getattr(parent, layer_name)
            
            # 替换为伪量化层
            if isinstance(original_layer, nn.Linear):
                quantized_layer = FakeQuantizedLinear(
                    original_layer.in_features,
                    original_layer.out_features,
                    bias=original_layer.bias is not None,
                    config=self.config
                )
                # 复制权重
                quantized_layer.weight.data = original_layer.weight.data.clone()
                if original_layer.bias is not None:
                    quantized_layer.bias.data = original_layer.bias.data.clone()
                
                setattr(parent, layer_name, quantized_layer)
            
            elif isinstance(original_layer, nn.Conv2d):
                quantized_layer = FakeQuantizedConv2d(
                    original_layer.in_channels,
                    original_layer.out_channels,
                    original_layer.kernel_size,
                    stride=original_layer.stride,
                    padding=original_layer.padding,
                    dilation=original_layer.dilation,
                    groups=original_layer.groups,
                    bias=original_layer.bias is not None,
                    padding_mode=original_layer.padding_mode,
                    config=self.config
                )
                # 复制权重
                quantized_layer.weight.data = original_layer.weight.data.clone()
                if original_layer.bias is not None:
                    quantized_layer.bias.data = original_layer.bias.data.clone()
                
                setattr(parent, layer_name, quantized_layer)
    
    def freeze_observer(self) -> None:
        """
        冻结观察器
        
        停止收集统计信息，固定量化参数。
        通常在训练的最后几个 epoch 调用。
        """
        if self.prepared_model is None:
            raise RuntimeError("模型尚未准备，请先调用 prepare()")
        
        for module in self.prepared_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(False)
    
    def unfreeze_observer(self) -> None:
        """
        解冻观察器
        
        恢复收集统计信息。
        """
        if self.prepared_model is None:
            raise RuntimeError("模型尚未准备，请先调用 prepare()")
        
        for module in self.prepared_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(True)
    
    def enable_fake_quant(self, enabled: bool = True) -> None:
        """
        启用/禁用伪量化
        
        Args:
            enabled: 是否启用
        """
        if self.prepared_model is None:
            raise RuntimeError("模型尚未准备，请先调用 prepare()")
        
        for module in self.prepared_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_fake_quant(enabled)
    
    def convert(self) -> nn.Module:
        """
        转换模型为真正的量化模型
        
        将伪量化模型转换为可以在推理时使用的量化模型。
        
        Returns:
            量化后的模型
        """
        if self.prepared_model is None:
            raise RuntimeError("模型尚未准备，请先调用 prepare()")
        
        # 这里简化处理，实际应该导出为 TorchScript 或使用 PyTorch 的量化转换
        # 在实际部署时，可以使用 torch.quantization.convert
        return self.prepared_model
    
    def get_quantization_params(self) -> Dict[str, Any]:
        """
        获取所有层的量化参数
        
        Returns:
            量化参数字典
        """
        if self.prepared_model is None:
            raise RuntimeError("模型尚未准备，请先调用 prepare()")
        
        params = {}
        for name, module in self.prepared_model.named_modules():
            if isinstance(module, FakeQuantize):
                scale = module.scale
                zero_point = module.zero_point
                params[name] = {
                    'scale': scale.cpu().numpy().tolist(),
                    'zero_point': zero_point.cpu().numpy().tolist(),
                }
        
        return params


def prepare_qat_model(
    model: nn.Module,
    config: QuantizationConfig
) -> nn.Module:
    """
    准备 QAT 模型的便捷函数
    
    Args:
        model: 原始模型
        config: 量化配置
    
    Returns:
        准备好的模型
    """
    qat = QuantizationAwareTraining(model, config)
    return qat.prepare()


def convert_qat_model(
    model: nn.Module,
    config: QuantizationConfig
) -> nn.Module:
    """
    转换 QAT 模型的便捷函数
    
    Args:
        model: 准备好的模型
        config: 量化配置
    
    Returns:
        量化后的模型
    """
    # 冻结观察器
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            module.enable_observer(False)
    
    # 这里简化处理，实际应该导出为 TorchScript
    return model
