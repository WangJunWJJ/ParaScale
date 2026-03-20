# -*- coding: utf-8 -*-
# @Time    : 2026/3/13
# @Author  : Jun Wang
# @File    : ptq.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 训练后量化（PTQ）模块

本模块实现了训练后量化（Post-Training Quantization, PTQ），
无需重新训练即可将模型量化为 INT8/INT4，适用于快速部署和边缘设备。

PTQ 流程：
1. 加载预训练模型（FP32）
2. 准备校准数据集（少量代表性样本）
3. 插入观察器，收集激活值统计信息
4. 执行校准（Calibration）：前向传播，不反向传播
5. 计算量化参数（scale 和 zero_point）
6. 量化权重
7. 导出量化模型（INT8）

Example:
    >>> from parascale import PostTrainingQuantization, QuantizationConfig
    >>> model = PreTrainedModel()
    >>> config = QuantizationConfig(mode="ptq", bits=8, calib_batches=100)
    >>> ptq = PostTrainingQuantization(model, config)
    >>> ptq.prepare()
    >>> ptq.calibrate(calib_loader)
    >>> quantized_model = ptq.convert()
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any, Tuple
from .base import QuantizationConfig
from .fake_quantize import FakeQuantize, FakeQuantizedLinear, FakeQuantizedConv2d
from .utils import fuse_modules, get_quantizable_layers, copy_model_weights
from .observers import MinMaxObserver, MovingAverageObserver
import logging

logger = logging.getLogger(__name__)


class PostTrainingQuantization:
    """
    训练后量化（PTQ）类
    
    管理 PTQ 的整个流程，包括模型准备、校准、权重量化和模型转换。
    
    Attributes:
        model: 原始预训练模型（FP32）
        config: 量化配置
        calibrated_model: 校准后的模型
        is_calibrated: 是否已完成校准
        is_converted: 是否已完成转换
    
    Example:
        >>> model = ResNet18(pretrained=True)
        >>> config = QuantizationConfig(mode="ptq", bits=8)
        >>> ptq = PostTrainingQuantization(model, config)
        >>> ptq.prepare()
        >>> ptq.calibrate(calib_loader)
        >>> quantized_model = ptq.convert()
    """
    
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        """
        初始化 PTQ
        
        Args:
            model: 预训练模型
            config: 量化配置
        """
        if config.mode != "ptq":
            logger.warning(f"配置模式为 {config.mode}，但正在使用 PTQ。建议设置 mode='ptq'")
        
        self.model = model
        self.config = config
        self.calibrated_model: Optional[nn.Module] = None
        self.is_calibrated = False
        self.is_converted = False
    
    def prepare(self) -> nn.Module:
        """
        准备模型进行 PTQ
        
        步骤：
        1. 复制模型
        2. 融合模块（Conv + BN + ReLU）
        3. 插入伪量化层
        4. 复制权重
        
        Returns:
            准备好的模型
        """
        logger.info("准备 PTQ 模型...")
        
        # 复制模型
        self.calibrated_model = self._copy_model(self.model)
        
        # 融合模块
        if self.config.fuse_modules:
            logger.info("正在融合模块...")
            fuse_modules(self.calibrated_model)
        
        # 插入伪量化层
        logger.info("正在插入伪量化层...")
        self._insert_fake_quantize(self.calibrated_model)
        
        # 复制权重
        logger.info("正在复制权重...")
        copy_model_weights(self.model, self.calibrated_model)
        
        logger.info(f"PTQ 模型准备完成，共插入 {len(get_quantizable_layers(self.calibrated_model))} 个伪量化层")
        return self.calibrated_model
    
    def calibrate(
        self, 
        calib_loader,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        校准模型
        
        使用校准数据集收集激活值和权重的统计信息。
        
        Args:
            calib_loader: 校准数据加载器
            progress_callback: 进度回调函数，接收 (current_batch, total_batches)
        
        Raises:
            RuntimeError: 如果尚未调用 prepare()
        """
        if self.calibrated_model is None:
            raise RuntimeError("请先调用 prepare() 准备模型")
        
        logger.info(f"开始 PTQ 校准，使用 {self.config.calib_batches} 个批次...")
        
        self.calibrated_model.eval()
        
        total_batches = min(self.config.calib_batches, len(calib_loader))
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(calib_loader):
                if batch_idx >= total_batches:
                    break
                
                # 支持不同的数据格式
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0]
                else:
                    inputs = batch_data
                
                # 前向传播（只收集统计信息，不计算损失）
                _ = self.calibrated_model(inputs)
                
                # 更新进度
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches)
                elif (batch_idx + 1) % 10 == 0:
                    logger.info(f"  校准进度：{batch_idx+1}/{total_batches}")
        
        logger.info("校准完成！")
        self.is_calibrated = True
        
        # 冻结观察器
        logger.info("正在冻结观察器...")
        self.freeze_observer()
    
    def freeze_observer(self) -> None:
        """
        冻结观察器
        
        停止收集统计信息，固定量化参数。
        """
        if self.calibrated_model is None:
            raise RuntimeError("请先调用 prepare()")
        
        count = 0
        for module in self.calibrated_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(False)
                count += 1
        
        logger.info(f"已冻结 {count} 个观察器")
    
    def quantize_weights(self) -> nn.Module:
        """
        量化权重
        
        将模型权重转换为量化表示（模拟量化）。
        
        Returns:
            权重量化后的模型
        
        Raises:
            RuntimeError: 如果尚未校准
        """
        if not self.is_calibrated:
            raise RuntimeError("请先校准模型")
        
        logger.info("正在量化权重...")
        
        count = 0
        for name, module in self.calibrated_model.named_modules():
            if isinstance(module, (FakeQuantizedLinear, FakeQuantizedConv2d)):
                # 获取权重量化器
                weight_quant = module.weight_fake_quant
                
                with torch.no_grad():
                    # 更新观察器（使用当前权重）
                    weight_quant.observer.update(module.weight.detach())
                    
                    # 计算量化参数
                    scale, zero_point = weight_quant.observer.calculate_qparams()
                    weight_quant.scale = scale
                    weight_quant.zero_point = zero_point
                    
                    # 量化并反量化权重（模拟量化误差）
                    module.weight.data = weight_quant(module.weight.data)
                    
                    count += 1
        
        logger.info(f"已量化 {count} 个层的权重")
        return self.calibrated_model
    
    def convert(self) -> nn.Module:
        """
        转换为真正的量化模型
        
        将伪量化模型转换为可以在推理时使用的量化模型。
        
        Returns:
            量化模型
        
        Raises:
            RuntimeError: 如果尚未校准
        """
        if not self.is_calibrated:
            raise RuntimeError("请先校准模型")
        
        logger.info("正在转换模型...")
        
        # 量化权重
        self.quantize_weights()
        
        # 禁用伪量化（推理时不需要）
        self._disable_fake_quant()
        
        self.is_converted = True
        logger.info("模型转换完成！")
        
        return self.calibrated_model
    
    def _disable_fake_quant(self) -> None:
        """禁用所有伪量化层"""
        if self.calibrated_model is None:
            return
        
        for module in self.calibrated_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_fake_quant(False)
    
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
            *parent_path, layer_name = name.split('.')
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            
            original_layer = getattr(parent, layer_name)
            
            if isinstance(original_layer, nn.Linear):
                quantized_layer = FakeQuantizedLinear(
                    original_layer.in_features,
                    original_layer.out_features,
                    bias=original_layer.bias is not None,
                    config=self.config
                )
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
                quantized_layer.weight.data = original_layer.weight.data.clone()
                if original_layer.bias is not None:
                    quantized_layer.bias.data = original_layer.bias.data.clone()
                
                setattr(parent, layer_name, quantized_layer)
    
    def get_quantization_params(self) -> Dict[str, Any]:
        """
        获取所有层的量化参数
        
        Returns:
            量化参数字典，格式：{layer_name: {'scale': ..., 'zero_point': ...}}
        
        Raises:
            RuntimeError: 如果尚未校准
        """
        if not self.is_calibrated:
            raise RuntimeError("请先校准模型")
        
        params = {}
        for name, module in self.calibrated_model.named_modules():
            if isinstance(module, FakeQuantize):
                scale = module.scale
                zero_point = module.zero_point
                params[name] = {
                    'scale': scale.cpu().numpy().tolist(),
                    'zero_point': zero_point.cpu().numpy().tolist(),
                    'min_val': module.observer.min_val.cpu().numpy().tolist() 
                               if module.observer.min_val is not None else None,
                    'max_val': module.observer.max_val.cpu().numpy().tolist()
                               if module.observer.max_val is not None else None,
                }
        
        return params
    
    def export(self, save_path: str) -> None:
        """
        导出量化模型
        
        Args:
            save_path: 保存路径
        
        Raises:
            RuntimeError: 如果尚未转换
        """
        if not self.is_converted:
            raise RuntimeError("请先转换模型")
        
        quant_params = self.get_quantization_params()
        
        checkpoint = {
            'model_state_dict': self.calibrated_model.state_dict(),
            'quantization_params': quant_params,
            'config': self.config.to_dict(),
            'is_quantized': True,
            'quantization_type': 'ptq'
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"量化模型已导出到：{save_path}")
    
    def evaluate(
        self, 
        test_loader, 
        criterion: nn.Module,
        device: Optional[torch.device] = None
    ) -> Tuple[float, float]:
        """
        评估量化模型性能
        
        Args:
            test_loader: 测试数据加载器
            criterion: 损失函数
            device: 设备
        
        Returns:
            元组 (平均损失，准确率)
        
        Raises:
            RuntimeError: 如果尚未准备模型
        """
        if self.calibrated_model is None:
            raise RuntimeError("请先准备模型")
        
        self.calibrated_model.eval()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, (list, tuple)):
                    inputs, targets = batch_data[0], batch_data[1]
                else:
                    inputs = batch_data
                    targets = None
                
                inputs = inputs.to(device)
                
                outputs = self.calibrated_model(inputs)
                
                if targets is not None:
                    targets = targets.to(device)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        if targets is not None:
            avg_loss = total_loss / total
            accuracy = 100. * correct / total
            logger.info(f"评估结果 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            return avg_loss, accuracy
        else:
            return 0.0, 0.0
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """
        获取量化信息摘要
        
        Returns:
            量化信息字典
        """
        if self.calibrated_model is None:
            return {}
        
        total_params = 0
        quantized_params = 0
        
        for name, module in self.calibrated_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                num_params = sum(p.numel() for p in module.parameters())
                total_params += num_params
                
                if hasattr(module, 'weight_fake_quant'):
                    quantized_params += num_params
        
        return {
            'total_params': total_params,
            'quantized_params': quantized_params,
            'quantization_ratio': quantized_params / total_params if total_params > 0 else 0,
            'bits': self.config.bits,
            'scheme': self.config.scheme,
            'per_channel': self.config.per_channel,
        }
    
    def print_quantization_info(self) -> None:
        """打印量化信息"""
        info = self.get_quantization_info()
        
        print("=" * 60)
        print("PTQ 量化信息")
        print("=" * 60)
        print(f"总参数：{info['total_params']:,}")
        print(f"已量化参数：{info['quantized_params']:,}")
        print(f"量化比例：{info['quantization_ratio']*100:.2f}%")
        print(f"量化位数：{info['bits']} bit")
        print(f"量化方案：{info['scheme']}")
        print(f"逐通道量化：{info['per_channel']}")
        print("=" * 60)


def ptq_quantize(
    model: nn.Module,
    config: QuantizationConfig,
    calib_loader,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> nn.Module:
    """
    PTQ 量化的便捷函数
    
    一站式完成 PTQ 流程：准备 -> 校准 -> 转换
    
    Args:
        model: 预训练模型
        config: 量化配置
        calib_loader: 校准数据加载器
        progress_callback: 进度回调函数
    
    Returns:
        量化后的模型
    
    Example:
        >>> config = QuantizationConfig(mode="ptq", bits=8)
        >>> quantized_model = ptq_quantize(model, config, calib_loader)
    """
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader, progress_callback)
    return ptq.convert()


def load_quantized_model(
    checkpoint_path: str,
    model: Optional[nn.Module] = None
) -> Tuple[nn.Module, QuantizationConfig, Dict[str, Any]]:
    """
    加载量化模型
    
    Args:
        checkpoint_path: 检查点路径
        model: 可选的模型实例（用于加载权重）
    
    Returns:
        元组 (模型，量化配置，量化参数)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = QuantizationConfig.from_dict(checkpoint['config'])
    quant_params = checkpoint['quantization_params']
    
    if model is None:
        # 需要用户自己提供模型结构
        raise ValueError("请提供模型实例以加载量化权重")
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    logger.info(f"已加载量化模型：{checkpoint_path}")
    logger.info(f"量化类型：{checkpoint.get('quantization_type', 'unknown')}")
    logger.info(f"量化位数：{config.bits} bit")
    
    return model, config, quant_params
