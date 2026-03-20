# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : utils.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 量化工具函数模块

本模块提供量化相关的工具函数，包括张量量化/反量化、
模块融合等辅助功能。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import QuantizationConfig


def calculate_scale_zero_point(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    config: QuantizationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算量化参数（scale 和 zero_point）
    
    Args:
        min_val: 最小值
        max_val: 最大值
        config: 量化配置
    
    Returns:
        元组 (scale, zero_point)
    """
    qmin, qmax = config.get_qmin_qmax()
    
    if config.scheme == "symmetric":
        # 对称量化
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = max_abs / qmax
        zero_point = torch.zeros_like(scale)
    else:
        # 非对称量化
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
    
    # 防止 scale 为 0
    scale = torch.clamp(scale, min=1e-8)
    
    return scale, zero_point


def quantize_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    config: QuantizationConfig
) -> torch.Tensor:
    """
    量化张量
    
    Args:
        x: 输入张量
        scale: 缩放因子
        zero_point: 零点
        config: 量化配置
    
    Returns:
        量化后的整数张量
    """
    qmin, qmax = config.get_qmin_qmax()
    
    # 量化
    x_quant = torch.round(x / scale + zero_point)
    
    # 钳制
    x_quant = torch.clamp(x_quant, qmin, qmax)
    
    return x_quant


def dequantize_tensor(
    x_quant: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    反量化张量
    
    Args:
        x_quant: 量化后的整数张量
        scale: 缩放因子
        zero_point: 零点
    
    Returns:
        反量化后的浮点张量
    """
    return (x_quant - zero_point) * scale


def fuse_modules(
    model: nn.Module,
    modules_to_fuse: Optional[list] = None
) -> nn.Module:
    """
    融合模块（Conv + BN + ReLU）
    
    模块融合可以减少计算量，提高量化精度。
    
    Args:
        model: 输入模型
        modules_to_fuse: 要融合的模块列表，如果为 None 则自动检测
    
    Returns:
        融合后的模型
    """
    if modules_to_fuse is None:
        # 自动检测可融合的模块
        modules_to_fuse = []
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                # 检查是否包含 Conv + BN + ReLU
                layers = list(module.children())
                if len(layers) >= 3:
                    if (isinstance(layers[0], (nn.Conv2d, nn.Conv1d)) and
                        isinstance(layers[1], nn.BatchNorm2d) and
                        isinstance(layers[2], nn.ReLU)):
                        modules_to_fuse.append([f"{name}.0", f"{name}.1", f"{name}.2"])
    
    # 使用 PyTorch 的融合功能
    if modules_to_fuse:
        try:
            torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
        except Exception as e:
            print(f"模块融合失败: {e}")
    
    return model


def get_quantizable_layers(model: nn.Module) -> list:
    """
    获取模型中所有可量化的层
    
    Args:
        model: 输入模型
    
    Returns:
        可量化层的名称列表
    """
    quantizable_types = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)
    quantizable_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, quantizable_types):
            quantizable_layers.append(name)
    
    return quantizable_layers


def copy_model_weights(src_model: nn.Module, dst_model: nn.Module) -> None:
    """
    复制模型权重
    
    Args:
        src_model: 源模型
        dst_model: 目标模型
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    # 只复制匹配的权重
    for name, param in src_state.items():
        if name in dst_state:
            if dst_state[name].shape == param.shape:
                dst_state[name].copy_(param)


def print_quantization_info(model: nn.Module) -> None:
    """
    打印模型的量化信息
    
    Args:
        model: 输入模型
    """
    print("=" * 60)
    print("模型量化信息")
    print("=" * 60)
    
    total_params = 0
    quantized_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            
            # 检查是否有伪量化器
            if hasattr(module, 'weight_fake_quant'):
                quantized_params += num_params
                print(f"{name}: 已量化 ({num_params:,} 参数)")
            else:
                print(f"{name}: 未量化 ({num_params:,} 参数)")
    
    print("-" * 60)
    print(f"总参数: {total_params:,}")
    print(f"已量化参数: {quantized_params:,}")
    print(f"量化比例: {quantized_params/total_params*100:.2f}%")
    print("=" * 60)
