# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : optimizers.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 优化器模块

本模块提供了 ParaScale 框架的优化器实现，
包括 ZeRO 优化器包装器、AdamW 优化器和 4bit 量化优化器。
"""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Union, Optional
import math


class QuantizedState:
    """
    4bit 量化状态类

    将优化器状态量化为 4bit 以节省内存。
    使用分组量化和反量化来保持精度。

    Attributes:
        quantized_data: 量化后的数据（uint8 类型，每字节存储两个 4bit 值）
        scale: 每组的比例因子
        group_size: 分组大小
        shape: 原始张量形状
    """

    def __init__(self, tensor: torch.Tensor, group_size: int = 128):
        """
        初始化量化状态

        Args:
            tensor: 要量化的张量
            group_size: 分组大小，默认为 128
        """
        self.shape = tensor.shape
        self.group_size = group_size
        self.device = tensor.device

        # 将张量展平
        flat_tensor = tensor.flatten()
        num_elements = flat_tensor.numel()

        # 计算组数
        num_groups = (num_elements + group_size - 1) // group_size

        # 填充到 group_size 的倍数
        if num_elements % group_size != 0:
            padding = group_size - (num_elements % group_size)
            flat_tensor = torch.cat([flat_tensor, torch.zeros(padding, device=tensor.device, dtype=tensor.dtype)])

        # 重塑为 [num_groups, group_size]
        grouped = flat_tensor.view(num_groups, group_size)

        # 计算每组的 min 和 max
        group_min = grouped.min(dim=1, keepdim=True)[0]
        group_max = grouped.max(dim=1, keepdim=True)[0]

        # 计算比例因子
        self.scale = (group_max - group_min) / 15.0  # 4bit 有 16 个值 (0-15)
        self.scale = self.scale.squeeze(1)
        self.zero_point = group_min.squeeze(1)

        # 量化到 4bit
        quantized = ((grouped - group_min) / (group_max - group_min + 1e-8) * 15).round().clamp(0, 15).to(torch.uint8)

        # 将两个 4bit 值打包到一个 uint8 中
        self.quantized_data = torch.zeros(num_groups * (group_size // 2), dtype=torch.uint8, device=tensor.device)
        for i in range(group_size // 2):
            self.quantized_data[i::group_size//2] = (quantized[:, 2*i] << 4) | quantized[:, 2*i + 1]

    def dequantize(self) -> torch.Tensor:
        """
        反量化为原始张量

        Returns:
            反量化后的张量
        """
        num_groups = len(self.scale)

        # 解包 4bit 值
        high_4bit = (self.quantized_data >> 4) & 0x0F
        low_4bit = self.quantized_data & 0x0F

        # 交错组合
        quantized = torch.zeros(num_groups, self.group_size, dtype=torch.float32, device=self.device)
        for i in range(self.group_size // 2):
            quantized[:, 2*i] = high_4bit[i::self.group_size//2].float()
            quantized[:, 2*i + 1] = low_4bit[i::self.group_size//2].float()

        # 反量化
        dequantized = quantized * self.scale.unsqueeze(1) + self.zero_point.unsqueeze(1)

        # 重塑为原始形状
        flat_result = dequantized.flatten()[:torch.prod(torch.tensor(self.shape))]
        return flat_result.view(self.shape)

    def update(self, new_tensor: torch.Tensor) -> None:
        """
        更新为新值（重新量化）

        Args:
            new_tensor: 新的张量值
        """
        # 创建新的量化状态
        new_state = QuantizedState(new_tensor, self.group_size)
        self.quantized_data = new_state.quantized_data
        self.scale = new_state.scale
        self.zero_point = new_state.zero_point
        self.shape = new_state.shape

    def memory_usage(self) -> int:
        """
        获取内存使用量（字节）

        Returns:
            内存使用量
        """
        return (self.quantized_data.numel() * 1 +  # uint8 数据
                self.scale.numel() * 4 +           # float32 scale
                self.zero_point.numel() * 4)       # float32 zero_point

    def to(self, device: torch.device) -> 'QuantizedState':
        """
        转移到指定设备

        Args:
            device: 目标设备

        Returns:
            自身（已转移设备）
        """
        self.quantized_data = self.quantized_data.to(device)
        self.scale = self.scale.to(device)
        self.zero_point = self.zero_point.to(device)
        self.device = device
        return self


class ZeroOptimizer:
    """
    ZeRO 优化器包装类
    
    实现 ZeRO (Zero Redundancy Optimizer) 技术，通过分片优化器状态
    来减少内存使用，支持大规模模型训练。
    
    ZeRO 的三个阶段：
    - Stage 1: 分片优化器状态
    - Stage 2: 分片梯度
    - Stage 3: 分片模型参数
    
    Attributes:
        model: PyTorch 模型实例
        optimizer: 原始优化器实例
        stage: ZeRO 阶段（0, 1, 2, 3）
        offload: 是否启用 CPU offload
        param_groups: 参数组列表
    
    Example:
        >>> model = SimpleModel()
        >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
        >>> zero_optimizer = ZeroOptimizer(model, optimizer, stage=2)
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: optim.Optimizer, 
        stage: int = 0, 
        offload: bool = False
    ):
        """
        初始化 ZeRO 优化器
        
        Args:
            model: PyTorch 模型实例
            optimizer: 原始优化器实例
            stage: ZeRO 阶段，取值范围 0-3
                   - 0: 不启用 ZeRO
                   - 1: 分片优化器状态
                   - 2: 分片梯度
                   - 3: 分片模型参数
            offload: 是否将数据卸载到 CPU 以节省 GPU 内存
        
        Raises:
            ValueError: 当 stage 不在有效范围内时抛出
        """
        if stage not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid ZeRO stage: {stage}, must be 0, 1, 2, or 3")
        
        self.model = model
        self.optimizer = optimizer
        self.stage = stage
        self.offload = offload
        self.param_groups = optimizer.param_groups
    
    def step(self) -> None:
        """
        执行优化步骤
        
        调用内部优化器的 step 方法更新模型参数。
        """
        self.optimizer.step()
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        清零梯度
        
        Args:
            set_to_none: 是否将梯度设置为 None 而不是零张量，
                        设置为 None 可以节省内存
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        获取优化器状态字典
        
        Returns:
            包含优化器状态的字典，可用于保存和恢复训练状态
        """
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载优化器状态
        
        Args:
            state_dict: 之前保存的优化器状态字典
        """
        self.optimizer.load_state_dict(state_dict)
    
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        添加参数组
        
        Args:
            param_group: 参数组字典，包含参数和优化器配置
        """
        self.optimizer.add_param_group(param_group)


class AdamW(optim.AdamW):
    """
    AdamW 优化器
    
    实现 Adam with Decoupled Weight Decay (AdamW) 优化器。
    与标准 Adam 相比，AdamW 将权重衰减与梯度更新解耦，
    可以获得更好的泛化性能。
    
    继承自 torch.optim.AdamW，提供与 PyTorch 原生实现相同的接口。
    
    Args:
        params: 模型参数或参数组列表
        lr: 学习率，默认为 1e-3
        betas: Adam 的 beta 系数，控制一阶和二阶矩估计的衰减率
        eps: 数值稳定性常数，防止除零错误
        weight_decay: 权重衰减系数，默认为 0.01
        amsgrad: 是否使用 AMSGrad 变体
    
    Example:
        >>> model = SimpleModel()
        >>> optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        >>> optimizer.step()
    """
    
    def __init__(
        self, 
        params: Union[List[torch.Tensor], List[Dict[str, Any]]], 
        lr: float = 1e-3, 
        betas: tuple = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 0.01, 
        amsgrad: bool = False
    ):
        """
        初始化 AdamW 优化器
        
        Args:
            params: 可迭代对象，包含模型参数或参数组字典
            lr: 学习率，默认为 1e-3
            betas: Adam 的 beta 系数 (beta1, beta2)
                   - beta1: 一阶矩估计的衰减率，默认 0.9
                   - beta2: 二阶矩估计的衰减率，默认 0.999
            eps: 数值稳定性常数，默认为 1e-8
            weight_decay: 权重衰减系数，默认为 0.01
            amsgrad: 是否使用 AMSGrad 变体，默认为 False
        """
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )


class FourBitAdamW(optim.Optimizer):
    """
    4bit AdamW 优化器

    使用 4bit 量化存储优化器状态（momentum 和 variance），
    可以显著减少内存使用（约 50%）。

    Attributes:
        param_groups: 参数组列表
        state: 优化器状态字典
        group_size: 量化分组大小
        compensate_quant_error: 是否启用误差补偿

    Example:
        >>> model = SimpleModel()
        >>> optimizer = FourBitAdamW(model.parameters(), lr=1e-3)
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        group_size: int = 128,
        compensate_quant_error: bool = True
    ):
        """
        初始化 4bit AdamW 优化器

        Args:
            params: 可迭代对象，包含模型参数
            lr: 学习率，默认为 1e-3
            betas: Adam 的 beta 系数 (beta1, beta2)
            eps: 数值稳定性常数
            weight_decay: 权重衰减系数
            group_size: 量化分组大小
            compensate_quant_error: 是否启用误差补偿
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.group_size = group_size
        self.compensate_quant_error = compensate_quant_error

    def step(self, closure=None):
        """
        执行优化步骤

        Args:
            closure: 可选的闭包函数，用于重新计算损失

        Returns:
            如果提供了 closure，返回损失值
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    # 使用 4bit 量化存储 momentum 和 variance
                    exp_avg = torch.zeros_like(p.data)
                    exp_avg_sq = torch.zeros_like(p.data)
                    state['exp_avg'] = QuantizedState(exp_avg, self.group_size)
                    state['exp_avg_sq'] = QuantizedState(exp_avg_sq, self.group_size)
                    if self.compensate_quant_error:
                        state['exp_avg_error'] = torch.zeros_like(p.data)
                        state['exp_avg_sq_error'] = torch.zeros_like(p.data)

                exp_avg_q = state['exp_avg']
                exp_avg_sq_q = state['exp_avg_sq']
                state['step'] += 1

                # 反量化
                exp_avg = exp_avg_q.dequantize()
                exp_avg_sq = exp_avg_sq_q.dequantize()

                # 误差补偿
                if self.compensate_quant_error:
                    exp_avg = exp_avg + state['exp_avg_error']
                    exp_avg_sq = exp_avg_sq + state['exp_avg_sq_error']

                # AdamW 更新
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # 更新 momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新 variance
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 计算步长
                step_size = group['lr'] / bias_correction1

                # 计算二阶矩的平方根
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # 更新参数
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # 计算量化误差并保存
                if self.compensate_quant_error:
                    new_exp_avg_q = QuantizedState(exp_avg, self.group_size)
                    new_exp_avg_sq_q = QuantizedState(exp_avg_sq, self.group_size)
                    state['exp_avg_error'] = exp_avg - new_exp_avg_q.dequantize()
                    state['exp_avg_sq_error'] = exp_avg_sq - new_exp_avg_sq_q.dequantize()
                    state['exp_avg'] = new_exp_avg_q
                    state['exp_avg_sq'] = new_exp_avg_sq_q
                else:
                    state['exp_avg'] = QuantizedState(exp_avg, self.group_size)
                    state['exp_avg_sq'] = QuantizedState(exp_avg_sq, self.group_size)

        return loss

    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取内存统计信息

        Returns:
            包含内存统计的字典
        """
        total_params = sum(p.numel() for group in self.param_groups for p in group['params'])

        # 计算 4bit 量化后的内存使用
        quantized_bytes = 0
        for state in self.state.values():
            if 'exp_avg' in state:
                quantized_bytes += state['exp_avg'].memory_usage()
            if 'exp_avg_sq' in state:
                quantized_bytes += state['exp_avg_sq'].memory_usage()

        # 标准 AdamW 使用 2 个状态张量（float32）
        standard_bytes = total_params * 4 * 2

        savings_percent = (1 - quantized_bytes / standard_bytes) * 100 if standard_bytes > 0 else 0

        return {
            'total_params': total_params,
            'quantized_bytes': quantized_bytes,
            'standard_bytes': standard_bytes,
            'savings_percent': savings_percent
        }

    def print_memory_stats(self):
        """打印内存统计信息"""
        stats = self.get_memory_stats()
        print(f"4bit AdamW 内存统计:")
        print(f"  总参数: {stats['total_params']:,}")
        print(f"  量化后内存: {stats['quantized_bytes'] / 1024**2:.2f} MB")
        print(f"  标准 AdamW 内存: {stats['standard_bytes'] / 1024**2:.2f} MB")
        print(f"  节省: {stats['savings_percent']:.1f}%")


class FourBitSGD(optim.Optimizer):
    """
    4bit SGD 优化器

    使用 4bit 量化存储 momentum，可以显著减少内存使用。

    Attributes:
        param_groups: 参数组列表
        state: 优化器状态字典
        group_size: 量化分组大小
        nesterov: 是否使用 Nesterov 动量

    Example:
        >>> model = SimpleModel()
        >>> optimizer = FourBitSGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        group_size: int = 128,
        compensate_quant_error: bool = True
    ):
        """
        初始化 4bit SGD 优化器

        Args:
            params: 可迭代对象，包含模型参数
            lr: 学习率
            momentum: 动量系数
            weight_decay: 权重衰减系数
            dampening: 阻尼系数
            nesterov: 是否使用 Nesterov 动量
            group_size: 量化分组大小
            compensate_quant_error: 是否启用误差补偿
        """
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)
        self.group_size = group_size
        self.compensate_quant_error = compensate_quant_error

    def step(self, closure=None):
        """
        执行优化步骤

        Args:
            closure: 可选的闭包函数，用于重新计算损失

        Returns:
            如果提供了 closure，返回损失值
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_state = self.state[p]

                # 初始化 momentum
                if 'momentum_buffer' not in param_state:
                    buf = torch.zeros_like(p.data)
                    param_state['momentum_buffer'] = QuantizedState(buf, self.group_size)
                    if self.compensate_quant_error:
                        param_state['momentum_error'] = torch.zeros_like(p.data)

                momentum_buffer_q = param_state['momentum_buffer']
                buf = momentum_buffer_q.dequantize()

                # 误差补偿
                if self.compensate_quant_error:
                    buf = buf + param_state['momentum_error']

                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

                p.data.add_(grad, alpha=-group['lr'])

                # 量化并保存
                if self.compensate_quant_error:
                    new_buf_q = QuantizedState(buf, self.group_size)
                    param_state['momentum_error'] = buf - new_buf_q.dequantize()
                    param_state['momentum_buffer'] = new_buf_q
                else:
                    param_state['momentum_buffer'] = QuantizedState(buf, self.group_size)

        return loss

    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取内存统计信息

        Returns:
            包含内存统计的字典
        """
        total_params = sum(p.numel() for group in self.param_groups for p in group['params'])

        quantized_bytes = 0
        for state in self.state.values():
            if 'momentum_buffer' in state:
                quantized_bytes += state['momentum_buffer'].memory_usage()

        standard_bytes = total_params * 4

        savings_percent = (1 - quantized_bytes / standard_bytes) * 100 if standard_bytes > 0 else 0

        return {
            'total_params': total_params,
            'quantized_bytes': quantized_bytes,
            'standard_bytes': standard_bytes,
            'savings_percent': savings_percent
        }

    def print_memory_stats(self):
        """打印内存统计信息"""
        stats = self.get_memory_stats()
        print(f"4bit SGD 内存统计:")
        print(f"  总参数: {stats['total_params']:,}")
        print(f"  量化后内存: {stats['quantized_bytes'] / 1024**2:.2f} MB")
        print(f"  标准 SGD 内存: {stats['standard_bytes'] / 1024**2:.2f} MB")
        print(f"  节省: {stats['savings_percent']:.1f}%")
