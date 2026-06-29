# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Experimental optimizer helpers for ParaScale.

Production sharding should use the runtime FSDP or DeepSpeed backends.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.optim as optim


class QuantizedState:

    def __init__(self, tensor: torch.Tensor, group_size: int = 128):
        self.shape = tensor.shape
        self.group_size = group_size
        self.device = tensor.device
        flat_tensor = tensor.flatten()
        num_elements = flat_tensor.numel()
        num_groups = (num_elements + group_size - 1) // group_size
        if num_elements % group_size != 0:
            padding = group_size - num_elements % group_size
            flat_tensor = torch.cat(
                [
                    flat_tensor,
                    torch.zeros(padding, device=tensor.device, dtype=tensor.dtype),
                ]
            )
        grouped = flat_tensor.view(num_groups, group_size)
        group_min = grouped.min(dim=1, keepdim=True)[0]
        group_max = grouped.max(dim=1, keepdim=True)[0]
        self.scale = (group_max - group_min) / 15.0
        self.scale = self.scale.squeeze(1)
        self.zero_point = group_min.squeeze(1)
        quantized = (
            ((grouped - group_min) / (group_max - group_min + 1e-08) * 15)
            .round()
            .clamp(0, 15)
            .to(torch.uint8)
        )
        self.quantized_data = torch.zeros(
            num_groups * (group_size // 2), dtype=torch.uint8, device=tensor.device
        )
        for i in range(group_size // 2):
            self.quantized_data[i :: group_size // 2] = (
                quantized[:, 2 * i] << 4 | quantized[:, 2 * i + 1]
            )

    def dequantize(self) -> torch.Tensor:
        num_groups = len(self.scale)
        high_4bit = self.quantized_data >> 4 & 15
        low_4bit = self.quantized_data & 15
        quantized = torch.zeros(
            num_groups, self.group_size, dtype=torch.float32, device=self.device
        )
        for i in range(self.group_size // 2):
            quantized[:, 2 * i] = high_4bit[i :: self.group_size // 2].float()
            quantized[:, 2 * i + 1] = low_4bit[i :: self.group_size // 2].float()
        dequantized = quantized * self.scale.unsqueeze(1) + self.zero_point.unsqueeze(1)
        flat_result = dequantized.flatten()[: torch.prod(torch.tensor(self.shape))]
        return flat_result.view(self.shape)

    def update(self, new_tensor: torch.Tensor) -> None:
        new_state = QuantizedState(new_tensor, self.group_size)
        self.quantized_data = new_state.quantized_data
        self.scale = new_state.scale
        self.zero_point = new_state.zero_point
        self.shape = new_state.shape

    def memory_usage(self) -> int:
        return (
            self.quantized_data.numel() * 1
            + self.scale.numel() * 4
            + self.zero_point.numel() * 4
        )

    def sparseify(self, threshold: float = 1e-06) -> "QuantizedState":
        dense = self.dequantize()
        sparse_tensor = dense.masked_fill(torch.abs(dense) <= threshold, 0)
        sparse_state = QuantizedState(sparse_tensor, self.group_size)
        sparse_state.is_sparse = True
        return sparse_state

    def to(self, device: torch.device) -> "QuantizedState":
        self.quantized_data = self.quantized_data.to(device)
        self.scale = self.scale.to(device)
        self.zero_point = self.zero_point.to(device)
        self.device = device
        return self


class ZeroOptimizer:

    def __init__(
        self,
        base_optimizer: optim.Optimizer,
        stage: int = 1,
        offload: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if stage not in [1, 2, 3]:
            raise ValueError(f"ZeRO stage must be 1, 2, or 3, got {stage}")
        self.base_optimizer = base_optimizer
        self.stage = stage
        self.offload = offload
        self.world_size = (
            world_size
            if world_size is not None
            else (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
        )
        self.rank = (
            rank
            if rank is not None
            else dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        if self.world_size > 1 and (
            not (dist.is_available() and dist.is_initialized())
        ):
            raise RuntimeError(
                "ZeroOptimizer requires torch.distributed to be initialized when world_size > 1"
            )
        if offload:
            warnings.warn(
                "ZeroOptimizer CPU offload is not implemented yet; falling back to the base optimizer.",
                RuntimeWarning,
            )
        if stage >= 2:
            warnings.warn(
                "ZeroOptimizer currently provides a compatibility wrapper and metadata only for stage >= 2. It does not implement DeepSpeed-equivalent gradient or parameter partitioning.",
                RuntimeWarning,
            )
        self.param_groups = base_optimizer.param_groups
        self._param_group_mapping: Dict[int, List[int]] = {}
        self._setup_partitioning()

    def _setup_partitioning(self) -> None:
        for group_id, param_group in enumerate(self.param_groups):
            params = param_group["params"]
            num_params = len(params)
            if self.world_size > 1:
                partition_size = (num_params + self.world_size - 1) // self.world_size
                start_idx = self.rank * partition_size
                end_idx = min(start_idx + partition_size, num_params)
                if start_idx < num_params:
                    self._param_group_mapping[group_id] = list(
                        range(start_idx, end_idx)
                    )
                else:
                    self._param_group_mapping[group_id] = []
            else:
                self._param_group_mapping[group_id] = list(range(num_params))

    def _get_partitioned_params(self, group_id: int) -> List[torch.nn.Parameter]:
        if group_id not in self._param_group_mapping:
            return []
        param_indices = self._param_group_mapping[group_id]
        return [
            self.param_groups[group_id]["params"][i]
            for i in param_indices
            if i < len(self.param_groups[group_id]["params"])
        ]

    def step(self) -> None:
        if self.world_size > 1 and self.stage >= 2:
            self._raise_stage_not_implemented()
        self.base_optimizer.step()
        if self.stage >= 3:
            self._raise_stage_not_implemented()

    def _raise_stage_not_implemented(self) -> None:
        raise NotImplementedError(
            "ZeroOptimizer stage 2/3 sharding is not implemented in ParaScale yet. Use stage=1 for wrapper compatibility or integrate a real ZeRO backend."
        )

    def _reduce_gradients(self) -> None:
        self._raise_stage_not_implemented()

    def _broadcast_parameters(self) -> None:
        self._raise_stage_not_implemented()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict)

    def partition_parameters(self) -> Dict[int, List[torch.nn.Parameter]]:
        return {
            gid: self._get_partitioned_params(gid) for gid in self._param_group_mapping
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        total_params = sum(
            (p.numel() for group in self.param_groups for p in group["params"])
        )
        partitioned_params = sum(
            (
                p.numel()
                for gid in self._param_group_mapping
                for p in self._get_partitioned_params(gid)
            )
        )
        return {
            "total_parameters": total_params,
            "partitioned_parameters": partitioned_params,
            "partitioned_ratio": (
                partitioned_params / total_params if total_params > 0 else 1.0
            ),
            "stage": self.stage,
            "world_size": self.world_size,
            "implemented_stage": 1,
            "offload_enabled": self.offload,
            "offload_implemented": False,
            "memory_savings_factor": {
                1: "metadata only; no real optimizer-state sharding",
                2: "not implemented",
                3: "not implemented",
            }.get(self.stage, "1x"),
        }

    def print_memory_stats(self) -> None:
        stats = self.get_memory_stats()
        print("=" * 60)
        print("ZeRO optimizer memory stats")
        print("=" * 60)
        print(f"total parameters: {stats['total_parameters']:,}")
        print(f"partitioned parameters: {stats['partitioned_parameters']:,}")
        print(f"partitioned ratio: {stats['partitioned_ratio'] * 100:.2f}%")
        print(f"ZeRO Stage: {stats['stage']}")
        print(f"World Size: {stats['world_size']}")
        print(f"estimated memory saving: {stats['memory_savings_factor']}")
        print("=" * 60)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        self.base_optimizer.add_param_group(param_group)
        self._setup_partitioning()


class AdamW(optim.AdamW):

    def __init__(
        self,
        params: Union[List[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


class FourBitAdamW(optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        group_size: int = 128,
        compensate_quant_error: bool = True,
        error_compensation_dtype: Optional[str] = None,
    ):
        if error_compensation_dtype not in (None, "fp16", "fp32"):
            raise ValueError("error_compensation_dtype must be None, 'fp16' or 'fp32'")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.group_size = group_size
        self.compensate_quant_error = compensate_quant_error
        self.error_compensation_dtype = error_compensation_dtype

    def _error_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        if self.error_compensation_dtype == "fp16":
            return torch.float16
        if self.error_compensation_dtype == "fp32":
            return torch.float32
        return tensor.dtype

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    exp_avg = torch.zeros_like(p.data)
                    exp_avg_sq = torch.zeros_like(p.data)
                    state["exp_avg"] = QuantizedState(exp_avg, self.group_size)
                    state["exp_avg_sq"] = QuantizedState(exp_avg_sq, self.group_size)
                    if self.compensate_quant_error:
                        error_dtype = self._error_dtype(p.data)
                        state["exp_avg_error"] = torch.zeros_like(
                            p.data, dtype=error_dtype
                        )
                        state["exp_avg_sq_error"] = torch.zeros_like(
                            p.data, dtype=error_dtype
                        )
                exp_avg_q = state["exp_avg"]
                exp_avg_sq_q = state["exp_avg_sq"]
                state["step"] += 1
                exp_avg = exp_avg_q.dequantize()
                exp_avg_sq = exp_avg_sq_q.dequantize()
                if self.compensate_quant_error:
                    exp_avg = exp_avg + state["exp_avg_error"].to(exp_avg.dtype)
                    exp_avg_sq = exp_avg_sq + state["exp_avg_sq_error"].to(
                        exp_avg_sq.dtype
                    )
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                if self.compensate_quant_error:
                    new_exp_avg_q = QuantizedState(exp_avg, self.group_size)
                    new_exp_avg_sq_q = QuantizedState(exp_avg_sq, self.group_size)
                    error_dtype = self._error_dtype(p.data)
                    state["exp_avg_error"] = (exp_avg - new_exp_avg_q.dequantize()).to(
                        error_dtype
                    )
                    state["exp_avg_sq_error"] = (
                        exp_avg_sq - new_exp_avg_sq_q.dequantize()
                    ).to(error_dtype)
                    state["exp_avg"] = new_exp_avg_q
                    state["exp_avg_sq"] = new_exp_avg_sq_q
                else:
                    state["exp_avg"] = QuantizedState(exp_avg, self.group_size)
                    state["exp_avg_sq"] = QuantizedState(exp_avg_sq, self.group_size)
        return loss

    def get_memory_stats(self) -> Dict[str, float]:
        total_params = sum(
            (p.numel() for group in self.param_groups for p in group["params"])
        )
        quantized_bytes = 0
        for state in self.state.values():
            if "exp_avg" in state:
                quantized_bytes += state["exp_avg"].memory_usage()
            if "exp_avg_sq" in state:
                quantized_bytes += state["exp_avg_sq"].memory_usage()
        standard_bytes = total_params * 4 * 2
        savings_percent = (
            (1 - quantized_bytes / standard_bytes) * 100 if standard_bytes > 0 else 0
        )
        return {
            "total_params": total_params,
            "quantized_bytes": quantized_bytes,
            "standard_bytes": standard_bytes,
            "savings_percent": savings_percent,
        }

    def print_memory_stats(self):
        stats = self.get_memory_stats()
        print("4bit AdamW memory stats:")
        print(f"  total parameters: {stats['total_params']:,}")
        print(f"  quantized memory: {stats['quantized_bytes'] / 1024 ** 2:.2f} MB")
        print(f"  standard AdamW memory: {stats['standard_bytes'] / 1024 ** 2:.2f} MB")
        print(f"  saving: {stats['savings_percent']:.1f}%")


class FourBitSGD(optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        group_size: int = 128,
        compensate_quant_error: bool = True,
        error_compensation_dtype: Optional[str] = None,
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if error_compensation_dtype not in (None, "fp16", "fp32"):
            raise ValueError("error_compensation_dtype must be None, 'fp16' or 'fp32'")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)
        self.group_size = group_size
        self.compensate_quant_error = compensate_quant_error
        self.error_compensation_dtype = error_compensation_dtype

    def _error_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        if self.error_compensation_dtype == "fp16":
            return torch.float16
        if self.error_compensation_dtype == "fp32":
            return torch.float32
        return tensor.dtype

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = torch.zeros_like(p.data)
                    param_state["momentum_buffer"] = QuantizedState(
                        buf, self.group_size
                    )
                    if self.compensate_quant_error:
                        param_state["momentum_error"] = torch.zeros_like(
                            p.data, dtype=self._error_dtype(p.data)
                        )
                momentum_buffer_q = param_state["momentum_buffer"]
                buf = momentum_buffer_q.dequantize()
                if self.compensate_quant_error:
                    buf = buf + param_state["momentum_error"].to(buf.dtype)
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf
                p.data.add_(grad, alpha=-group["lr"])
                if self.compensate_quant_error:
                    new_buf_q = QuantizedState(buf, self.group_size)
                    param_state["momentum_error"] = (buf - new_buf_q.dequantize()).to(
                        self._error_dtype(p.data)
                    )
                    param_state["momentum_buffer"] = new_buf_q
                else:
                    param_state["momentum_buffer"] = QuantizedState(
                        buf, self.group_size
                    )
        return loss

    def get_memory_stats(self) -> Dict[str, float]:
        total_params = sum(
            (p.numel() for group in self.param_groups for p in group["params"])
        )
        quantized_bytes = 0
        for state in self.state.values():
            if "momentum_buffer" in state:
                quantized_bytes += state["momentum_buffer"].memory_usage()
        standard_bytes = total_params * 4
        savings_percent = (
            (1 - quantized_bytes / standard_bytes) * 100 if standard_bytes > 0 else 0
        )
        return {
            "total_params": total_params,
            "quantized_bytes": quantized_bytes,
            "standard_bytes": standard_bytes,
            "savings_percent": savings_percent,
        }

    def print_memory_stats(self):
        stats = self.get_memory_stats()
        print("4bit SGD memory stats:")
        print(f"  total parameters: {stats['total_params']:,}")
        print(f"  quantized memory: {stats['quantized_bytes'] / 1024 ** 2:.2f} MB")
        print(f"  standard SGD memory: {stats['standard_bytes'] / 1024 ** 2:.2f} MB")
        print(f"  saving: {stats['savings_percent']:.1f}%")
