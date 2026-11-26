# -*- coding: utf-8 -*-
# @Time : 2026/7/6 上午9:52
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Configuration-driven optimizer construction."""

from __future__ import annotations

from typing import Any, Dict

from .spec import OptimizerSpec


def build_optimizer(model: Any, config_data: Dict[str, Any]) -> Any:
    """Build over final trainable parameters, preserving their identity and order.

    Mixed Tensor/DTensor AdamW parameters use scalar updates. Four-bit state
    does not support DTensor parameters and is rejected before construction.
    """
    import torch.distributed as dist
    import torch.optim as optim

    from .four_bit import FourBitAdamW, FourBitSGD

    spec = OptimizerSpec.from_config(config_data)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("optimizer construction requires trainable parameters")
    distributed_count = 0
    if dist.is_available():
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            from torch.distributed._tensor import DTensor

        distributed_count = sum(isinstance(parameter, DTensor) for parameter in params)
    if distributed_count and spec.is_four_bit:
        raise ValueError("Four-bit optimizer state does not support DTensor; use adamw")
    mixed_parameters = 0 < distributed_count < len(params)
    if spec.type == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=spec.lr,
            betas=spec.betas,
            eps=spec.eps,
            weight_decay=spec.weight_decay,
            foreach=False if mixed_parameters else None,
            fused=False if mixed_parameters else None,
        )
    elif spec.type == "torchao_adamw4bit":
        from .torchao_adapter import load_optimizer_class

        optimizer = load_optimizer_class()(
            params, lr=spec.lr, betas=spec.betas, eps=spec.eps,
            weight_decay=spec.weight_decay, block_size=128,
            amsgrad=False, bf16_stochastic_round=False,
        )
    elif spec.type == "four_bit_adamw":
        optimizer = FourBitAdamW(
            params,
            lr=spec.lr,
            betas=spec.betas,
            eps=spec.eps,
            weight_decay=spec.weight_decay,
            group_size=spec.group_size,
            compensate_quant_error=spec.compensate_quant_error,
            error_compensation_dtype=spec.error_compensation_dtype,
            error_compensation_mode=spec.error_compensation_mode,
        )
    else:
        optimizer = FourBitSGD(
            params,
            lr=spec.lr,
            momentum=spec.momentum,
            dampening=spec.dampening,
            weight_decay=spec.weight_decay,
            nesterov=spec.nesterov,
            group_size=spec.group_size,
            compensate_quant_error=spec.compensate_quant_error,
            error_compensation_dtype=spec.error_compensation_dtype,
            error_compensation_mode=spec.error_compensation_mode,
        )
    metadata = spec.to_metadata()
    metadata["trainable_params"] = sum(parameter.numel() for parameter in params)
    setattr(optimizer, "_parascale_optimizer_metadata", metadata)
    return optimizer


__all__ = ["build_optimizer"]
