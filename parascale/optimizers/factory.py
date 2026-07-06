# -*- coding: utf-8 -*-
# @Time : 2026/7/6 上午9:52
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Configuration-driven optimizer construction."""

from __future__ import annotations

from typing import Any, Dict

from .spec import OptimizerSpec


def build_optimizer(model: Any, config_data: Dict[str, Any]) -> Any:
    """Build the configured optimizer over trainable model parameters."""
    import torch.optim as optim

    from .four_bit import FourBitAdamW, FourBitSGD

    spec = OptimizerSpec.from_config(config_data)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("optimizer construction requires trainable parameters")
    if spec.type == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=spec.lr,
            betas=spec.betas,
            eps=spec.eps,
            weight_decay=spec.weight_decay,
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
        )
    metadata = spec.to_metadata()
    metadata["trainable_params"] = sum(parameter.numel() for parameter in params)
    setattr(optimizer, "_parascale_optimizer_metadata", metadata)
    return optimizer


__all__ = ["build_optimizer"]
