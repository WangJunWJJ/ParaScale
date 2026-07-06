# -*- coding: utf-8 -*-
# @Time : 2026/7/6 上午9:52
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free optimizer configuration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

_COMMON_KEYS = {
    "type",
    "lr",
    "weight_decay",
    "group_size",
    "compensate_quant_error",
    "error_compensation_dtype",
}
_ALLOWED_KEYS = {
    "adamw": {"type", "lr", "betas", "eps", "weight_decay"},
    "four_bit_adamw": _COMMON_KEYS | {"betas", "eps"},
    "four_bit_sgd": _COMMON_KEYS | {"momentum", "dampening", "nesterov"},
}


@dataclass(frozen=True)
class OptimizerSpec:
    """Validated optimizer settings shared by workloads and runtime backends."""

    type: str = "adamw"
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False
    group_size: int = 128
    compensate_quant_error: bool = True
    error_compensation_dtype: str | None = None

    @classmethod
    def from_config(cls, config_data: Dict[str, Any]) -> "OptimizerSpec":
        optimizer = config_data.get("optimizer", {})
        optimizer = optimizer if isinstance(optimizer, dict) else {}
        optimizer_type = str(optimizer.get("type", "adamw") or "adamw").lower()
        if optimizer_type not in _ALLOWED_KEYS:
            supported = ", ".join(sorted(_ALLOWED_KEYS))
            raise ValueError(
                f"unsupported optimizer.type={optimizer_type!r}; supported: {supported}"
            )
        unexpected = sorted(set(optimizer) - _ALLOWED_KEYS[optimizer_type])
        if unexpected:
            raise ValueError(
                f"optimizer field {unexpected[0]!r} is not valid for {optimizer_type}"
            )
        betas = tuple(optimizer.get("betas", (0.9, 0.999)))
        if len(betas) != 2 or any(not 0.0 <= float(beta) < 1.0 for beta in betas):
            raise ValueError("optimizer.betas must contain two values in [0, 1)")
        group_size = int(optimizer.get("group_size", 128) or 128)
        if group_size <= 0 or group_size % 2:
            raise ValueError("optimizer.group_size must be a positive even integer")
        compensation_dtype = optimizer.get("error_compensation_dtype")
        if compensation_dtype not in {None, "fp16", "fp32"}:
            raise ValueError(
                "optimizer.error_compensation_dtype must be fp16, fp32, or null"
            )
        return cls(
            type=optimizer_type,
            lr=float(optimizer.get("lr", 0.001)),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(optimizer.get("eps", 1e-8)),
            weight_decay=float(optimizer.get("weight_decay", 0.01)),
            momentum=float(optimizer.get("momentum", 0.9)),
            dampening=float(optimizer.get("dampening", 0.0)),
            nesterov=bool(optimizer.get("nesterov", False)),
            group_size=group_size,
            compensate_quant_error=bool(
                optimizer.get("compensate_quant_error", True)
            ),
            error_compensation_dtype=compensation_dtype,
        )

    @property
    def is_four_bit(self) -> bool:
        return self.type in {"four_bit_adamw", "four_bit_sgd"}

    def validate_backend(self, training_backend: str, zero_stage: int = 0) -> None:
        if not self.is_four_bit:
            return
        backend = str(training_backend)
        if backend not in {"native", "native_ddp"}:
            raise ValueError(
                f"optimizer.type={self.type} is not supported with {backend}"
            )
        if int(zero_stage or 0) > 0:
            raise ValueError(
                f"optimizer.type={self.type} is not supported with zero_stage={zero_stage}"
            )

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "state_schema_version": 1,
            "group_size": self.group_size if self.is_four_bit else None,
            "compensate_quant_error": (
                self.compensate_quant_error if self.is_four_bit else None
            ),
            "error_compensation_dtype": (
                self.error_compensation_dtype if self.is_four_bit else None
            ),
        }


__all__ = ["OptimizerSpec"]
