# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:51
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Mixed precision controller for the training runtime."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


class PrecisionController:
    """Own AMP scaler, autocast, backward, and optimizer-step branching."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def setup_scaler(self) -> None:
        precision = str(getattr(self.engine.config, "precision", "fp32"))
        if precision != "fp16" or self.backend_name() not in {"native", "native_ddp"}:
            self.engine.amp_scaler = None
            return
        try:
            import torch

            if not torch.cuda.is_available():
                self.engine.amp_scaler = None
                return
            self.engine.amp_scaler = torch.cuda.amp.GradScaler()
        except Exception:
            self.engine.amp_scaler = None

    def autocast_context(self):
        precision = str(getattr(self.engine.config, "precision", "fp32"))
        if precision not in {"bf16", "fp16"} or self.backend_name() not in {
            "native",
            "native_ddp",
        }:
            return nullcontext()
        try:
            import torch

            if not torch.cuda.is_available():
                return nullcontext()
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        except Exception:
            return nullcontext()

    def backward(self, loss: Any) -> bool:
        scaler = getattr(self.engine, "amp_scaler", None)
        if scaler is not None and self.backend_name() in {"native", "native_ddp"}:
            scaler.scale(loss).backward()
            return True
        return False

    def step(self, optimizer: Any = None) -> bool:
        scaler = getattr(self.engine, "amp_scaler", None)
        if scaler is not None and self.backend_name() in {"native", "native_ddp"}:
            if optimizer is not None:
                scaler.step(optimizer)
                scaler.update()
                if hasattr(optimizer, "zero_grad"):
                    optimizer.zero_grad()
            return True
        return False

    def backend_name(self) -> str:
        backend_name = getattr(self.engine, "_backend_name", None)
        if callable(backend_name):
            return str(backend_name())
        return str(getattr(self.engine.config, "training_backend", "native"))


__all__ = ["PrecisionController"]
