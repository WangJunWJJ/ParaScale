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
        self.optimizer_updated = False
        self._pending_scaler_state = None

    def setup_scaler(self) -> None:
        precision = str(getattr(self.engine.config, "precision", "fp32"))
        if precision != "fp16" or self.backend_name() not in {"native", "native_ddp"}:
            if self._pending_scaler_state is not None:
                raise ValueError("checkpoint AMP scaler requires native FP16 training")
            self.engine.amp_scaler = None
            return
        if self.engine.amp_scaler is not None:
            self._restore_pending_scaler()
            return
        try:
            import torch

            if torch.cuda.is_available():
                self.engine.amp_scaler = torch.cuda.amp.GradScaler()
            else:
                self.engine.amp_scaler = None
        except ImportError:
            self.engine.amp_scaler = None
        self._restore_pending_scaler()

    def _restore_pending_scaler(self) -> None:
        if self._pending_scaler_state is not None:
            if self.engine.amp_scaler is None:
                raise RuntimeError("cannot restore AMP scaler without a supported accelerator")
            self.engine.amp_scaler.load_state_dict(self._pending_scaler_state)
            self._pending_scaler_state = None

    def state_dict(self) -> dict[str, Any]:
        """Return the scaler state needed for numerically equivalent resume."""
        scaler = getattr(self.engine, "amp_scaler", None)
        return {"amp_scaler": scaler.state_dict() if scaler is not None else self._pending_scaler_state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore lazily, since backend/scaler setup follows checkpoint loading."""
        self._pending_scaler_state = state.get("amp_scaler")

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
            self.optimizer_updated = False
            if optimizer is not None:
                clip_norm = getattr(self.engine.config, "grad_clip_norm", None)
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                    self.engine.training_backend.clip_gradients(float(clip_norm), optimizer)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                self.optimizer_updated = scaler.get_scale() >= scale_before
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
