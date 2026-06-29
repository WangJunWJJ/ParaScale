# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:58
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Base training backend contract."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple


class TrainingBackend:
    name = "base"

    def __init__(
        self,
        model: Any = None,
        optimizer: Any = None,
        config: Any = None,
        local_rank: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.local_rank = local_rank

    def setup(self) -> Tuple[Any, Any]:
        self.model = self.setup_model(self.model)
        self.optimizer = self.setup_optimizer(self.optimizer)
        return self.model, self.optimizer

    def setup_model(self, model: Any) -> Any:
        return model

    def setup_optimizer(self, optimizer: Any) -> Any:
        return optimizer

    def prepare_batch(self, batch: Any) -> Any:
        try:
            torch = _require_torch()
        except ImportError:
            return batch
        from .devices import (
            current_accelerator,
            move_batch_to_device,
            resolve_torch_device,
        )

        if current_accelerator(torch) in {"cuda", "npu"}:
            device = resolve_torch_device(torch, local_rank=self.local_rank)
            return move_batch_to_device(batch, str(device))
        return batch

    def backward(self, loss: Any) -> None:
        backward = getattr(loss, "backward", None)
        if callable(backward):
            backward()

    def step(self, optimizer: Any = None) -> None:
        optimizer = optimizer if optimizer is not None else self.optimizer
        if optimizer is None:
            return None
        if hasattr(optimizer, "step"):
            optimizer.step()
        if hasattr(optimizer, "zero_grad"):
            optimizer.zero_grad()
        return None

    def no_sync(self):
        return nullcontext()

    def state_dict(self) -> Dict[str, Any]:
        model = self._unwrap_model(self.model)
        adapter_only = bool(getattr(self.config, "adapter_only_checkpoint", False))
        adapter_state = (
            model.adapter_state_dict()
            if adapter_only and hasattr(model, "adapter_state_dict")
            else None
        )
        model_state = None
        if adapter_state is None and hasattr(self.model, "state_dict"):
            model_state = self.model.state_dict()
        optimizer_state = (
            self.optimizer.state_dict()
            if hasattr(self.optimizer, "state_dict")
            else None
        )
        return {
            "backend": self.name,
            "model_state_dict": model_state,
            "adapter_state_dict": adapter_state,
            "adapter_only_checkpoint": adapter_state is not None,
            "optimizer_state_dict": optimizer_state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        model = self._unwrap_model(self.model)
        if (
            model is not None
            and state.get("adapter_state_dict") is not None
            and hasattr(model, "load_adapter_state_dict")
        ):
            model.load_adapter_state_dict(state["adapter_state_dict"])
        elif self.model is not None and state.get("model_state_dict") is not None:
            self.model.load_state_dict(state["model_state_dict"])
        if self.optimizer is not None and state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        return None

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        while hasattr(model, "module"):
            model = model.module
        return model

    def save_checkpoint(
        self,
        checkpoint_manager: Any,
        step: Any = None,
        client_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(checkpoint_manager, str):
            return self._save_torch_checkpoint(
                checkpoint_manager, str(step or "checkpoint"), client_state
            )

        from parascale.checkpoint import CheckpointManifest

        return checkpoint_manager.write_manifest(
            CheckpointManifest(step=step, metadata=self.state_dict())
        )

    def load_checkpoint(
        self, checkpoint_manager: Any, step: Any = None, **kwargs: Any
    ) -> Any:
        if isinstance(checkpoint_manager, str):
            return self._load_torch_checkpoint(checkpoint_manager)
        return checkpoint_manager.read_manifest(step)

    def _save_torch_checkpoint(
        self, save_dir: str, tag: str, client_state: Optional[Dict[str, Any]] = None
    ) -> str:
        torch = _require_torch()
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{tag}.pt")
        torch.save(
            {"backend_state": self.state_dict(), "client_state": client_state or {}},
            path,
        )
        return path

    def _load_torch_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        torch = _require_torch()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        backend_state = checkpoint.get("backend_state", checkpoint)
        if "model_state_dict" in backend_state:
            self.load_state_dict(backend_state)
        return checkpoint.get("client_state", {})


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise ImportError("This backend operation requires PyTorch.") from exc
    return torch


__all__ = ["TrainingBackend", "_require_torch"]
