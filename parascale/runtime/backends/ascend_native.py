# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:26
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Ascend native training backend entrypoint."""

from __future__ import annotations

import importlib.util
from typing import Any

from .base import TrainingBackend
from .devices import move_batch_to_device, resolve_torch_device, set_current_device


class AscendNativeTrainingBackend(TrainingBackend):
    name = "ascend_native"

    def setup(self):
        if importlib.util.find_spec("torch_npu") is None:
            raise RuntimeError(
                "Ascend native backend requires torch_npu and an Ascend NPU runtime."
            )
        import torch_npu  # noqa: F401

        return super().setup()

    def setup_model(self, model: Any) -> Any:
        if model is None:
            return model
        import torch
        import torch_npu  # noqa: F401

        device = set_current_device(torch, local_rank=self.local_rank)
        return model.to(device)

    def prepare_batch(self, batch: Any) -> Any:
        try:
            import torch
            import torch_npu  # noqa: F401
        except Exception:
            torch = None
        if torch is None:
            return move_batch_to_device(batch, f"npu:{self.local_rank}")
        device = resolve_torch_device(torch, local_rank=self.local_rank)
        return move_batch_to_device(batch, str(device))


__all__ = ["AscendNativeTrainingBackend"]
