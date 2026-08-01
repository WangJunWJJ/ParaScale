# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:58
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Native PyTorch training backends."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from .base import TrainingBackend, _require_torch
from .devices import (
    current_accelerator,
    set_current_device,
)


class NativeTrainingBackend(TrainingBackend):
    name = "native"

    def setup_model(self, model: Any) -> Any:
        if model is None:
            return None
        torch = _require_torch()
        accelerator = current_accelerator(torch)
        if accelerator in {"cuda", "npu"}:
            device = set_current_device(torch, local_rank=self.local_rank)
            return model.to(device)
        return model

    def setup_optimizer(self, optimizer: Any) -> Any:
        if optimizer is None:
            return None
        zero_stage = int(getattr(self.config, "zero_stage", 0) or 0)
        if zero_stage == 0:
            return optimizer
        if zero_stage != 1:
            raise NotImplementedError(
                "native backend supports only ZeRO Stage 1 via PyTorch ZeroRedundancyOptimizer; "
                "use DeepSpeed/FSDP for Stage 2/3."
            )
        _require_torch()
        import torch.distributed as dist

        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() <= 1
        ):
            return optimizer
        from parascale.optimizers import create_native_zero_redundancy_optimizer

        params = [
            param
            for group in optimizer.param_groups
            for param in group.get("params", [])
        ]
        kwargs = dict(getattr(optimizer, "defaults", {}))
        return create_native_zero_redundancy_optimizer(
            params,
            optimizer.__class__,
            stage=1,
            **kwargs,
        )


class NativeDdpTrainingBackend(NativeTrainingBackend):
    name = "native_ddp"

    def setup_model(self, model: Any) -> Any:
        if model is None:
            return None
        torch = _require_torch()
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() <= 1
        ):
            return model
        accelerator = current_accelerator(torch)
        if accelerator == "cuda":
            device = set_current_device(torch, local_rank=self.local_rank)
            model = model.to(device)
            wrapped = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                **self._ddp_common_kwargs(),
            )
            self._register_comm_hook(wrapped)
            return wrapped
        if accelerator == "npu":
            device = set_current_device(torch, local_rank=self.local_rank)
            model = model.to(device)
            wrapped = self._wrap_npu_ddp(
                DistributedDataParallel,
                model,
                int(str(device).split(":", 1)[1]),
            )
            self._register_comm_hook(wrapped)
            return wrapped
        wrapped = DistributedDataParallel(
            model,
            **self._ddp_common_kwargs(),
        )
        self._register_comm_hook(wrapped)
        return wrapped

    def prepare_batch(self, batch: Any) -> Any:
        return super().prepare_batch(batch)

    def _wrap_npu_ddp(self, ddp_cls: Any, model: Any, device_id: int) -> Any:
        kwargs = self._ddp_common_kwargs()
        try:
            return ddp_cls(model, device_ids=[device_id], **kwargs)
        except (TypeError, ValueError):
            return ddp_cls(model, **kwargs)

    def _ddp_common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "find_unused_parameters": bool(
                getattr(self.config, "ddp_find_unused_parameters", False)
            ),
            "gradient_as_bucket_view": bool(
                getattr(self.config, "ddp_gradient_as_bucket_view", True)
            ),
            "static_graph": self._static_graph_enabled(),
        }
        bucket_cap_mb = getattr(self.config, "ddp_bucket_cap_mb", None)
        if bucket_cap_mb is not None:
            kwargs["bucket_cap_mb"] = int(bucket_cap_mb)
        return kwargs

    def _static_graph_enabled(self) -> bool:
        return (
            bool(getattr(self.config, "ddp_static_graph", False))
            and int(getattr(self.config, "gradient_accumulation_steps", 1) or 1) <= 1
        )

    def _register_comm_hook(self, model: Any) -> None:
        hook_name = str(getattr(self.config, "ddp_comm_hook", "none") or "none")
        if hook_name == "auto":
            from parascale.communication import recommend_ddp_hook

            hook_name = recommend_ddp_hook(
                precision=str(getattr(self.config, "precision", "fp32")),
                task_type=str(getattr(self.config, "task_type", "")),
                model_family=str(getattr(self.config, "model_family", "")),
            ).hook
        if hook_name == "none":
            return
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        except Exception as exc:
            raise ImportError(
                "DDP communication hooks require PyTorch distributed hooks."
            ) from exc
        hook = {
            "fp16_compress": default_hooks.fp16_compress_hook,
            "bf16_compress": default_hooks.bf16_compress_hook,
        }.get(hook_name)
        if hook is None:
            raise ValueError(f"unsupported ddp_comm_hook: {hook_name}")
        model.register_comm_hook(state=None, hook=hook)

    def no_sync(self):
        if self.model is not None and hasattr(self.model, "no_sync"):
            return self.model.no_sync()
        return nullcontext()


__all__ = ["NativeDdpTrainingBackend", "NativeTrainingBackend"]
