# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Experimental ZeRO planning and wrapper for ParaScale v1.

The remote prototype contains a fuller ZeRO optimizer attempt. For v1 we keep a
truthful contract: plan memory, expose metadata, and wrap a base optimizer
without pretending to implement DeepSpeed-equivalent Stage 2/3 sharding.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Type


class ZeroStage(IntEnum):
    DISABLED = 0
    OPTIMIZER_STATES = 1
    GRADIENTS = 2
    PARAMETERS = 3


@dataclass(frozen=True)
class ZeroPlan:
    stage: ZeroStage = ZeroStage.DISABLED
    world_size: int = 1
    offload_optimizer: bool = False
    offload_params: bool = False
    overlap_comm: bool = False
    estimated_memory_savings: float = 1.0
    implementation_status: str = "disabled"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": int(self.stage),
            "world_size": self.world_size,
            "offload_optimizer": self.offload_optimizer,
            "offload_params": self.offload_params,
            "overlap_comm": self.overlap_comm,
            "estimated_memory_savings": self.estimated_memory_savings,
            "implementation_status": self.implementation_status,
        }


def build_zero_plan(
    *,
    stage: int = 0,
    world_size: int = 1,
    offload_optimizer: bool = False,
    offload_params: bool = False,
    overlap_comm: bool = False,
) -> ZeroPlan:
    if stage not in {0, 1, 2, 3}:
        raise ValueError("ZeRO stage must be 0, 1, 2, or 3")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    zero_stage = ZeroStage(stage)
    if zero_stage == ZeroStage.DISABLED:
        savings = 1.0
        status = "disabled"
    elif zero_stage == ZeroStage.OPTIMIZER_STATES:
        savings = float(world_size)
        status = "wrapper_metadata_only"
    elif zero_stage == ZeroStage.GRADIENTS:
        savings = float(world_size) * 1.5
        status = "requires_backend_sharding"
    else:
        savings = float(world_size) * 2.0
        status = "requires_backend_parameter_sharding"
    if offload_optimizer or offload_params:
        status += "_with_offload_plan"
    return ZeroPlan(
        stage=zero_stage,
        world_size=world_size,
        offload_optimizer=offload_optimizer,
        offload_params=offload_params,
        overlap_comm=overlap_comm,
        estimated_memory_savings=savings,
        implementation_status=status,
    )


class ExperimentalZeroOptimizer:
    """A transparent optimizer wrapper with honest ZeRO metadata."""

    def __init__(
        self,
        base_optimizer: Any,
        *,
        stage: int = 1,
        world_size: int = 1,
        rank: int = 0,
        offload_optimizer: bool = False,
        offload_params: bool = False,
        overlap_comm: bool = False,
    ):
        if base_optimizer is None:
            raise ValueError("base_optimizer is required")
        self.base_optimizer = base_optimizer
        self.stage = ZeroStage(stage)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.plan = build_zero_plan(
            stage=int(self.stage),
            world_size=self.world_size,
            offload_optimizer=offload_optimizer,
            offload_params=offload_params,
            overlap_comm=overlap_comm,
        )
        if self.stage >= ZeroStage.GRADIENTS:
            warnings.warn(
                "ExperimentalZeroOptimizer does not implement real Stage 2/3 sharding yet; "
                "use DeepSpeed/FSDP backend for production sharding.",
                RuntimeWarning,
            )
        self.param_groups = base_optimizer.param_groups

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_optimizer.step(*args, **kwargs)

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "zero_plan": self.plan.to_dict(),
            "base_optimizer": self.base_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        base_state = state_dict.get("base_optimizer", state_dict)
        self.base_optimizer.load_state_dict(base_state)

    def get_memory_stats(self) -> Dict[str, Any]:
        total_params = sum(
            p.numel()
            for group in self.param_groups
            for p in group.get("params", [])
            if hasattr(p, "numel")
        )
        return {
            "total_parameters": total_params,
            "rank": self.rank,
            "world_size": self.world_size,
            "zero_plan": self.plan.to_dict(),
        }


def wrap_zero_optimizer(
    base_optimizer: Any, config: Optional[Dict[str, Any]] = None
) -> ExperimentalZeroOptimizer:
    config = config or {}
    return ExperimentalZeroOptimizer(
        base_optimizer,
        stage=int(config.get("stage", 1)),
        world_size=int(config.get("world_size", 1)),
        rank=int(config.get("rank", 0)),
        offload_optimizer=bool(config.get("offload_optimizer", False)),
        offload_params=bool(config.get("offload_params", False)),
        overlap_comm=bool(config.get("overlap_comm", False)),
    )


def create_native_zero_redundancy_optimizer(
    params: Any,
    optimizer_class: Type[Any],
    *,
    stage: int = 1,
    **optimizer_kwargs: Any,
) -> Any:
    """Create a real native ZeRO Stage 1 optimizer using PyTorch ZRO.

    Native Stage 2/3 sharding is intentionally not claimed here. Those modes
    should use DeepSpeed/FSDP until ParaScale owns tested gradient/parameter
    sharding kernels.
    """

    if int(stage) != 1:
        raise NotImplementedError(
            "native ZeRO currently supports only Stage 1 optimizer-state sharding; "
            "use DeepSpeed/FSDP for Stage 2/3."
        )
    try:
        import torch.distributed as dist
    except Exception as exc:
        raise ImportError("native ZeRO Stage 1 requires torch.distributed") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "native ZeRO Stage 1 requires an initialized torch.distributed process group"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from torch.distributed.optim import ZeroRedundancyOptimizer
    except Exception as exc:
        raise ImportError(
            "native ZeRO Stage 1 requires torch.distributed.optim.ZeroRedundancyOptimizer"
        ) from exc
    return ZeroRedundancyOptimizer(
        params, optimizer_class=optimizer_class, **optimizer_kwargs
    )
