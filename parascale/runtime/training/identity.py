# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inspect final runtime objects once, outside the timed training loop."""

from __future__ import annotations

import warnings
from typing import Any

from parascale.runtime.lifecycle import initialized_distributed

from .coordination import coordinated_call


def collect_execution_identity(engine: Any) -> list[dict[str, Any]]:
    """Return rank-local optimizer and scheduler identities from all active ranks."""
    def snapshot():
        optimizer = engine.training_backend.optimizer
        scheduler = engine.scheduler
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from torch.distributed.optim import ZeroRedundancyOptimizer
        except ImportError:
            zero_optimizer = None
        else:
            zero_optimizer = isinstance(optimizer, ZeroRedundancyOptimizer)

        dist = initialized_distributed()
        return {
            "rank": int(dist.get_rank()) if dist is not None else 0,
            "world_size": int(dist.get_world_size()) if dist is not None else 1,
            "distributed_backend": str(dist.get_backend()) if dist is not None else None,
            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "zero_redundancy_optimizer": zero_optimizer,
            "scheduler_bound_to_optimizer": (
                scheduler.optimizer is optimizer if scheduler is not None else None
            ),
        }

    local = coordinated_call(snapshot, phase="execution identity")
    dist = initialized_distributed()
    if dist is None:
        return [local]
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, local)
    return records
