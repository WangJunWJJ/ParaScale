# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Serializable data pipeline plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DataLoaderPlan:
    """Serializable dataloader settings aligned with DeepSpeed training loops."""

    batch_sampler: str = "sample"
    batch_size: int = 1
    max_tokens_per_batch: Optional[int] = None
    max_patch_tokens_per_batch: Optional[int] = None
    task_type: str = "generic"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_sampler": self.batch_sampler,
            "batch_size": self.batch_size,
            "max_tokens_per_batch": self.max_tokens_per_batch,
            "max_patch_tokens_per_batch": self.max_patch_tokens_per_batch,
            "task_type": self.task_type,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "drop_last": self.drop_last,
            "seed": self.seed,
        }


def build_dataloader_plan(config: Any, world_size: int = 1) -> DataLoaderPlan:
    """Build dataloader knobs from ParaScale config using DeepSpeed-like defaults."""
    batch_policy = getattr(config, "batching_strategy", "sample")
    batch_size = int(getattr(config, "batch_size", 1))
    max_tokens = getattr(config, "max_tokens_per_batch", None)
    max_patch_tokens = getattr(config, "max_patch_tokens_per_batch", None)
    task_type = getattr(config, "task_type", "generic")
    if batch_policy == "token_budget" and max_tokens is None:
        max_tokens = 8192
    return DataLoaderPlan(
        batch_sampler=batch_policy,
        batch_size=(
            max(1, batch_size // max(1, world_size))
            if batch_policy == "sample"
            else batch_size
        ),
        max_tokens_per_batch=max_tokens,
        max_patch_tokens_per_batch=max_patch_tokens,
        task_type=task_type,
        num_workers=int(getattr(config, "dataloader_num_workers", 4)),
        pin_memory=bool(getattr(config, "dataloader_pin_memory", True)),
        prefetch_factor=int(getattr(config, "dataloader_prefetch_factor", 2)),
        persistent_workers=bool(getattr(config, "dataloader_persistent_workers", True)),
        drop_last=bool(getattr(config, "dataloader_drop_last", False)),
        seed=int(getattr(config, "seed", 42)),
    )
