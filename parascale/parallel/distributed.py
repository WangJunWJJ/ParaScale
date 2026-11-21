# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit, bounded topology for experimental model-parallel execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedParallelConfig:
    """A DP x PP x TP rank grid, with TP ranks adjacent.

    This API is separate from the data-parallel workload CLI. Only single-node
    CUDA execution using at most four ranks is admitted until broader validation.
    The caller partitions stages and supplies the model-specific TP plan.
    """

    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    microbatches: int = 2
    pipeline_schedule: str = "gpipe"

    def __post_init__(self) -> None:
        for name in ("dp_size", "tp_size", "pp_size", "microbatches"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.pipeline_schedule) is not str or self.pipeline_schedule not in (
            "gpipe",
            "1f1b",
        ):
            raise ValueError("pipeline_schedule must be gpipe or 1f1b")
        if self.pipeline_schedule == "1f1b" and (
            self.pp_size == 1 or self.microbatches < self.pp_size
        ):
            raise ValueError("1f1b requires PP and microbatches >= pp_size")
        if self.world_size > 4:
            raise ValueError(
                "Execution is limited to four GPUs; full 3D is unvalidated"
            )
        if (self.dp_size, self.tp_size, self.pp_size) not in {
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 2, 1),
            (2, 1, 2),
            (1, 2, 2),
        }:
            raise ValueError(
                "This topology is outside the validated one/two/four-GPU matrix"
            )

    @property
    def world_size(self) -> int:
        return self.dp_size * self.pp_size * self.tp_size

    def validate_world_size(self, world_size: int) -> None:
        """Reject mismatched launcher sizes before creating any subgroups."""
        if world_size != self.world_size:
            raise ValueError(
                f"world size {world_size} does not match DP*PP*TP={self.world_size}"
            )

    def coordinates(self, rank: int) -> tuple[int, int, int]:
        """Return (data replica, pipeline stage, tensor shard) for a global rank."""
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")
        replica, remainder = divmod(rank, self.pp_size * self.tp_size)
        stage, shard = divmod(remainder, self.tp_size)
        return replica, stage, shard

    def global_batch_size(self, microbatch_size: int) -> int:
        """Count distinct samples, not replicated TP/PP work."""
        if type(microbatch_size) is not int or microbatch_size < 1:
            raise ValueError("microbatch_size must be a positive integer")
        return microbatch_size * self.microbatches * self.dp_size
