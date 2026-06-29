# -*- coding: utf-8 -*-
# @Time : 2026/6/9 下午8:12
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Declarative parallelism plans used by runtime backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ParallelDimension:
    name: str
    size: int = 1
    backend: str = "runtime"
    placement: str = "auto"

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"{self.name} size must be >= 1, got {self.size}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "backend": self.backend,
            "placement": self.placement,
        }


@dataclass
class ParallelPlan:
    data: ParallelDimension = field(default_factory=lambda: ParallelDimension("data"))
    tensor: ParallelDimension = field(
        default_factory=lambda: ParallelDimension("tensor")
    )
    pipeline: ParallelDimension = field(
        default_factory=lambda: ParallelDimension("pipeline")
    )
    sequence: ParallelDimension = field(
        default_factory=lambda: ParallelDimension("sequence")
    )
    expert: ParallelDimension = field(
        default_factory=lambda: ParallelDimension("expert")
    )
    sharding: str = "none"
    backend: str = "native"
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return (
            self.data.size
            * self.tensor.size
            * self.pipeline.size
            * self.sequence.size
            * self.expert.size
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "world_size": self.world_size,
            "dimensions": {
                "data": self.data.to_dict(),
                "tensor": self.tensor.to_dict(),
                "pipeline": self.pipeline.to_dict(),
                "sequence": self.sequence.to_dict(),
                "expert": self.expert.to_dict(),
            },
            "sharding": self.sharding,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def build_parallel_plan(config: Any, strategy_plan: Any = None) -> ParallelPlan:
    backend = getattr(strategy_plan, "backend", None) or getattr(
        config, "training_backend", "native"
    )
    sharding = "none"
    zero_stage = int(
        getattr(strategy_plan, "zero_stage", getattr(config, "zero_stage", 0)) or 0
    )
    if backend == "fsdp":
        sharding = "fsdp"
    elif backend == "deepspeed" or zero_stage > 0:
        sharding = f"zero_stage_{zero_stage}"

    plan = ParallelPlan(
        data=ParallelDimension(
            "data",
            int(
                getattr(
                    strategy_plan, "dp_size", getattr(config, "data_parallel_size", 1)
                )
                or 1
            ),
        ),
        tensor=ParallelDimension(
            "tensor",
            int(
                getattr(
                    strategy_plan, "tp_size", getattr(config, "tensor_parallel_size", 1)
                )
                or 1
            ),
            placement="homogeneous_first",
        ),
        pipeline=ParallelDimension(
            "pipeline",
            int(
                getattr(
                    strategy_plan,
                    "pp_size",
                    getattr(config, "pipeline_parallel_size", 1),
                )
                or 1
            ),
            placement=(
                "profile_required"
                if int(getattr(strategy_plan, "pp_size", 1) or 1) > 1
                else "auto"
            ),
        ),
        sequence=ParallelDimension(
            "sequence", int(getattr(config, "sequence_parallel_size", 1) or 1)
        ),
        expert=ParallelDimension(
            "expert", int(getattr(config, "expert_parallel_size", 1) or 1)
        ),
        sharding=sharding,
        backend=backend,
        reasons=[
            "parallelism is declarative; execution is delegated to the selected runtime backend"
        ],
    )
    if plan.tensor.size > 1 and backend in {"native", "fsdp"}:
        plan.warnings.append(
            "tensor parallel requires explicit TensorParallelAdapter integration for the target model"
        )
    if plan.pipeline.size > 1:
        plan.warnings.append(
            "pipeline parallel execution requires explicit benchmark validation"
        )
    return plan
