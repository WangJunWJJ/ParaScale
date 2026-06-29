# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午2:41
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pipeline-parallel stage planning and local schedule helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List


@dataclass(frozen=True)
class PipelineStage:
    stage_id: int
    start_layer: int
    end_layer: int
    virtual_chunks: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "num_layers": self.end_layer - self.start_layer,
            "virtual_chunks": self.virtual_chunks,
        }


def build_pipeline_stages(
    num_layers: int, pp_size: int, virtual_chunks: int = 1
) -> List[PipelineStage]:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if pp_size < 1:
        raise ValueError("pp_size must be >= 1")
    if virtual_chunks < 1:
        raise ValueError("virtual_chunks must be >= 1")
    stages: List[PipelineStage] = []
    for stage_id in range(pp_size):
        start = (num_layers * stage_id) // pp_size
        end = (num_layers * (stage_id + 1)) // pp_size
        stages.append(
            PipelineStage(
                stage_id=stage_id,
                start_layer=start,
                end_layer=end,
                virtual_chunks=virtual_chunks,
            )
        )
    return stages


class LocalPipelineExecutor:
    """Sequential executor that preserves the same stage contract as distributed PP."""

    def __init__(self, stages: Iterable[Callable[[Any], Any]]):
        self.stages = list(stages)
        if not self.stages:
            raise ValueError("LocalPipelineExecutor requires at least one stage")

    def run(self, batch: Any) -> Any:
        value = batch
        for stage in self.stages:
            value = stage(value)
        return value

    def run_microbatches(self, microbatches: Iterable[Any]) -> List[Any]:
        return [self.run(microbatch) for microbatch in microbatches]

    def plan(self) -> Dict[str, Any]:
        return {
            "type": "pipeline_parallel",
            "status": "local_sequential_schedule",
            "num_stages": len(self.stages),
        }
