# -*- coding: utf-8 -*-
# @Time : 2026/5/3 下午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Heterogeneous cluster placement planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class NodeGroup:
    name: str
    device_type: str
    ranks: List[int]
    memory_bytes: int = 0
    bandwidth_weight: float = 1.0
    compute_weight: float = 1.0

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device_type": self.device_type,
            "ranks": list(self.ranks),
            "world_size": self.world_size,
            "memory_bytes": self.memory_bytes,
            "bandwidth_weight": self.bandwidth_weight,
            "compute_weight": self.compute_weight,
        }


@dataclass
class HeterogeneousParallelPlan:
    groups: List[NodeGroup] = field(default_factory=list)
    placement_policy: str = "homogeneous_fast_path"
    cross_group_parallelism: str = "replicated_data_parallel"
    warnings: List[str] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return sum(group.world_size for group in self.groups)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groups": [group.to_dict() for group in self.groups],
            "world_size": self.world_size,
            "placement_policy": self.placement_policy,
            "cross_group_parallelism": self.cross_group_parallelism,
            "warnings": list(self.warnings),
        }


def build_heterogeneous_parallel_plan(
    nodes: Iterable[Any],
) -> HeterogeneousParallelPlan:
    groups: Dict[str, NodeGroup] = {}
    next_rank = 0
    for index, node in enumerate(nodes):
        device_type = str(
            _get(node, "device_type", _get(node, "accelerator", "gpu"))
        ).lower()
        device_count = int(
            _get(node, "device_count", _get(node, "num_devices", 1)) or 1
        )
        memory_bytes = int(_get(node, "memory_bytes", _get(node, "gpu_memory", 0)) or 0)
        compute_weight = float(_get(node, "compute_weight", 1.0) or 1.0)
        bandwidth_weight = float(_get(node, "bandwidth_weight", 1.0) or 1.0)
        key = device_type
        ranks = list(range(next_rank, next_rank + device_count))
        next_rank += device_count
        previous = groups.get(key)
        if previous is None:
            groups[key] = NodeGroup(
                name=f"{device_type}-{index}",
                device_type=device_type,
                ranks=ranks,
                memory_bytes=memory_bytes,
                compute_weight=compute_weight,
                bandwidth_weight=bandwidth_weight,
            )
        else:
            groups[key] = NodeGroup(
                name=previous.name,
                device_type=device_type,
                ranks=previous.ranks + ranks,
                memory_bytes=min(
                    previous.memory_bytes or memory_bytes,
                    memory_bytes or previous.memory_bytes,
                ),
                compute_weight=min(previous.compute_weight, compute_weight),
                bandwidth_weight=min(previous.bandwidth_weight, bandwidth_weight),
            )

    group_list = list(groups.values())
    if len(group_list) <= 1:
        return HeterogeneousParallelPlan(groups=group_list)

    return HeterogeneousParallelPlan(
        groups=group_list,
        placement_policy="heterogeneous_islands",
        cross_group_parallelism="weighted_data_parallel",
        warnings=[
            "Mixed accelerators are isolated into device islands; use cross-group data parallelism unless kernels are verified."
        ],
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
