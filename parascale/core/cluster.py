# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Cluster topology model for homogeneous and heterogeneous deployments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from parascale.strategy import (
    HeterogeneousParallelPlan,
    build_heterogeneous_parallel_plan,
)


@dataclass(frozen=True)
class DeviceSpec:
    kind: str
    count: int
    memory_bytes: int = 0
    interconnect: str = "unknown"
    compute_weight: float = 1.0
    bandwidth_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class NodeSpec:
    hostname: str
    devices: DeviceSpec
    zone: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "devices": self.devices.to_dict(),
            "zone": self.zone,
        }


@dataclass
class ClusterTopology:
    nodes: List[NodeSpec] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return sum(max(0, node.devices.count) for node in self.nodes)

    @property
    def device_kinds(self) -> List[str]:
        return sorted({node.devices.kind for node in self.nodes})

    @property
    def is_heterogeneous(self) -> bool:
        return len(self.device_kinds) > 1

    @classmethod
    def from_dicts(cls, nodes: Iterable[Dict[str, Any]]) -> "ClusterTopology":
        specs = []
        for index, node in enumerate(nodes):
            device = node.get("devices", node)
            device_spec = DeviceSpec(
                kind=str(device.get("kind", device.get("device_type", "gpu"))).lower(),
                count=int(
                    device.get(
                        "count",
                        device.get("device_count", device.get("num_devices", 1)),
                    )
                    or 1
                ),
                memory_bytes=int(
                    device.get("memory_bytes", device.get("gpu_memory", 0)) or 0
                ),
                interconnect=str(device.get("interconnect", "unknown")),
                compute_weight=float(device.get("compute_weight", 1.0) or 1.0),
                bandwidth_weight=float(device.get("bandwidth_weight", 1.0) or 1.0),
            )
            specs.append(
                NodeSpec(
                    hostname=str(node.get("hostname", f"node-{index}")),
                    devices=device_spec,
                    zone=str(node.get("zone", "default")),
                )
            )
        return cls(specs)

    def build_parallel_plan(self) -> HeterogeneousParallelPlan:
        return build_heterogeneous_parallel_plan(
            {
                "device_type": node.devices.kind,
                "device_count": node.devices.count,
                "memory_bytes": node.devices.memory_bytes,
                "compute_weight": node.devices.compute_weight,
                "bandwidth_weight": node.devices.bandwidth_weight,
            }
            for node in self.nodes
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "world_size": self.world_size,
            "device_kinds": self.device_kinds,
            "is_heterogeneous": self.is_heterogeneous,
        }
