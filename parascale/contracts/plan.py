# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Lightweight runtime plan contracts.

These dataclasses describe cross-module boundaries. They are created before a
run and passed by shallow reference during execution; they are not per-step
validation wrappers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class DevicePlan:
    kind: str = "cpu"
    communication_backend: str = "gloo"
    local_rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendPlan:
    name: str = "native"
    distributed: bool = False
    zero_stage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommunicationPlan:
    backend: str = "gloo"
    ddp_hook: str = "none"
    no_sync: bool = False
    overlap_h2d: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataPlan:
    kind: str = "generic"
    cache_enabled: bool = False
    num_workers: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointPlan:
    enabled: bool = True
    interval_steps: int = 0
    adapter_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferencePlan:
    enabled: bool = False
    task: str = "none"
    batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePlan:
    mode: str
    device: DevicePlan
    backend: BackendPlan
    communication: CommunicationPlan
    data: DataPlan
    checkpoint: CheckpointPlan
    inference: InferencePlan
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "BackendPlan",
    "CheckpointPlan",
    "CommunicationPlan",
    "DataPlan",
    "DevicePlan",
    "InferencePlan",
    "RuntimePlan",
]
