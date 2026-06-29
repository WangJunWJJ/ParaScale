# -*- coding: utf-8 -*-
# @Time : 2026/6/9 下午5:59
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Shared runtime context for ParaScale v1.

This module is intentionally light: it records the decisions that every
production path needs before doing real work. Training, serving, planning, and
benchmarking should agree on the same task, topology, backend, and budget view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from parascale.config import ParaScaleConfig
from parascale.core import ClusterTopology
from parascale.strategy import StrategyPlan, build_strategy_plan


@dataclass(frozen=True)
class WorkloadDescriptor:
    """Task-level intent used by planner, runtime, and benchmark code."""

    task_type: str = "generic"
    model_family: str = "unknown"
    workload: str = "unknown"
    target_scale: str = "local"
    optimize_for: str = "balanced"
    modalities: List[str] = field(default_factory=list)

    @classmethod
    def from_config_data(
        cls, config: ParaScaleConfig, config_data: Dict[str, Any]
    ) -> "WorkloadDescriptor":
        task = _section(config_data, "task")
        training = _section(config_data, "training")
        serving = _section(config_data, "serving")
        data = _section(config_data, "data")
        task_type = str(task.get("type") or data.get("type") or config.task_type)
        modalities = task.get("modalities")
        if modalities is None:
            modalities = _infer_modalities(task_type)
        return cls(
            task_type=task_type,
            model_family=str(task.get("model_family") or config.model_family),
            workload=str(
                training.get("workload")
                or serving.get("workload")
                or task.get("workload")
                or "unknown"
            ),
            target_scale=str(task.get("target_scale") or config.target_scale),
            optimize_for=str(task.get("optimize_for") or config.optimize_for),
            modalities=[str(item) for item in modalities],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "model_family": self.model_family,
            "workload": self.workload,
            "target_scale": self.target_scale,
            "optimize_for": self.optimize_for,
            "modalities": list(self.modalities),
        }


@dataclass
class RuntimeContext:
    """The single source of runtime truth for a ParaScale process."""

    config: ParaScaleConfig
    workload: WorkloadDescriptor
    strategy_plan: StrategyPlan
    model_profile: Dict[str, Any] = field(default_factory=dict)
    hardware_profile: Dict[str, Any] = field(default_factory=dict)
    topology: Optional[ClusterTopology] = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    mode: str = "plan"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_vision_or_multimodal(self) -> bool:
        return self.workload.task_type in {"vision", "multimodal"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "is_distributed": self.is_distributed,
            "workload": self.workload.to_dict(),
            "strategy_plan": self.strategy_plan.to_dict(),
            "topology": self.topology.to_dict() if self.topology is not None else {},
            "model_profile": dict(self.model_profile),
            "hardware_profile": dict(self.hardware_profile),
            "budgets": {
                "batch_size": self.config.batch_size,
                "max_tokens_per_batch": self.config.max_tokens_per_batch,
                "max_patch_tokens_per_batch": self.config.max_patch_tokens_per_batch,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            },
        }


def build_runtime_context(
    config_data: Dict[str, Any],
    *,
    mode: str = "plan",
    rank: int = 0,
    local_rank: int = 0,
    world_size: Optional[int] = None,
) -> RuntimeContext:
    """Build the v1 context from a CLI/config dictionary."""

    config = ParaScaleConfig.from_dict(_section(config_data, "parascale"))
    model_profile = _section(config_data, "model_profile")
    hardware_profile = _section(config_data, "hardware_profile")
    topology = _topology_from_config(config_data, hardware_profile)
    workload = WorkloadDescriptor.from_config_data(config, config_data)

    if (
        workload.task_type == "vision"
        and config.max_patch_tokens_per_batch
        and config.batching_strategy == "sample"
    ):
        config.batching_strategy = "token_budget"
    if (
        workload.task_type == "multimodal"
        and config.max_tokens_per_batch
        and config.batching_strategy == "sample"
    ):
        config.batching_strategy = "token_budget"

    inferred_world_size = world_size or _infer_world_size(hardware_profile, topology)
    strategy_plan = build_strategy_plan(model_profile, hardware_profile, config)
    return RuntimeContext(
        config=config,
        workload=workload,
        strategy_plan=strategy_plan,
        model_profile=model_profile,
        hardware_profile=hardware_profile,
        topology=topology,
        rank=rank,
        local_rank=local_rank,
        world_size=inferred_world_size,
        mode=mode,
    )


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _topology_from_config(
    config_data: Dict[str, Any], hardware_profile: Dict[str, Any]
) -> Optional[ClusterTopology]:
    cluster = _section(config_data, "cluster")
    nodes = cluster.get("nodes") or hardware_profile.get("nodes")
    if isinstance(nodes, list) and nodes:
        return ClusterTopology.from_dicts(nodes)
    return None


def _infer_world_size(
    hardware_profile: Dict[str, Any], topology: Optional[ClusterTopology]
) -> int:
    if topology is not None and topology.world_size > 0:
        return topology.world_size
    return max(
        1,
        int(
            hardware_profile.get("num_gpus", hardware_profile.get("world_size", 1)) or 1
        ),
    )


def _infer_modalities(task_type: str) -> List[str]:
    if task_type == "llm":
        return ["text"]
    if task_type == "vision":
        return ["image"]
    if task_type == "multimodal":
        return ["text", "image"]
    return []
