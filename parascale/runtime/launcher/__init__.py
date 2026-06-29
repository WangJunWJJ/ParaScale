# -*- coding: utf-8 -*-
# @Time : 2026/6/9 下午5:59
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Launcher planning for local, torchrun, DeepSpeed, and Ascend runs."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from parascale.runtime.context import RuntimeContext

from .deepspeed import deepspeed_command
from .local import local_command
from .torchrun import torchrun_command


@dataclass(frozen=True)
class LaunchPlan:
    launcher: str
    command: List[str]
    world_size: int
    nproc_per_node: int
    nnodes: int = 1
    node_rank: int = 0
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    env: Dict[str, str] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "launcher": self.launcher,
            "command": list(self.command),
            "world_size": self.world_size,
            "nproc_per_node": self.nproc_per_node,
            "nnodes": self.nnodes,
            "node_rank": self.node_rank,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "env": dict(self.env),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def build_launch_plan(
    context: RuntimeContext,
    *,
    entrypoint: str = "python -m parascale.cli train",
    config_path: str = "config.yaml",
    nnodes: Optional[int] = None,
    node_rank: int = 0,
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
) -> LaunchPlan:
    """Return the recommended launcher for the current context.

    This is deliberately conservative. ParaScale should make a clear plan and
    fail with diagnostics before it tries a fragile multi-process launch.
    """

    backend = context.strategy_plan.backend
    world_size = max(1, context.world_size)
    nproc = _gpus_per_node(context)
    resolved_nnodes = max(1, int(nnodes or _infer_nnodes(context, world_size, nproc)))
    resolved_world_size = max(world_size, resolved_nnodes * nproc)
    reasons = [
        f"task={context.workload.task_type}",
        f"backend={backend}",
        f"target_scale={context.workload.target_scale}",
    ]
    env = {
        "PYTHONUNBUFFERED": "1",
        "PARASCALE_TASK_TYPE": context.workload.task_type,
    }
    entrypoint_args = shlex.split(entrypoint)

    if world_size == 1 and backend == "native":
        return LaunchPlan(
            launcher="local",
            command=local_command(entrypoint_args, config_path),
            world_size=1,
            nproc_per_node=1,
            nnodes=1,
            node_rank=0,
            master_addr=master_addr,
            master_port=master_port,
            env=env,
            reasons=[*reasons, "single-process native path"],
        )

    if backend == "deepspeed":
        command = deepspeed_command(entrypoint_args, config_path, nproc_per_node=nproc)
        return LaunchPlan(
            launcher="deepspeed",
            command=command,
            world_size=resolved_world_size,
            nproc_per_node=nproc,
            nnodes=resolved_nnodes,
            node_rank=int(node_rank),
            master_addr=master_addr,
            master_port=master_port,
            env=env,
            reasons=[*reasons, "DeepSpeed backend requested"],
            warnings=_distributed_warnings(context),
        )

    command = torchrun_command(
        entrypoint_args,
        config_path,
        world_size=resolved_world_size,
        nproc_per_node=nproc,
        nnodes=resolved_nnodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    return LaunchPlan(
        launcher="torchrun",
        command=command,
        world_size=resolved_world_size,
        nproc_per_node=nproc,
        nnodes=resolved_nnodes,
        node_rank=int(node_rank),
        master_addr=master_addr,
        master_port=master_port,
        env=env,
        reasons=[*reasons, "distributed torch runtime path"],
        warnings=_distributed_warnings(context),
    )


def _gpus_per_node(context: RuntimeContext) -> int:
    hardware = context.hardware_profile
    value = (
        hardware.get("gpus_per_node") or hardware.get("num_gpus") or context.world_size
    )
    return max(1, int(value or 1))


def _infer_nnodes(context: RuntimeContext, world_size: int, nproc: int) -> int:
    hardware = context.hardware_profile
    value = hardware.get("num_nodes") or hardware.get("nnodes")
    if value:
        return max(1, int(value))
    return max(1, int(world_size) // max(1, int(nproc)))


def _distributed_warnings(context: RuntimeContext) -> List[str]:
    warnings: List[str] = []
    if (
        context.workload.task_type in {"vision", "multimodal"}
        and context.config.max_patch_tokens_per_batch is None
    ):
        warnings.append(
            "vision/multimodal runs should set max_patch_tokens_per_batch for stable memory use"
        )
    if context.world_size > 8 and context.strategy_plan.backend == "native":
        warnings.append(
            "native backend is not recommended beyond one node; use FSDP or DeepSpeed baseline"
        )
    if context.topology is not None and context.topology.is_heterogeneous:
        warnings.append(
            "heterogeneous topology detected; high-frequency TP/FSDP groups should stay homogeneous"
        )
    return warnings
