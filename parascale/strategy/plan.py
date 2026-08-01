# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Strategy plan data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

BackendName = Literal["native", "native_ddp", "fsdp", "deepspeed"]
BatchPolicy = Literal["sample", "length_bucket", "token_budget"]


@dataclass
class StrategyPlan:
    backend: BackendName = "native"
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    zero_stage: int = 0
    zero_offload: bool = False
    precision: str = "fp32"
    fsdp_state_dict_type: str = "full"
    ddp_comm_hook: str = "none"
    ddp_bucket_cap_mb: Optional[int] = None
    ddp_gradient_as_bucket_view: bool = True
    ddp_static_graph: bool = False
    activation_checkpointing: bool = False
    batch_policy: BatchPolicy = "sample"
    max_tokens_per_batch: Optional[int] = None
    checkpoint_policy: str = "rank0_file"
    estimated_memory_per_gpu: int = 0
    estimated_total_training_memory: int = 0
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    topology: Dict[str, Any] = field(default_factory=dict)
    communication_plan: Dict[str, Any] = field(default_factory=dict)

    @property
    def strategy_type(self) -> str:
        if self.tp_size > 1 and self.pp_size > 1:
            return "hybrid"
        if self.tp_size > 1:
            return "tensor"
        if self.pp_size > 1:
            return "pipeline"
        if self.dp_size > 1:
            return "data"
        return "single"

    def validate(self, world_size: int) -> bool:
        return self.dp_size * self.tp_size * self.pp_size == world_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "dp_size": self.dp_size,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "strategy_type": self.strategy_type,
            "zero_stage": self.zero_stage,
            "zero_offload": self.zero_offload,
            "precision": self.precision,
            "fsdp_state_dict_type": self.fsdp_state_dict_type,
            "ddp_comm_hook": self.ddp_comm_hook,
            "ddp_bucket_cap_mb": self.ddp_bucket_cap_mb,
            "ddp_gradient_as_bucket_view": self.ddp_gradient_as_bucket_view,
            "ddp_static_graph": self.ddp_static_graph,
            "activation_checkpointing": self.activation_checkpointing,
            "batch_policy": self.batch_policy,
            "max_tokens_per_batch": self.max_tokens_per_batch,
            "checkpoint_policy": self.checkpoint_policy,
            "estimated_memory_per_gpu": self.estimated_memory_per_gpu,
            "estimated_total_training_memory": self.estimated_total_training_memory,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "topology": dict(self.topology),
            "communication_plan": dict(self.communication_plan),
        }
