# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Experimental v1 sequence-parallel adapter.

The remote ParaScale branch contains a larger Megatron/Ulysses-inspired
implementation. This local module captures the stable contract first: config,
shape planning, and safe single-process tensor helpers. Real distributed
collectives can be plugged in behind this contract later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

SequenceParallelMode = Literal["standard", "ulysses"]


@dataclass(frozen=True)
class SequenceParallelConfig:
    sp_size: int = 1
    tp_size: int = 1
    mode: SequenceParallelMode = "standard"
    sequence_dim: int = 1
    scatter_input: bool = True
    gather_output: bool = True
    enable_for_layernorm: bool = True
    enable_for_dropout: bool = True
    enable_for_activation: bool = True

    def __post_init__(self) -> None:
        if self.sp_size < 1:
            raise ValueError("sp_size must be >= 1")
        if self.tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if self.mode not in {"standard", "ulysses"}:
            raise ValueError(f"unsupported sequence parallel mode: {self.mode}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sp_size": self.sp_size,
            "tp_size": self.tp_size,
            "mode": self.mode,
            "sequence_dim": self.sequence_dim,
            "scatter_input": self.scatter_input,
            "gather_output": self.gather_output,
            "enable_for_layernorm": self.enable_for_layernorm,
            "enable_for_dropout": self.enable_for_dropout,
            "enable_for_activation": self.enable_for_activation,
        }


@dataclass(frozen=True)
class SequenceShardSpec:
    global_shape: Tuple[int, ...]
    local_shape: Tuple[int, ...]
    sequence_dim: int
    sp_size: int
    sp_rank: int
    start: int
    end: int
    padded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_shape": list(self.global_shape),
            "local_shape": list(self.local_shape),
            "sequence_dim": self.sequence_dim,
            "sp_size": self.sp_size,
            "sp_rank": self.sp_rank,
            "start": self.start,
            "end": self.end,
            "padded": self.padded,
        }


class SequenceParallelAdapter:
    """Small contract wrapper for sequence parallel planning and tensor slicing."""

    def __init__(
        self,
        config: Optional[SequenceParallelConfig] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config or SequenceParallelConfig()
        self.rank = rank
        self.world_size = world_size
        self.sp_rank = rank % self.config.sp_size

    def shard_spec(self, shape: Tuple[int, ...]) -> SequenceShardSpec:
        dim = _normalize_dim(self.config.sequence_dim, len(shape))
        seq_len = int(shape[dim])
        chunk = (seq_len + self.config.sp_size - 1) // self.config.sp_size
        start = min(seq_len, self.sp_rank * chunk)
        end = min(seq_len, start + chunk)
        local = list(shape)
        local[dim] = end - start
        return SequenceShardSpec(
            global_shape=tuple(shape),
            local_shape=tuple(local),
            sequence_dim=dim,
            sp_size=self.config.sp_size,
            sp_rank=self.sp_rank,
            start=start,
            end=end,
            padded=seq_len % self.config.sp_size != 0,
        )

    def scatter(self, tensor: Any) -> Any:
        """Return the local sequence shard for tensor-like objects supporting narrow."""

        spec = self.shard_spec(tuple(tensor.shape))
        narrow = getattr(tensor, "narrow", None)
        if not callable(narrow):
            raise TypeError(
                "scatter expects a tensor-like object with .shape and .narrow"
            )
        return narrow(spec.sequence_dim, spec.start, spec.end - spec.start).contiguous()

    def gather(self, local_tensor: Any) -> Any:
        """Single-process safe gather placeholder.

        Real distributed gather will be wired through the v1 collective backend.
        For sp_size=1 this is the identity, which keeps CPU/unit tests cheap.
        """

        if self.config.sp_size == 1:
            return local_tensor
        raise NotImplementedError(
            "distributed sequence gather is not wired into v1 collectives yet"
        )

    def plan(self, sample_shape: Tuple[int, ...]) -> Dict[str, Any]:
        return {
            "type": "sequence_parallel",
            "experimental": True,
            "config": self.config.to_dict(),
            "shard": self.shard_spec(sample_shape).to_dict(),
            "status": "metadata_and_single_process_scatter",
        }


def _normalize_dim(dim: int, rank: int) -> int:
    normalized = dim + rank if dim < 0 else dim
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"sequence_dim {dim} is out of range for rank {rank}")
    return normalized
