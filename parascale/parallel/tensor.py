# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午2:40
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Tensor-parallel planning and local tensor helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TensorShardSpec:
    global_shape: Tuple[int, ...]
    local_shape: Tuple[int, ...]
    dim: int
    tp_size: int
    tp_rank: int
    start: int
    end: int
    padded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_shape": list(self.global_shape),
            "local_shape": list(self.local_shape),
            "dim": self.dim,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "start": self.start,
            "end": self.end,
            "padded": self.padded,
        }


class TensorParallelAdapter:
    """Local tensor slicing contract for future distributed TP kernels."""

    def __init__(self, tp_size: int = 1, rank: int = 0, dim: int = -1):
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        self.tp_size = int(tp_size)
        self.rank = int(rank)
        self.tp_rank = self.rank % self.tp_size
        self.dim = int(dim)

    def shard_spec(self, shape: Tuple[int, ...]) -> TensorShardSpec:
        dim = _normalize_dim(self.dim, len(shape))
        width = int(shape[dim])
        chunk = (width + self.tp_size - 1) // self.tp_size
        start = min(width, self.tp_rank * chunk)
        end = min(width, start + chunk)
        local = list(shape)
        local[dim] = end - start
        return TensorShardSpec(
            global_shape=tuple(shape),
            local_shape=tuple(local),
            dim=dim,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            start=start,
            end=end,
            padded=width % self.tp_size != 0,
        )

    def shard(self, tensor: Any) -> Any:
        spec = self.shard_spec(tuple(tensor.shape))
        narrow = getattr(tensor, "narrow", None)
        if not callable(narrow):
            raise TypeError(
                "shard expects a tensor-like object with .shape and .narrow"
            )
        return narrow(spec.dim, spec.start, spec.end - spec.start).contiguous()

    def gather(self, shards: Any, dim: int | None = None) -> Any:
        torch = _require_torch()
        if not isinstance(shards, (list, tuple)):
            if self.tp_size == 1:
                return shards
            raise TypeError("gather expects a list of local shards when tp_size > 1")
        return torch.cat(
            list(shards),
            dim=_normalize_dim(self.dim if dim is None else dim, len(shards[0].shape)),
        )

    def plan(self, sample_shape: Tuple[int, ...]) -> Dict[str, Any]:
        return {
            "type": "tensor_parallel",
            "status": "local_shard_contract",
            "shard": self.shard_spec(sample_shape).to_dict(),
        }


def column_parallel_linear(
    input_tensor: Any, weight: Any, bias: Any = None, *, tp_size: int = 1, rank: int = 0
) -> Any:
    """Run the local slice of a column-parallel linear layer."""
    torch = _require_torch()
    adapter = TensorParallelAdapter(tp_size=tp_size, rank=rank, dim=0)
    local_weight = adapter.shard(weight)
    local_bias = adapter.shard(bias) if bias is not None else None
    return torch.nn.functional.linear(input_tensor, local_weight, local_bias)


def row_parallel_linear(
    input_tensor: Any, weight: Any, bias: Any = None, *, tp_size: int = 1, rank: int = 0
) -> Any:
    """Run the local matmul contribution of a row-parallel linear layer."""
    torch = _require_torch()
    input_adapter = TensorParallelAdapter(tp_size=tp_size, rank=rank, dim=-1)
    weight_adapter = TensorParallelAdapter(tp_size=tp_size, rank=rank, dim=1)
    local_input = input_adapter.shard(input_tensor)
    local_weight = weight_adapter.shard(weight)
    output = torch.nn.functional.linear(local_input, local_weight, None)
    if tp_size == 1 and bias is not None:
        output = output + bias
    return output


def _normalize_dim(dim: int, rank: int) -> int:
    normalized = dim + rank if dim < 0 else dim
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"dim {dim} is out of range for rank {rank}")
    return normalized


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise ImportError("tensor parallel helpers require PyTorch") from exc
    return torch
