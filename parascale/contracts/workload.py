# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:54
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Workload adapter contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol

LossFn = Callable[[Any, Dict[str, Any]], Any]


class NormalizedLossFn(Protocol):
    """A mean loss with an explicit sample/token denominator for accumulation."""

    def __call__(self, output: Any, batch: Dict[str, Any]) -> Any: ...

    def normalization_weight(self, batch: Dict[str, Any]) -> float: ...


def causal_token_weight(batch: Mapping[str, Any]) -> float:
    """Count non-ignored shifted labels for a standard causal-LM mean loss."""
    return float((batch["labels"][..., 1:] != -100).sum().item())


@dataclass
class WorkloadComponents:
    model: Any
    optimizer: Any
    dataloader: Iterable[Dict[str, Any]]
    loss_fn: LossFn
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkloadAdapter(Protocol):
    name: str

    def build(self, config_data: Mapping[str, Any]) -> Any: ...
