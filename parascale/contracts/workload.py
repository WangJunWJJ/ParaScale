# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:54
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Workload adapter contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol

LossFn = Callable[[Any, Dict[str, Any]], Any]


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
