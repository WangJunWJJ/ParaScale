# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:54
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Backend contract for training runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class BackendState:
    name: str
    distributed: bool = False
    zero_stage: int = 0
    sharded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "distributed": self.distributed,
            "zero_stage": self.zero_stage,
            "sharded": self.sharded,
            "metadata": dict(self.metadata),
        }


class BackendContract(Protocol):
    name: str

    def setup(self) -> tuple[Any, Any]: ...

    def backward(self, loss: Any) -> None: ...

    def step(self, optimizer: Any = None) -> None: ...

    def state_dict(self) -> Dict[str, Any]: ...

    def load_state_dict(self, state: Dict[str, Any]) -> None: ...
