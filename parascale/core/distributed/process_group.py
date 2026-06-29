# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Process group specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ProcessGroupSpec:
    name: str
    ranks: List[int]
    backend: str = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ranks": list(self.ranks),
            "backend": self.backend,
            "metadata": dict(self.metadata),
        }


__all__ = ["ProcessGroupSpec"]
