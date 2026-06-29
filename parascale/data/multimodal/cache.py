# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:57
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Lightweight multimodal cache primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MultiModalMemoryCache:
    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def clear(self) -> None:
        self.values.clear()


__all__ = ["MultiModalMemoryCache"]
