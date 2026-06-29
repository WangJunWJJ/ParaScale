# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Small inference batcher skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class InferenceBatcher:
    max_batch_size: int = 1
    pending: List[Any] = field(default_factory=list)

    def submit(self, item: Any) -> None:
        self.pending.append(item)

    def next_batch(self) -> List[Any]:
        batch = self.pending[: self.max_batch_size]
        del self.pending[: self.max_batch_size]
        return batch


__all__ = ["InferenceBatcher"]
