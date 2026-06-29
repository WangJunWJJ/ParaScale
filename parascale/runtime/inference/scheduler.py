# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inference scheduler skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .batcher import InferenceBatcher


@dataclass
class InferenceScheduler:
    batcher: InferenceBatcher

    def submit(self, item: Any) -> None:
        self.batcher.submit(item)

    def step(self) -> List[Any]:
        return self.batcher.next_batch()


__all__ = ["InferenceScheduler"]
