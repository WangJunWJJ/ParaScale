# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Continuous batching scheduler with queue metrics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

from .api import ServeRequest


@dataclass
class ContinuousBatchScheduler:
    max_batch_size: int = 8
    queue: Deque[ServeRequest] = field(default_factory=deque)
    submitted: int = 0
    dispatched: int = 0

    def submit(self, request: ServeRequest) -> None:
        request.metadata.setdefault("queued_at", time.perf_counter())
        self.queue.append(request)
        self.submitted += 1

    def next_batch(self) -> List[ServeRequest]:
        batch: List[ServeRequest] = []
        while self.queue and len(batch) < self.max_batch_size:
            batch.append(self.queue.popleft())
        self.dispatched += len(batch)
        return batch

    def pending(self) -> int:
        return len(self.queue)

    def stats(self) -> dict:
        return {
            "max_batch_size": self.max_batch_size,
            "pending": len(self.queue),
            "submitted": self.submitted,
            "dispatched": self.dispatched,
        }
