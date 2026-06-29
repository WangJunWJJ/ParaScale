# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:26
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Serving engine built on top of ServeEngine primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from parascale.runtime import ServeEngine

from .api import ServeRequest, ServeResponse
from .kv_cache import KVCacheManager
from .sampler import SamplingConfig
from .scheduler import ContinuousBatchScheduler


@dataclass
class ServingEngine:
    runtime: ServeEngine = field(default_factory=ServeEngine)
    scheduler: ContinuousBatchScheduler = field(
        default_factory=ContinuousBatchScheduler
    )
    kv_cache: KVCacheManager = field(default_factory=KVCacheManager)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    requests_completed: int = 0
    requests_failed: int = 0
    batches_completed: int = 0
    total_latency_ms: float = 0.0

    def submit(self, request: ServeRequest) -> None:
        self.scheduler.submit(request)

    def step(self) -> List[ServeResponse]:
        batch = self.scheduler.next_batch()
        if not batch:
            return []
        start = time.perf_counter()
        for request in batch:
            self.kv_cache.put(request.request_id, {"status": "prefill"})
        try:
            result = self.runtime.generate([request.payload for request in batch])
            outputs = self._normalize_outputs(result.get("outputs"), len(batch))
            mode = result.get("mode", "unknown")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            responses = []
            for request, output in zip(batch, outputs):
                queued_at = request.metadata.get("queued_at")
                queue_latency_ms = 0.0
                if isinstance(queued_at, (int, float)):
                    queue_latency_ms = max(0.0, (start - float(queued_at)) * 1000.0)
                self.kv_cache.put(request.request_id, {"status": "decoded"})
                responses.append(
                    ServeResponse(
                        request_id=request.request_id,
                        output=output,
                        metadata={
                            "mode": mode,
                            "batch_size": len(batch),
                            "latency_ms": elapsed_ms,
                            "queue_latency_ms": queue_latency_ms,
                            "sampling": self.sampling.to_dict(),
                        },
                    )
                )
                self.kv_cache.release(request.request_id)
            self.requests_completed += len(responses)
            self.batches_completed += 1
            self.total_latency_ms += elapsed_ms
            return responses
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.requests_failed += len(batch)
            self.total_latency_ms += elapsed_ms
            for request in batch:
                self.kv_cache.release(request.request_id)
            return [
                ServeResponse(
                    request_id=request.request_id,
                    output=None,
                    error=str(exc),
                    metadata={
                        "mode": "error",
                        "batch_size": len(batch),
                        "latency_ms": elapsed_ms,
                    },
                )
                for request in batch
            ]

    def drain(self) -> List[ServeResponse]:
        responses: List[ServeResponse] = []
        while self.scheduler.pending():
            responses.extend(self.step())
        return responses

    def metrics(self) -> Dict[str, Any]:
        total_requests = self.requests_completed + self.requests_failed
        average_latency_ms = self.total_latency_ms / max(1, self.batches_completed)
        return {
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
            "total_requests": total_requests,
            "batches_completed": self.batches_completed,
            "average_batch_latency_ms": average_latency_ms,
            "scheduler": self.scheduler.stats(),
            "kv_cache": self.kv_cache.stats(),
        }

    @staticmethod
    def _normalize_outputs(outputs: Any, batch_size: int) -> List[Any]:
        if outputs is None:
            return [None for _ in range(batch_size)]
        if isinstance(outputs, list):
            if len(outputs) == batch_size:
                return outputs
            raise ValueError(
                f"runtime returned {len(outputs)} outputs for batch size {batch_size}"
            )
        return [outputs for _ in range(batch_size)]
