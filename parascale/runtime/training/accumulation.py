# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午5:40
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Gradient accumulation controller for the training runtime."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Dict

from .metrics import merge_accumulated_batches, metric_value


class AccumulationController:
    """Run one optimizer step over one or more micro-batches."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def run(
        self,
        first_batch: Any,
        iterator: Any,
        *,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        loss_fn: Any = None,
        step_fn: Any = None,
        dataloader_wait_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        if step_fn is not None:
            raise RuntimeError(
                "gradient accumulation with custom step_fn is not supported by the "
                "default step_fn protocol. Provide model/loss_fn or implement an "
                "accumulation-aware step function before enabling accumulation."
            )

        accumulation_steps = self.engine._gradient_accumulation_steps()
        start = time.perf_counter()
        self.engine.memory.synchronize_device()
        batches = [first_batch]
        wait_total = float(dataloader_wait_seconds)
        for _ in range(1, accumulation_steps):
            wait_start = time.perf_counter()
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
            wait_total += time.perf_counter() - wait_start

        if len(batches) != accumulation_steps:
            raise RuntimeError(
                "Gradient accumulation requires "
                f"{accumulation_steps} micro-batches, but the dataloader "
                f"received {len(batches)} before exhaustion. Partial optimizer "
                "steps are forbidden because distributed ranks may diverge."
            )
        actual_steps = max(1, len(batches))
        losses = []
        for micro_index, batch in enumerate(batches):
            batch = self._prepare_batch(batch)
            batches[micro_index] = batch
            sync_context = (
                self.engine.training_backend.no_sync()
                if (
                    micro_index < actual_steps - 1
                    and self.engine.training_backend is not None
                )
                else nullcontext()
            )
            with sync_context:
                with self.engine.precision.autocast_context():
                    output = model(**batch) if isinstance(batch, dict) else model(batch)
                    loss = loss_fn(output, batch) / float(actual_steps)
                self.engine.backward(loss)
                losses.append(loss)

        self.engine.step(optimizer)
        self.engine.step_scheduler(scheduler)
        self.engine.memory.synchronize_device()
        elapsed = self.engine.memory.elapsed_since(start, synchronized=True)
        metrics: Dict[str, Any] = {
            "loss": metric_value(sum(losses)),
            "gradient_accumulation_steps": actual_steps,
        }
        metrics = self.engine._with_throughput_metrics(
            metrics,
            merge_accumulated_batches(batches),
            elapsed,
        )
        metrics["dataloader_wait_ms"] = wait_total * 1000.0
        return metrics

    def _prepare_batch(self, batch: Any) -> Any:
        backend = getattr(self.engine, "training_backend", None)
        prepare = getattr(backend, "prepare_batch", None)
        if callable(prepare):
            return prepare(batch)
        return batch
