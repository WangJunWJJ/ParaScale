# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:51
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Single-step training lifecycle runner."""

from __future__ import annotations

import inspect
import time
from typing import Any, Dict

from .metrics import metric_value


class StepRunner:
    """Run one non-accumulated training step for TrainEngine-like runtimes."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def run_step(
        self,
        batch: Any,
        *,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        loss_fn: Any = None,
        step_fn: Any = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        self.engine.memory.synchronize_device()
        batch = self._prepare_batch(batch)
        if step_fn is not None:
            result = self.call_step_fn(step_fn, batch)
            metrics = (
                dict(result or {})
                if isinstance(result, dict)
                else {"step_result": result}
            )
            return self.engine._with_throughput_metrics(
                metrics,
                batch,
                self.engine.memory.elapsed_since(start),
            )

        with self.engine.precision.autocast_context():
            output = model(**batch) if isinstance(batch, dict) else model(batch)
            loss = loss_fn(output, batch)
        self.engine.backward(loss)
        self.engine.step(optimizer)
        self.engine.step_scheduler(scheduler)
        self.engine.memory.synchronize_device()
        metrics: Dict[str, Any] = {"loss": metric_value(loss)}
        if not isinstance(batch, dict) and hasattr(batch, "__len__"):
            metrics["batch_size"] = len(batch)
        return self.engine._with_throughput_metrics(
            metrics,
            batch,
            self.engine.memory.elapsed_since(start, synchronized=True),
        )

    def call_step_fn(self, step_fn: Any, batch: Any) -> Any:
        signature = inspect.signature(step_fn)
        if len(signature.parameters) >= 2:
            return step_fn(batch, self.engine)
        return step_fn(batch)

    def _prepare_batch(self, batch: Any) -> Any:
        backend = getattr(self.engine, "training_backend", None)
        prepare = getattr(backend, "prepare_batch", None)
        if callable(prepare):
            return prepare(batch)
        return batch


__all__ = ["StepRunner"]
