# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午4:11
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Generic inference runner with device placement and runtime metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

from parascale.runtime.backends.devices import move_batch_to_device
from parascale.runtime.training.memory import RuntimeMemoryTracker


@dataclass
class InferenceRunner:
    model: Any
    task: str
    device: str = "cpu"
    memory_getter: Callable[[], Any] | None = None

    def run(
        self,
        batches: Iterable[Any],
        *,
        warmup_steps: int = 1,
    ) -> Dict[str, Any]:
        self._prepare_model()
        memory = RuntimeMemoryTracker(accelerator_getter=self.memory_getter)
        memory.reset_peak_memory_stats()
        outputs: List[Any] = []
        latencies: List[float] = []
        requests = 0
        images = 0
        image_text_pairs = 0
        batch_list = list(batches)
        for batch in batch_list[: max(0, int(warmup_steps))]:
            self._predict(self._prepare_batch(batch))
        for batch in batch_list:
            prepared = self._prepare_batch(batch)
            start = time.perf_counter()
            output = self._predict(prepared)
            memory.synchronize_device()
            elapsed = max(time.perf_counter() - start, 1e-9)
            outputs.append(output)
            latencies.append(elapsed * 1000.0)
            requests += 1
            images += self._count(prepared, "num_images")
            image_text_pairs += self._count(prepared, "num_pairs")
        metrics: Dict[str, Any] = self._metrics(
            latencies,
            requests=requests,
            images=images,
            image_text_pairs=image_text_pairs,
        )
        memory.add_peak_memory_metrics(metrics)
        return {
            "task": self.task,
            "device": self.device,
            "outputs": outputs,
            "metrics": metrics,
        }

    def _prepare_model(self) -> None:
        to_device = getattr(self.model, "to", None)
        if callable(to_device) and self.device != "cpu":
            to_device(self.device)
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def _prepare_batch(self, batch: Any) -> Any:
        if self.device == "cpu":
            return batch
        return move_batch_to_device(batch, self.device)

    def _predict(self, batch: Any) -> Any:
        for method_name in ("predict", "detect", "embed", "generate"):
            method = getattr(self.model, method_name, None)
            if callable(method):
                return method(batch)
        if callable(self.model):
            return self.model(batch)
        raise RuntimeError(
            "Inference model must implement predict, detect, embed, generate, or __call__."
        )

    @staticmethod
    def _count(batch: Any, key: str) -> int:
        if isinstance(batch, dict) and key in batch:
            return int(batch.get(key) or 0)
        if isinstance(batch, list):
            return sum(InferenceRunner._count(item, key) for item in batch)
        return 0

    @staticmethod
    def _metrics(
        latencies: List[float],
        *,
        requests: int,
        images: int,
        image_text_pairs: int,
    ) -> Dict[str, Any]:
        elapsed_ms = sum(latencies)
        elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
        avg = elapsed_ms / max(1, len(latencies))
        return {
            "requests": int(requests),
            "images": int(images),
            "image_text_pairs": int(image_text_pairs),
            "latency_ms_avg": float(avg),
            "latency_ms_total": float(elapsed_ms),
            "requests_per_second": float(requests / elapsed_s),
            "images_per_second": float(images / elapsed_s),
            "image_text_pairs_per_second": float(image_text_pairs / elapsed_s),
        }


__all__ = ["InferenceRunner"]
