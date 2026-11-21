# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午4:11
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Generic inference runner with device placement and runtime metrics."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

from parascale.runtime.backends.devices import move_batch_to_device
from parascale.runtime.training.memory import RuntimeMemoryTracker

from .execution import inference_execution
from .tasks import InferenceTaskAdapter, default_inference_task_registry


@dataclass
class InferenceRunner:
    model: Any
    task: str
    device: str = "cpu"
    memory_getter: Callable[[], Any] | None = None
    task_adapter: InferenceTaskAdapter | None = None

    def __post_init__(self) -> None:
        if self.task_adapter is None:
            self.task_adapter = default_inference_task_registry().resolve(self.task)

    def run(
        self,
        batches: Iterable[Any],
        *,
        warmup_steps: int = 1,
        max_outputs: int | None = 32,
        on_output: Callable[[Any], Any] | None = None,
    ) -> Dict[str, Any]:
        """Predict batches with online metrics and bounded result retention.

        Keep the first ``max_outputs`` measured outputs (0 for none, explicit
        None for unbounded collection). A synchronous ``on_output`` consumes
        every measured output before the next batch; its return value is ignored.
        Timing covers prediction/postprocessing, excluding preparation and the
        callback. Invalid options, model/device and callback errors propagate.
        Retained outputs and callback arguments refer to the same objects.
        """
        if max_outputs is not None and (
            type(max_outputs) is not int or max_outputs < 0
        ):
            raise ValueError("max_outputs must be a nonnegative integer or None")
        if on_output is not None and not callable(on_output):
            raise TypeError("on_output must be callable")
        if inspect.iscoroutinefunction(on_output) or inspect.isasyncgenfunction(
            on_output
        ):
            raise TypeError("on_output must be synchronous")
        self._prepare_model()
        memory = RuntimeMemoryTracker(accelerator_getter=self.memory_getter)
        memory.reset_peak_memory_stats()
        outputs: List[Any] = []
        elapsed_ms_total = 0.0
        requests = 0
        images = 0
        image_text_pairs = 0
        tokens = 0
        requested_warmup_steps = max(0, int(warmup_steps))
        warmup_limit = self._effective_warmup_steps(
            batches,
            warmup_steps=requested_warmup_steps,
        )
        effective_warmup_steps = 0
        iterator = iter(batches)
        for _ in range(warmup_limit):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            self._predict(self._prepare_batch(batch))
            effective_warmup_steps += 1
        if effective_warmup_steps:
            memory.synchronize_device()
        for batch in iterator:
            prepared = self._prepare_batch(batch)
            # Exclude asynchronous warmup and batch transfers from prediction time.
            memory.synchronize_device()
            start = time.perf_counter()
            output = self._predict(prepared)
            memory.synchronize_device()
            elapsed = max(time.perf_counter() - start, 1e-9)
            counts = self.task_adapter.metric_counts(prepared)
            images += int(counts.get("images", 0))
            image_text_pairs += int(counts.get("image_text_pairs", 0))
            tokens += int(counts.get("tokens", 0))
            if on_output is not None:
                returned = on_output(output)
                if inspect.isawaitable(returned) or inspect.isasyncgen(returned):
                    if inspect.iscoroutine(returned):
                        returned.close()
                    raise TypeError("on_output must be synchronous")
                del returned
            if max_outputs is None or len(outputs) < max_outputs:
                outputs.append(output)
            elapsed_ms_total += elapsed * 1000.0
            requests += 1
        metrics: Dict[str, Any] = self._metrics(
            elapsed_ms_total,
            requests=requests,
            images=images,
            image_text_pairs=image_text_pairs,
            tokens=tokens,
        )
        memory.add_peak_memory_metrics(metrics)
        measurement_window = {
            "warmup_steps_requested": requested_warmup_steps,
            "warmup_steps_effective": effective_warmup_steps,
            "measured_batches": requests,
            "warmup_excluded_from_metrics": effective_warmup_steps > 0,
        }
        return {
            "task": self.task,
            "device": self.device,
            "outputs": outputs,
            "output_retention": {
                "max_outputs": max_outputs,
                "retained": len(outputs),
                "dropped": requests - len(outputs),
                "callback_enabled": on_output is not None,
            },
            "metrics": metrics,
            "measurement_window": measurement_window,
        }

    @staticmethod
    def _effective_warmup_steps(batches: Iterable[Any], *, warmup_steps: int) -> int:
        requested = max(0, int(warmup_steps))
        if requested == 0:
            return 0
        try:
            available = len(batches)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return requested
        if int(available) <= 0:
            return 0
        return min(requested, max(0, int(available) - 1))

    def _prepare_model(self) -> None:
        to_device = getattr(self.model, "to", None)
        if callable(to_device) and self.device != "cpu":
            to_device(self.device)
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def _prepare_batch(self, batch: Any) -> Any:
        batch = self.task_adapter.prepare_batch(batch)
        if self.device == "cpu":
            return batch
        return move_batch_to_device(batch, self.device)

    def _predict(self, batch: Any) -> Any:
        with inference_execution(self.model):
            output = self.task_adapter.predict(self.model, batch)
            return self.task_adapter.postprocess(output)

    @staticmethod
    def _metrics(
        elapsed_ms: float,
        *,
        requests: int,
        images: int,
        image_text_pairs: int,
        tokens: int,
    ) -> Dict[str, Any]:
        elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
        avg = elapsed_ms / max(1, requests)
        return {
            "requests": int(requests),
            "images": int(images),
            "image_text_pairs": int(image_text_pairs),
            "tokens": int(tokens),
            "latency_ms_avg": float(avg),
            "latency_ms_total": float(elapsed_ms),
            "requests_per_second": float(requests / elapsed_s),
            "images_per_second": float(images / elapsed_s),
            "image_text_pairs_per_second": float(image_text_pairs / elapsed_s),
            "tokens_per_second": float(tokens / elapsed_s),
        }


__all__ = ["InferenceRunner"]
