# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded stage updates with committed data progress and explicit timing scopes."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.distributed as dist

from .coordination import coordinated_call


class StageUpdateLoop:
    """Orchestrate an executor; checkpoint only fully committed updates."""

    def __init__(
        self,
        trainer: Any,
        optimizer: Any,
        progress: Any,
        cursor: Any,
        plan: Any,
        device: Any,
        save_checkpoint: Any,
    ) -> None:
        self.trainer, self.optimizer, self.progress = trainer, optimizer, progress
        self.cursor, self.plan, self.device = cursor, plan, device
        self.save_checkpoint = save_checkpoint

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _window(self, iterator: Any) -> tuple[Any, Any]:
        batches = []
        for _ in range(self.plan.topology.microbatches):
            try:
                batches.append(next(iterator))
            except StopIteration as exc:
                raise RuntimeError(
                    "Token corpus exhausted before a complete update"
                ) from exc
        return (
            torch.cat([batch["input_ids"] for batch in batches]),
            torch.cat([batch["labels"] for batch in batches]),
        )

    def _finite_gradients(self) -> None:
        flags = []
        for parameter in self.trainer.module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if hasattr(gradient, "to_local"):
                gradient = gradient.to_local()
            flags.append(torch.isfinite(gradient).all())
        if not flags or not torch.stack(flags).all().item():
            raise FloatingPointError("Missing or non-finite stage gradient")

    def _update(self) -> None:
        self.optimizer.step()
        if self.progress.scheduler is not None:
            self.progress.scheduler.step()
        self._sync()

    def _max_times(self, values: list[float]) -> list[float]:
        if not dist.is_available() or not dist.is_initialized():
            return values
        timings = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        return timings.cpu().tolist()

    def run(self) -> dict[str, Any]:
        iterator = coordinated_call(
            lambda: self.progress.data_resume.prepare_iterator(self.cursor),
            phase="data iterator",
        )
        state = self.progress.state
        initial_step = state.global_step
        totals = dict(samples=0, tokens=0, scored_tokens=0)
        timings = dict(
            data_wait_seconds=0.0,
            transfer_seconds=0.0,
            training_seconds=0.0,
            checkpoint_seconds=0.0,
        )
        self._sync()
        started = time.perf_counter()
        for update in range(self.plan.max_steps):
            begin = time.perf_counter()
            inputs, labels = coordinated_call(
                lambda: self._window(iterator), phase="data window"
            )
            data_end = time.perf_counter()

            def transfer():
                result = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                self._sync()
                return result

            inputs, labels = coordinated_call(transfer, phase="batch transfer")
            transfer_end = time.perf_counter()
            result = self.trainer.forward_backward(inputs, labels, return_outputs=False)

            def validate():
                if not math.isfinite(result.loss):
                    raise FloatingPointError("Non-finite stage loss")
                self._finite_gradients()

            coordinated_call(validate, phase="gradient validation")
            coordinated_call(self._update, phase="optimizer update")
            training_end = time.perf_counter()
            # A failure after mutation is terminal; published progress stays at the
            # last successful update, recovered from its immutable checkpoint.
            local_samples = inputs.shape[0]

            def commit():
                self.progress.data_resume.record_consumption(
                    {
                        "gradient_accumulation_steps": self.plan.topology.microbatches,
                        "batch_size": local_samples,
                    }
                )
                state.global_step += 1

            coordinated_call(commit, phase="consumption commit")
            counts = {
                "samples": local_samples * self.plan.topology.dp_size,
                "tokens": inputs.numel() * self.plan.topology.dp_size,
                "scored_tokens": result.normalization_count,
            }
            checkpoint_begin = time.perf_counter()
            if (
                state.global_step % self.plan.checkpoint_interval == 0
                or update + 1 == self.plan.max_steps
            ):
                self.save_checkpoint(state.global_step)
            checkpoint_end = time.perf_counter()
            durations = self._max_times(
                [
                    data_end - begin,
                    transfer_end - data_end,
                    training_end - transfer_end,
                    checkpoint_end - checkpoint_begin,
                ]
            )
            metrics = {
                "global_step": state.global_step,
                "loss": result.loss,
                **counts,
                **dict(zip(timings, durations)),
            }
            state.last_metrics = metrics
            if self.plan.metrics_history_limit:
                state.metrics_history.append(metrics)
                if len(state.metrics_history) > self.plan.metrics_history_limit:
                    del state.metrics_history[0]
                    state.metrics_history_dropped += 1
            else:
                state.metrics_history_dropped += 1
            for name in totals:
                totals[name] += counts[name]
            for name, duration in zip(timings, durations):
                timings[name] += duration
        self._sync()
        elapsed = self._max_times([time.perf_counter() - started])[0]
        return {
            "initial_step": initial_step,
            "global_step": state.global_step,
            "completed_updates": state.global_step - initial_step,
            **totals,
            **timings,
            "elapsed_seconds": elapsed,
            "scored_tokens_per_second": totals["scored_tokens"] / elapsed,
            "timing_scope": "update loop including data, coordination and checkpoints; excludes setup",
            "metrics_history": state.metrics_history,
            "metrics_history_dropped": state.metrics_history_dropped,
        }
