# -*- coding: utf-8 -*-
# @Time : 2026/6/23 上午9:11
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Outer fit-loop lifecycle for TrainEngine-like runtimes."""

from __future__ import annotations

import time
from typing import Any, Optional


class FitLoopRunner:
    """Own dataloader wait, memory metrics, record, and checkpoint cadence."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def run(
        self,
        dataloader: Any,
        *,
        max_steps: Optional[int] = None,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        loss_fn: Any = None,
        step_fn: Any = None,
        checkpoint_manager: Any = None,
        checkpoint_interval: Optional[int] = None,
    ) -> Any:
        max_steps = None if max_steps is None else max(0, int(max_steps))
        self._validate_dataloader_window(dataloader, max_steps=max_steps)
        self.engine.precision.setup_scaler()
        self.engine.memory.reset_peak_memory_stats()
        data_resume = getattr(self.engine, "data_resume", None)
        source_iterator = (
            data_resume.prepare_iterator(dataloader)
            if data_resume is not None
            else iter(dataloader)
        )
        iterator = self.engine._maybe_cuda_prefetch_iterator(source_iterator)
        index = 0
        while True:
            if max_steps is not None and index >= max_steps:
                break
            wait_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            dataloader_wait_seconds = time.perf_counter() - wait_start
            if (
                self.engine._gradient_accumulation_steps() > 1
                and self.engine._backend_name() != "deepspeed"
            ):
                metrics = self.engine._run_accumulated_step(
                    batch,
                    iterator,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    step_fn=step_fn,
                    dataloader_wait_seconds=dataloader_wait_seconds,
                )
                dataloader_wait_seconds = 0.0
            else:
                metrics = self.engine._run_step(
                    batch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    step_fn=step_fn,
                )
            self.engine._add_end_to_end_metrics(metrics, batch, dataloader_wait_seconds)
            self.engine.memory.add_peak_memory_metrics(metrics)
            if data_resume is not None:
                data_resume.record_consumption(metrics)
            self.engine.record_step(metrics)
            if checkpoint_manager is not None and checkpoint_interval:
                if self.engine.state.global_step % int(checkpoint_interval) == 0:
                    self.engine.save_checkpoint(
                        checkpoint_manager,
                        scheduler=scheduler,
                    )
                    barrier = getattr(self.engine, "_distributed_barrier", None)
                    if callable(barrier):
                        barrier()
            index += 1
        return self.engine.state

    def _validate_dataloader_window(
        self,
        dataloader: Any,
        *,
        max_steps: Optional[int],
    ) -> None:
        if max_steps is None or max_steps == 0:
            return
        try:
            available = len(dataloader)
        except (TypeError, AttributeError):
            return
        accumulation_steps = max(1, int(self.engine._gradient_accumulation_steps()))
        if self.engine._backend_name() == "deepspeed":
            accumulation_steps = 1
        required = int(max_steps) * accumulation_steps
        state = getattr(self.engine, "state", None)
        data_state = dict(getattr(state, "data_state", {}) or {})
        if str(data_state.get("resume_mode", "")) == "replay_skip":
            required += max(
                0,
                int(data_state.get("consumed_micro_batches", 0) or 0),
            )
        if int(available) < required:
            raise ValueError(
                "Training window requires "
                f"{required} micro-batches, but the finite dataloader provides "
                f"{available}. Increase the dataset size, enable a repeatable "
                "streaming source, or reduce max_steps."
            )


__all__ = ["FitLoopRunner"]
