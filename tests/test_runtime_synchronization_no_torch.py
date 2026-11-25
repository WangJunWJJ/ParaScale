# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午1:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Device completion failures and measured-window boundaries."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from parascale.runtime.inference.runner import InferenceRunner
from parascale.runtime.training.accumulation import AccumulationController
from parascale.runtime.training.fit_loop import FitLoopRunner
from parascale.runtime.training.memory import RuntimeMemoryTracker
from parascale.runtime.training.step import StepRunner


class Accelerator:
    def __init__(self, fail_on=0):
        self.calls = 0
        self.fail_on = fail_on

    def synchronize(self):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("asynchronous device failure")

    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return 0

    def memory_allocated(self):
        return 0


@pytest.mark.parametrize("operation", ["synchronize", "elapsed"])
def test_memory_tracker_propagates_device_execution_failure(operation):
    accelerator = Accelerator(fail_on=1)
    tracker = RuntimeMemoryTracker(accelerator_getter=lambda: accelerator)
    with pytest.raises(RuntimeError, match="asynchronous device failure"):
        if operation == "synchronize":
            tracker.synchronize_device()
        else:
            tracker.elapsed_since(0.0)


@pytest.mark.parametrize("fail_on", [1, 2, 3])
def test_inference_does_not_return_success_after_device_failure(fail_on):
    accelerator = Accelerator(fail_on=fail_on)
    model = SimpleNamespace(predict=lambda batch: batch)
    runner = InferenceRunner(
        model, "vision_detection", memory_getter=lambda: accelerator
    )
    with pytest.raises(RuntimeError, match="asynchronous device failure"):
        runner.run([{"num_images": 1}], warmup_steps=0)


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_training_failure_does_not_record_step_or_publish_checkpoint(
    accumulation_steps,
):
    accelerator = Accelerator(fail_on=2)
    updates, checkpoints = [], []
    engine = SimpleNamespace(
        state=SimpleNamespace(global_step=0),
        precision=SimpleNamespace(
            setup_scaler=lambda: None, autocast_context=nullcontext
        ),
        memory=RuntimeMemoryTracker(accelerator_getter=lambda: accelerator),
        training_backend=None,
        backward=lambda loss: None,
        step=lambda optimizer: updates.append(1),
        step_scheduler=lambda scheduler: None,
        _gradient_accumulation_steps=lambda: accumulation_steps,
        _maybe_cuda_prefetch_iterator=lambda iterator: iterator,
        _with_throughput_metrics=lambda metrics, batch, elapsed: metrics,
        _add_end_to_end_metrics=lambda *args: None,
        save_checkpoint=lambda *args, **kwargs: checkpoints.append(True),
    )

    def record_step(metrics):
        engine.state.global_step += 1

    engine.record_step = record_step
    engine._run_step = StepRunner(engine).run_step
    engine._run_accumulated_step = AccumulationController(engine).run
    execution = (
        {"step_fn": lambda batch: updates.append(batch)}
        if accumulation_steps == 1
        else {"model": lambda batch: batch, "loss_fn": lambda output, batch: output}
    )
    with pytest.raises(RuntimeError, match="asynchronous device failure"):
        FitLoopRunner(engine).run(
            [1] * accumulation_steps,
            max_steps=1,
            checkpoint_manager=object(),
            checkpoint_interval=1,
            **execution,
        )
    assert updates == [1]
    assert engine.state.global_step == 0
    assert not checkpoints


def test_inference_timing_excludes_pending_warmup_and_batch_preparation(monkeypatch):
    clock = SimpleNamespace(elapsed=0.0, pending=0.0)

    class QueuedAccelerator(Accelerator):
        def synchronize(self):
            clock.elapsed += clock.pending
            clock.pending = 0.0

    class Task:
        def prepare_batch(self, batch):
            clock.pending += 3.0
            return batch

        def predict(self, model, batch):
            clock.pending += 20.0 if batch == 0 else 2.0
            return batch

        def postprocess(self, output):
            return output

        def metric_counts(self, batch):
            return {"images": 1}

    monkeypatch.setattr(
        "parascale.runtime.inference.runner.time.perf_counter", lambda: clock.elapsed
    )

    initialized = []

    def prepare_model():
        if not initialized:
            clock.pending += 7.0
            initialized.append(True)

    runner = InferenceRunner(
        SimpleNamespace(eval=prepare_model),
        "queued",
        memory_getter=QueuedAccelerator,
        task_adapter=Task(),
    )
    result = runner.run([0, 1, 2], warmup_steps=1)
    assert result["outputs"] == [1, 2]
    assert result["metrics"]["latency_ms_total"] == 4000.0
    assert result["metrics"]["latency_ms_avg"] == 2000.0
    assert result["metrics"]["images_per_second"] == 0.5
    assert result["timing"]["setup_seconds"] == 7.0
    assert result["timing"]["warmup_seconds"] == 23.0


@pytest.mark.parametrize("available", [0, 1, 2])
def test_unsized_short_warmup_reports_actual_completed_batches(available):
    accelerator = Accelerator()
    runner = InferenceRunner(
        SimpleNamespace(predict=lambda batch: batch),
        "vision_detection",
        memory_getter=lambda: accelerator,
    )
    result = runner.run(iter(range(available)), warmup_steps=4)
    window = result["measurement_window"]
    assert window["warmup_steps_requested"] == 4
    assert window["warmup_steps_effective"] == available
    assert window["warmup_excluded_from_metrics"] is (available > 0)
    assert window["measured_batches"] == 0
    # Setup completes before its timer stops; only actual warmup adds a barrier.
    assert accelerator.calls == 1 + (1 if available else 0)


def test_warmup_only_device_failure_is_not_silently_accepted():
    accelerator = Accelerator(fail_on=2)
    predicted = []
    runner = InferenceRunner(
        SimpleNamespace(predict=lambda batch: predicted.append(batch) or batch),
        "vision_detection",
        memory_getter=lambda: accelerator,
    )
    with pytest.raises(RuntimeError, match="asynchronous device failure"):
        runner.run(iter([1]), warmup_steps=4)
    assert predicted == [1]
