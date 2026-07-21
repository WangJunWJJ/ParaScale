# -*- coding: utf-8 -*-
# @Time : 2026/6/16 下午4:19
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest


def test_runtime_core_precision_memory_and_step_controllers_without_torch():
    from parascale.runtime.training import (
        PrecisionController,
        RuntimeMemoryTracker,
        StepRunner,
    )
    from parascale.runtime.training.metrics import RuntimeMetrics

    class FakeCuda:
        def __init__(self):
            self.reset_called = False
            self.sync_called = False

        def reset_peak_memory_stats(self):
            self.reset_called = True

        def max_memory_allocated(self):
            return 123

        def memory_allocated(self):
            return 45

        def synchronize(self):
            self.sync_called = True

    fake_cuda = FakeCuda()
    memory = RuntimeMemoryTracker(cuda_getter=lambda: fake_cuda)
    metrics = {}

    memory.reset_peak_memory_stats()
    memory.add_peak_memory_metrics(metrics)
    memory.synchronize_device()

    assert fake_cuda.reset_called is True
    assert fake_cuda.sync_called is True
    assert metrics["peak_memory_bytes"] == 123
    assert metrics["allocated_memory_bytes"] == 45

    class FakeEngine:
        amp_scaler = None
        training_backend = None
        config = type(
            "Config", (), {"precision": "fp32", "training_backend": "native"}
        )()
        memory = RuntimeMemoryTracker(cuda_getter=lambda: None)

        def __init__(self):
            self.backward_calls = 0
            self.step_calls = 0
            self.scheduler_calls = 0
            self.precision = PrecisionController(self)

        def _backend_name(self):
            return "native"

        def backward(self, _loss):
            self.backward_calls += 1

        def step(self, _optimizer):
            self.step_calls += 1

        def step_scheduler(self, _scheduler):
            self.scheduler_calls += 1

        def _with_throughput_metrics(self, metrics, batch, elapsed_seconds):
            return RuntimeMetrics(world_size=1).with_throughput_metrics(
                metrics,
                batch,
                elapsed_seconds,
            )

    class FakeLoss:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    engine = FakeEngine()
    engine.precision.setup_scaler()
    result = StepRunner(engine).run_step(
        {"num_images": 2, "x": 3},
        model=lambda **batch: batch["x"] * 2,
        optimizer=object(),
        scheduler=object(),
        loss_fn=lambda output, _batch: FakeLoss(output + 1),
    )

    assert engine.amp_scaler is None
    assert engine.backward_calls == 1
    assert engine.step_calls == 1
    assert engine.scheduler_calls == 1
    assert result["loss"] == 7
    assert result["images"] == 2
    assert result["images_per_second"] > 0

def test_fit_loop_runner_records_wait_memory_and_checkpoint_without_torch():
    from parascale.runtime.training import FitLoopRunner

    class State:
        initialized = True
        global_step = 0
        last_metrics = {}
        metrics_history = []

    class FakeMemory:
        def __init__(self):
            self.reset_called = False

        def reset_peak_memory_stats(self):
            self.reset_called = True

        def add_peak_memory_metrics(self, metrics):
            metrics["peak_memory_bytes"] = 123
            metrics["allocated_memory_bytes"] = 45

    class FakePrecision:
        def __init__(self):
            self.setup_called = False

        def setup_scaler(self):
            self.setup_called = True

    class FakeEngine:
        def __init__(self):
            self.config = type(
                "Config",
                (),
                {"training_backend": "native", "gradient_accumulation_steps": 1},
            )()
            self.state = State()
            self.memory = FakeMemory()
            self.precision = FakePrecision()
            self.saved_steps = []
            self.seen = []

        def _maybe_cuda_prefetch_iterator(self, iterator):
            return iterator

        def _gradient_accumulation_steps(self):
            return 1

        def _backend_name(self):
            return "native"

        def _run_step(self, batch, **_kwargs):
            self.seen.append(batch["x"])
            return {"loss": float(batch["x"])}

        def _add_end_to_end_metrics(self, metrics, batch, dataloader_wait_seconds):
            metrics["dataloader_wait_ms"] = dataloader_wait_seconds * 1000.0
            metrics["num_images"] = batch.get("num_images", 0)

        def record_step(self, metrics):
            self.state.global_step += 1
            self.state.last_metrics = dict(metrics)
            self.state.metrics_history.append(dict(metrics))
            return self.state

        def save_checkpoint(self, checkpoint_manager, scheduler=None):
            self.saved_steps.append((self.state.global_step, scheduler))
            checkpoint_manager.append(self.state.global_step)

    checkpoints = []
    scheduler = object()
    engine = FakeEngine()

    state = FitLoopRunner(engine).run(
        [{"x": 1, "num_images": 2}, {"x": 2, "num_images": 3}],
        max_steps=2,
        scheduler=scheduler,
        checkpoint_manager=checkpoints,
        checkpoint_interval=1,
    )

    assert state.global_step == 2
    assert engine.memory.reset_called is True
    assert engine.precision.setup_called is True
    assert engine.seen == [1, 2]
    assert checkpoints == [1, 2]
    assert engine.saved_steps == [(1, scheduler), (2, scheduler)]
    assert state.last_metrics["loss"] == 2.0
    assert state.last_metrics["peak_memory_bytes"] == 123
    assert state.last_metrics["allocated_memory_bytes"] == 45
    assert state.last_metrics["dataloader_wait_ms"] >= 0.0

def test_fit_loop_rejects_finite_dataloader_shorter_than_training_window():
    from parascale.runtime.training import FitLoopRunner

    class FakeEngine:
        def _gradient_accumulation_steps(self):
            return 2

        def _backend_name(self):
            return "native_ddp"

    with pytest.raises(ValueError, match="requires 6 micro-batches.*provides 2"):
        FitLoopRunner(FakeEngine()).run([{"x": 1}, {"x": 2}], max_steps=3)

def test_accumulation_controller_merges_micro_batches_without_torch():
    from parascale.runtime.training.accumulation import AccumulationController
    from parascale.runtime.training.metrics import RuntimeMetrics

    class FakeLoss:
        def __init__(self, value):
            self.value = float(value)

        def __truediv__(self, other):
            return FakeLoss(self.value / float(other))

        def __add__(self, other):
            return FakeLoss(self.value + float(other))

        def __radd__(self, other):
            return FakeLoss(float(other) + self.value)

        def __float__(self):
            return self.value

        def item(self):
            return self.value

    class FakeBackend:
        def no_sync(self):
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, *_args):
                    return False

            return Context()

    class FakeEngine:
        training_backend = FakeBackend()

        def __init__(self):
            self.backward_calls = 0
            self.step_calls = 0
            from parascale.runtime.training import PrecisionController

            self.config = type(
                "Config", (), {"precision": "fp32", "training_backend": "native"}
            )()
            self.memory = type(
                "Memory",
                (),
                {
                    "synchronize_device": lambda _self: None,
                    "elapsed_since": lambda _self, _start, synchronized=False: 0.5,
                },
            )()
            self.precision = PrecisionController(self)

        def _gradient_accumulation_steps(self):
            return 2

        def _backend_name(self):
            return "native"

        def backward(self, _loss):
            self.backward_calls += 1

        def step(self, _optimizer):
            self.step_calls += 1

        def step_scheduler(self, _scheduler):
            return None

        def _with_throughput_metrics(self, metrics, batch, elapsed_seconds):
            return RuntimeMetrics(world_size=1).with_throughput_metrics(
                metrics,
                batch,
                elapsed_seconds,
            )

    engine = FakeEngine()
    first = {"num_images": 2, "loss": 4.0}
    second = {"num_images": 3, "loss": 8.0}

    metrics = AccumulationController(engine).run(
        first,
        iter([second]),
        model=lambda **batch: batch["loss"],
        optimizer=object(),
        loss_fn=lambda output, _batch: FakeLoss(output),
        dataloader_wait_seconds=0.01,
    )

    assert engine.backward_calls == 2
    assert engine.step_calls == 1
    assert metrics["loss"] == 6.0
    assert metrics["gradient_accumulation_steps"] == 2
    assert metrics["images"] == 5
    assert metrics["images_per_second"] == 10.0
    assert metrics["dataloader_wait_ms"] >= 10.0

def test_accumulation_controller_rejects_incomplete_micro_batch_window():
    from parascale.runtime.training.accumulation import AccumulationController

    class FakeEngine:
        memory = type("Memory", (), {"synchronize_device": lambda _self: None})()

        def _gradient_accumulation_steps(self):
            return 2

    with pytest.raises(RuntimeError, match="requires 2 micro-batches.*received 1"):
        AccumulationController(FakeEngine()).run(
            {"x": 1},
            iter([]),
            model=lambda **batch: batch["x"],
            optimizer=object(),
            loss_fn=lambda output, _batch: output,
        )

def test_accumulation_controller_prepares_each_micro_batch_without_torch():
    from parascale.runtime.training.accumulation import AccumulationController
    from parascale.runtime.training.metrics import RuntimeMetrics

    class FakeLoss:
        def __init__(self, value):
            self.value = float(value)

        def __truediv__(self, other):
            return FakeLoss(self.value / float(other))

        def __add__(self, other):
            return FakeLoss(self.value + float(other))

        def __radd__(self, other):
            return FakeLoss(float(other) + self.value)

        def __float__(self):
            return self.value

        def item(self):
            return self.value

    class FakeBackend:
        def __init__(self):
            self.prepared = []

        def prepare_batch(self, batch):
            prepared = dict(batch)
            prepared["prepared"] = True
            self.prepared.append(prepared["id"])
            return prepared

        def no_sync(self):
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, *_args):
                    return False

            return Context()

    class FakeEngine:
        def __init__(self):
            from parascale.runtime.training import PrecisionController

            self.training_backend = FakeBackend()
            self.seen_prepared = []
            self.config = type(
                "Config", (), {"precision": "fp32", "training_backend": "native"}
            )()
            self.memory = type(
                "Memory",
                (),
                {
                    "synchronize_device": lambda _self: None,
                    "elapsed_since": lambda _self, _start, synchronized=False: 0.5,
                },
            )()
            self.precision = PrecisionController(self)

        def _gradient_accumulation_steps(self):
            return 2

        def _backend_name(self):
            return "native"

        def backward(self, _loss):
            return None

        def step(self, _optimizer):
            return None

        def step_scheduler(self, _scheduler):
            return None

        def _with_throughput_metrics(self, metrics, batch, elapsed_seconds):
            return RuntimeMetrics(world_size=1).with_throughput_metrics(
                metrics,
                batch,
                elapsed_seconds,
            )

    engine = FakeEngine()

    def model(**batch):
        engine.seen_prepared.append(bool(batch.get("prepared")))
        return batch["loss"]

    AccumulationController(engine).run(
        {"id": "a", "num_images": 1, "loss": 2.0},
        iter([{"id": "b", "num_images": 1, "loss": 4.0}]),
        model=model,
        optimizer=object(),
        loss_fn=lambda output, _batch: FakeLoss(output),
    )

    assert engine.training_backend.prepared == ["a", "b"]
    assert engine.seen_prepared == [True, True]

def test_accumulation_controller_rejects_step_fn_without_accumulation_protocol():
    from parascale.runtime.training.accumulation import AccumulationController

    class FakeEngine:
        def _gradient_accumulation_steps(self):
            return 2

    try:
        AccumulationController(FakeEngine()).run(
            {"x": 1},
            iter([{"x": 2}]),
            step_fn=lambda batch: {"loss": batch["x"]},
        )
    except RuntimeError as exc:
        assert "gradient accumulation" in str(exc)
        assert "step_fn" in str(exc)
    else:
        raise AssertionError("step_fn with accumulation must fail fast")
