# -*- coding: utf-8 -*-
# @Time : 2026/6/16 下午4:19
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest


def test_cuda_stream_prefetcher_moves_nested_batches_without_torch():
    from parascale.runtime.training.prefetch import (
        CudaStreamPrefetchIterator,
        maybe_device_prefetch_iterator,
    )

    class FakeTensor:
        def __init__(self):
            self.device = type("Device", (), {"type": "cpu"})()
            self.moved_to = None

        def to(self, device, non_blocking=False):
            self.moved_to = str(device)
            self.non_blocking = non_blocking
            self.device = type("Device", (), {"type": "cuda"})()
            return self

    class FakeStream:
        def wait_stream(self, _stream):
            return None

    class FakeCuda:
        def Stream(self, device=None):
            return FakeStream()

        def current_stream(self, device=None):
            return FakeStream()

        def stream(self, _stream):
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, *_args):
                    return False

            return Context()

    class FakeTorch:
        cuda = FakeCuda()

    tensor = FakeTensor()
    iterator = CudaStreamPrefetchIterator(
        FakeTorch(),
        iter([{"x": tensor, "pipeline_profile": {}}]),
        "cuda:0",
    )

    batch = next(iterator)

    assert batch["x"].moved_to == "cuda:0"
    assert batch["x"].non_blocking is True
    assert batch["pipeline_profile"]["cuda_prefetch_h2d_ms"] >= 0.0
    assert batch["pipeline_profile"]["cuda_prefetch_wait_ms"] >= 0.0

    config = type("Config", (), {"device_prefetch": False, "cuda_prefetch": False})()
    source = iter([{"x": 1}])
    assert maybe_device_prefetch_iterator(source, config=config, local_rank=0) is source


def test_device_prefetch_uses_unified_batch_mover_without_torch():
    from pathlib import Path

    source = Path("parascale/runtime/training/prefetch.py").read_text(encoding="utf-8")

    assert "move_batch_to_device" in source
    assert "value.to(self.device, non_blocking=True)" not in source


def test_native_backend_places_model_on_resolved_accelerator_without_torch(monkeypatch):
    from parascale.runtime.backends.native import NativeTrainingBackend

    class FakeCuda:
        def is_available(self):
            return True

        def set_device(self, _device):
            return None

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(value):
            return value

    class FakeModel:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = str(device)
            return self

    monkeypatch.setattr(
        "parascale.runtime.backends.native._require_torch",
        lambda: FakeTorch(),
    )

    model = FakeModel()
    placed = NativeTrainingBackend(local_rank=1).setup_model(model)

    assert placed is model
    assert model.moved_to == "cuda:1"


def test_accumulated_pipeline_cache_hit_is_normalized_without_torch():
    from parascale.runtime.training.metrics import merge_pipeline_profiles

    profiles = [
        {"cache_hit": 1.0, "prompt_template_ms": 0.1},
        {"cache_hit": 1.0, "prompt_template_ms": 0.2},
        {"cache_hit": 1.0, "prompt_template_ms": 0.3},
        {"cache_hit": 1.0, "prompt_template_ms": 0.4},
    ]

    merged = merge_pipeline_profiles(profiles)

    assert merged["cache_hit"] == 1.0
    assert merged["cache_hit_count"] == 4.0
    assert merged["cache_sample_count"] == 4.0
    assert merged["prompt_template_ms"] == 1.0


def test_accumulated_pipeline_cache_hit_supports_partial_hits_without_torch():
    from parascale.runtime.training.metrics import merge_pipeline_profiles

    profiles = [
        {"cache_hit": 1.0},
        {"cache_hit": 0.0},
        {"cache_hit": 0.5},
        {"cache_hit": 1.0},
    ]

    merged = merge_pipeline_profiles(profiles)

    assert merged["cache_hit"] == 0.625
    assert merged["cache_hit_count"] == 2.5
    assert merged["cache_sample_count"] == 4.0


def test_multinode_capability_level_is_marked_as_smoke_without_torch():
    from parascale.runtime.orchestrator import _capability_level_for_scope

    config_data = {
        "hardware_profile": {
            "world_size": 2,
            "gpus_per_node": 1,
            "num_nodes": 2,
        }
    }

    capability = _capability_level_for_scope(
        "local_native_clip_contrastive_datacomp_wds", config_data
    )

    assert capability == "multi_node_smoke"


def test_ascend_backend_prepares_nested_batch_for_npu_without_torch():
    from parascale.runtime.backends.ascend_native import AscendNativeTrainingBackend

    class FakeTensor:
        def __init__(self):
            self.moved_to = None

        def to(self, device, non_blocking=False):
            self.moved_to = str(device)
            self.non_blocking = non_blocking
            return self

    tensor = FakeTensor()
    backend = AscendNativeTrainingBackend(local_rank=3)

    batch = backend.prepare_batch({"input_ids": tensor, "labels": [FakeTensor()]})

    assert batch["input_ids"].moved_to == "npu:3"
    assert batch["input_ids"].non_blocking is True
    assert batch["labels"][0].moved_to == "npu:3"


def test_training_backends_share_accelerator_batch_placement_without_torch(monkeypatch):
    from parascale.runtime.backends import (
        DeepSpeedTrainingBackend,
        FSDPTrainingBackend,
        NativeTrainingBackend,
    )

    class FakeCuda:
        def is_available(self):
            return True

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(value):
            return str(value)

    class FakeTensor:
        def __init__(self):
            self.moved_to = None

        def to(self, device, non_blocking=False):
            self.moved_to = str(device)
            self.non_blocking = non_blocking
            return self

    monkeypatch.setattr(
        "parascale.runtime.backends.base._require_torch",
        lambda: FakeTorch(),
    )

    for backend_cls in (
        NativeTrainingBackend,
        FSDPTrainingBackend,
        DeepSpeedTrainingBackend,
    ):
        tensor = FakeTensor()
        batch = backend_cls(local_rank=2).prepare_batch({"pixel_values": tensor})

        assert batch["pixel_values"].moved_to == "cuda:2"
        assert batch["pixel_values"].non_blocking is True


def test_ascend_visible_devices_use_logical_rank_without_torch(monkeypatch):
    from parascale.runtime.backends.devices import resolve_ascend_device_id

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,4,6")

    assert resolve_ascend_device_id(0) == 0
    assert resolve_ascend_device_id(2) == 2


def test_ascend_device_resolution_rejects_rank_beyond_visible_count():
    from parascale.runtime.backends.devices import resolve_ascend_device_id

    class FakeNpu:
        def device_count(self):
            return 2

    fake_torch = type("FakeTorch", (), {"npu": FakeNpu()})()

    assert resolve_ascend_device_id(1, fake_torch) == 1
    try:
        resolve_ascend_device_id(2, fake_torch)
    except RuntimeError as exc:
        assert "visible logical device count" in str(exc)
    else:
        raise AssertionError("expected rank beyond visible NPU count to fail")


def test_runtime_device_helpers_select_and_set_cuda_or_npu_without_torch():
    from parascale.runtime.backends.devices import (
        current_accelerator,
        resolve_torch_device,
        set_current_device,
    )

    class FakeCuda:
        def __init__(self, available):
            self.available = available
            self.set_to = None

        def is_available(self):
            return self.available

        def set_device(self, device_id):
            self.set_to = int(device_id)

    class FakeNpu:
        def __init__(self, available, count=2):
            self.available = available
            self.count = count
            self.set_to = None

        def is_available(self):
            return self.available

        def device_count(self):
            return self.count

        def set_device(self, device_id):
            self.set_to = int(device_id)

    class FakeTorch:
        def __init__(self, cuda_available=False, npu_available=False):
            self.cuda = FakeCuda(cuda_available)
            self.npu = FakeNpu(npu_available)

        def device(self, value):
            return str(value)

    cuda_torch = FakeTorch(cuda_available=True)
    npu_torch = FakeTorch(npu_available=True)

    assert current_accelerator(cuda_torch) == "cuda"
    assert resolve_torch_device(cuda_torch, local_rank=1) == "cuda:1"
    assert set_current_device(cuda_torch, local_rank=1) == "cuda:1"
    assert cuda_torch.cuda.set_to == 1

    assert current_accelerator(npu_torch) == "npu"
    assert resolve_torch_device(npu_torch, local_rank=1) == "npu:1"
    assert set_current_device(npu_torch, local_rank=1) == "npu:1"
    assert npu_torch.npu.set_to == 1


def test_memory_tracker_supports_npu_backend_without_torch():
    from parascale.runtime.training.memory import RuntimeMemoryTracker

    class FakeNpu:
        def __init__(self):
            self.reset_called = False
            self.sync_called = False

        def is_available(self):
            return True

        def reset_peak_memory_stats(self):
            self.reset_called = True

        def max_memory_allocated(self):
            return 456

        def memory_allocated(self):
            return 78

        def synchronize(self):
            self.sync_called = True

    fake_npu = FakeNpu()
    tracker = RuntimeMemoryTracker(accelerator_getter=lambda: fake_npu)
    metrics = {}

    tracker.reset_peak_memory_stats()
    tracker.add_peak_memory_metrics(metrics)
    tracker.synchronize_device()

    assert fake_npu.reset_called is True
    assert fake_npu.sync_called is True
    assert metrics["peak_memory_bytes"] == 456
    assert metrics["allocated_memory_bytes"] == 78


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


def test_train_engine_exposes_checkpoint_collective_rank_and_barrier():
    from parascale.core import MockCollectiveBackend
    from parascale.runtime.training import TrainEngine

    collective = MockCollectiveBackend(initialized=True, world_size=2, rank=1)
    engine = TrainEngine(config=object(), collective=collective)

    assert engine._distributed_rank() == 1
    assert engine._distributed_world_size() == 2

    engine._distributed_barrier()

    assert collective.history[-1]["op"] == "barrier"


def test_nonzero_rank_checkpoint_result_skips_manifest_validation():
    from parascale.runtime.orchestrator import _validate_final_checkpoint_result

    class Manager:
        def read_manifest_path(self, _path):
            raise AssertionError("nonzero rank must not read the manifest")

    path, validation = _validate_final_checkpoint_result(
        Manager(),
        {
            "step": 2,
            "rank": 1,
            "skipped": True,
            "reason": "checkpoint manifest is written by rank 0 only",
        },
    )

    assert path is None
    assert validation["ok"] is True
    assert validation["skipped"] is True
    assert validation["rank"] == 1
