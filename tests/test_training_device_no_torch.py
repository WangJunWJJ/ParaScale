# -*- coding: utf-8 -*-
# @Time : 2026/6/16 下午4:19
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com



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


def test_native_ddp_backend_builds_bucket_cap_kwargs_without_torch():
    from parascale.config import ParaScaleConfig
    from parascale.runtime.backends.native import NativeDdpTrainingBackend

    config = ParaScaleConfig(
        training_backend="native_ddp",
        ddp_bucket_cap_mb=100,
        ddp_gradient_as_bucket_view=True,
        ddp_static_graph=True,
    )
    backend = NativeDdpTrainingBackend(config=config)

    kwargs = backend._ddp_common_kwargs()

    assert kwargs["bucket_cap_mb"] == 100
    assert kwargs["gradient_as_bucket_view"] is True
    assert kwargs["static_graph"] is True

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
    backend = AscendNativeTrainingBackend(local_rank=1)

    batch = backend.prepare_batch({"input_ids": tensor, "labels": [FakeTensor()]})

    assert batch["input_ids"].moved_to == "npu:1"
    assert batch["input_ids"].non_blocking is True
    assert batch["labels"][0].moved_to == "npu:1"

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
