# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Stream ordering and allocator ownership without an accelerator dependency."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from parascale.runtime.training.prefetch import CudaStreamPrefetchIterator


class Stream:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_stream(self, other):
        self.events.append(("wait", self.name, other.name))


class Tensor:
    def __init__(self, name, events, device="cpu"):
        self.name = name
        self.events = events
        self.device = device
        self.is_cuda = device.startswith("cuda")

    def to(self, device, non_blocking=False):
        assert non_blocking
        self.events.append(("move", self.name, device))
        return Tensor(self.name + "_moved", self.events, device)

    def record_stream(self, stream):
        assert self.is_cuda
        self.events.append(("record", self.name, stream.name))


def runtime(events):
    copy = Stream("copy", events)
    current = {
        "cuda:0": Stream("producer0", events),
        "cuda:1": Stream("producer1", events),
    }

    @contextmanager
    def context(stream):
        events.append(("enter", stream.name))
        previous = current["cuda:0"]
        current["cuda:0"] = stream
        try:
            yield
        finally:
            current["cuda:0"] = previous
            events.append(("exit", stream.name))

    cuda = SimpleNamespace(
        Stream=lambda **kwargs: copy,
        current_stream=lambda device: current[str(device)],
        stream=context,
    )
    return SimpleNamespace(cuda=cuda), current


def test_cpu_source_records_nested_device_outputs_before_next_preload():
    events = []
    torch, current = runtime(events)
    tensors = [Tensor(str(index), events) for index in range(3)]

    def batches():
        yield {"x": tensors[0], "nested": [tensors[1], (tensors[2], 7)]}
        events.append(("next_source",))
        yield {"x": Tensor("next", events)}

    iterator = CudaStreamPrefetchIterator(torch, batches(), "cuda:0")
    assert not any(event[0] in {"wait", "record"} for event in events)
    current["cuda:0"] = Stream("consumer", events)
    result = next(iterator)
    wait = events.index(("wait", "consumer", "copy"))
    for index in range(3):
        recorded = events.index(("record", str(index) + "_moved", "consumer"))
        assert wait < recorded < events.index(("next_source",))
    assert result["nested"][1][1] == 7


@pytest.mark.parametrize("source_device", ["cuda:0", "cuda:1"])
def test_cuda_source_waits_and_records_before_transfer(source_device):
    events = []
    torch, _ = runtime(events)
    value = Tensor("input", events, source_device)
    iterator = CudaStreamPrefetchIterator(torch, iter([{"x": value}]), "cuda:0")
    expected_producer = "producer" + source_device[-1]
    wait = events.index(("wait", "copy", expected_producer))
    record = events.index(("record", "input", "copy"))
    enter = events.index(("enter", "copy"))
    assert wait < record < enter
    next(iterator)
    assert (
        "record",
        "input" if source_device == "cuda:0" else "input_moved",
        "producer0",
    ) in events


def test_source_ownership_is_recorded_even_when_transfer_raises():
    events = []
    torch, _ = runtime(events)

    class BrokenTensor(Tensor):
        def to(self, *args, **kwargs):
            raise RuntimeError("transfer failed")

    source = BrokenTensor("input", events, "cuda:1")
    with pytest.raises(RuntimeError, match="transfer failed"):
        CudaStreamPrefetchIterator(torch, iter([source]), "cuda:0")
    assert ("record", "input", "copy") in events


def test_aliased_cuda_tensor_is_recorded_once_per_stream():
    events = []
    torch, _ = runtime(events)
    value = Tensor("input", events, "cuda:0")
    iterator = CudaStreamPrefetchIterator(
        torch, iter([{"a": value, "b": [value, (value,)]}]), "cuda:0"
    )
    next(iterator)
    assert events.count(("record", "input", "copy")) == 1
    assert events.count(("record", "input", "producer0")) == 1
