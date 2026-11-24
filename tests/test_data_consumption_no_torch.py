# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Checkpoint cursors must describe consumed, not prefetched, batches."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from parascale.runtime.training.data_state import DataResumeController
from parascale.runtime.training.prefetch import CudaStreamPrefetchIterator


class StatefulLoader:
    def __init__(self):
        self.cursor = {"position": 0}

    def __iter__(self):
        while self.cursor["position"] < 12:
            value = self.cursor["position"]
            self.cursor["position"] += 1
            yield {"value": value}

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = dict(state["cursor"])


def controller():
    return DataResumeController(
        SimpleNamespace(
            config=SimpleNamespace(),
            state=SimpleNamespace(
                data_state={}, consumed_micro_batches=0, consumed_samples=0
            ),
        )
    )


def prefetch(source):
    stream = SimpleNamespace(wait_stream=lambda other: None)
    cuda = SimpleNamespace(
        Stream=lambda **kwargs: stream,
        current_stream=lambda *args: stream,
        stream=lambda value: nullcontext(),
    )
    return CudaStreamPrefetchIterator(SimpleNamespace(cuda=cuda), source, "cuda:0")


@pytest.mark.parametrize("accumulation", [1, 3])
@pytest.mark.parametrize("optimizer_updated", [False, True])
def test_prefetched_cursor_commits_only_consumed_batches(
    accumulation, optimizer_updated
):
    resume = controller()
    source = StatefulLoader()
    iterator = prefetch(resume.prepare_iterator(source))
    assert resume.capture()["state"]["cursor"]["position"] == 0
    assert [next(iterator)["value"] for _ in range(accumulation)] == list(
        range(accumulation)
    )
    resume.record_consumption(
        {
            "gradient_accumulation_steps": accumulation,
            "batch_size": accumulation,
            "optimizer_updated": optimizer_updated,
        }
    )
    saved = resume.capture()
    assert source.cursor["position"] == accumulation + 1
    assert saved["consumed_micro_batches"] == accumulation
    assert saved["state"]["cursor"]["position"] == accumulation
    restored = controller()
    restored.engine.state.data_state = saved
    assert next(restored.prepare_iterator(StatefulLoader()))["value"] == accumulation


def test_failed_step_does_not_commit_cursor_and_capture_does_not_alias_state():
    resume = controller()
    iterator = prefetch(resume.prepare_iterator(StatefulLoader()))
    next(iterator)
    resume.record_consumption({})
    saved = resume.capture()
    next(iterator)  # Simulate a step that fails before acknowledging consumption.
    assert resume.capture()["state"]["cursor"]["position"] == 1
    assert saved["state"]["cursor"]["position"] == 1
    saved["state"]["cursor"]["position"] = 99
    assert resume.capture()["state"]["cursor"]["position"] == 1


def test_consumption_cannot_exceed_issued_batches():
    resume = controller()
    iterator = resume.prepare_iterator(StatefulLoader())
    next(iterator)
    with pytest.raises(RuntimeError, match="consumption.*issued"):
        resume.record_consumption({"gradient_accumulation_steps": 2})
    assert resume.engine.state.consumed_micro_batches == 0
    assert resume.capture()["state"]["cursor"]["position"] == 0


def test_worker_prefetched_sampler_is_not_a_consumed_cursor():
    class WorkerLoader:
        num_workers = 2
        sampler = StatefulLoader()

        def __iter__(self):
            return iter(self.sampler)

    loader = WorkerLoader()
    with pytest.raises(ValueError, match="loader-level"):
        controller().prepare_iterator(loader)


def test_loader_level_state_protocol_can_account_for_workers():
    loader = StatefulLoader()
    loader.num_workers = 2
    resume = controller()
    iterator = prefetch(resume.prepare_iterator(loader))
    next(iterator)
    resume.record_consumption({})
    assert resume.capture()["state"]["cursor"]["position"] == 1


def test_end_of_source_does_not_advance_committed_state():
    resume = controller()
    iterator = prefetch(resume.prepare_iterator(StatefulLoader()))
    for expected in range(12):
        assert next(iterator)["value"] == expected
        resume.record_consumption({})
    with pytest.raises(StopIteration):
        next(iterator)
    assert resume.capture()["state"]["cursor"]["position"] == 12


def test_restarting_same_loader_rewinds_unconsumed_prefetch_and_invalidates_old_iterator():
    resume = controller()
    loader = StatefulLoader()
    old = prefetch(resume.prepare_iterator(loader))
    assert next(old)["value"] == 0
    resume.record_consumption({})
    fresh = resume.prepare_iterator(loader)
    with pytest.raises(RuntimeError, match="superseded"):
        next(old)
    assert next(fresh)["value"] == 1
    resume.record_consumption({})
    assert resume.capture()["state"]["cursor"]["position"] == 2


def test_initial_cursor_is_captured_before_iterator_can_prefetch():
    class EagerLoader(StatefulLoader):
        def __iter__(self):
            iterator = super().__iter__()
            first = next(iterator)
            return iter([first, *iterator])

    resume = controller()
    resume.prepare_iterator(EagerLoader())
    assert resume.capture()["state"]["cursor"]["position"] == 0


def test_loaded_cursor_is_preserved_before_next_fit():
    resume = controller()
    iterator = prefetch(resume.prepare_iterator(StatefulLoader()))
    next(iterator)
    resume.record_consumption({})
    saved = resume.capture()
    next(iterator)
    resume.restore_manifest(SimpleNamespace(data_state=saved, consumed_samples=1))
    assert resume.capture() == saved
    with pytest.raises(RuntimeError, match="superseded"):
        next(iterator)
