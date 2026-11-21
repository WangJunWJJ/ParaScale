# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Auxiliary pipeline output ownership, independent of native communication."""

import copy
import weakref
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def schedule_fixture(monkeypatch, precision):
    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    from parascale.parallel.pipeline_stage import (
        OutputAwareGPipe,
        PrecisionPipelineStage,
    )

    stage = object.__new__(PrecisionPipelineStage)
    stage.submod = torch.nn.Linear(4, 4)
    stage.device = torch.device("cpu")
    stage.precision = precision
    stage.retain_outputs = False
    stage.output_chunks, stage.fwd_cache = [], {}
    references = []

    def forward(self, index, args, kwargs=None):
        output = self.forward_maybe_with_nosync(*args, **(kwargs or {}))
        self.output_chunks.append(output)
        self.fwd_cache[index] = output
        references.append(weakref.ref(output))
        return output

    monkeypatch.setattr(PipelineStage, "forward_one_chunk", forward)

    def run(self, inputs, *, target, fail=False):
        stage = self._stage
        stage.output_chunks.clear()
        stage.fwd_cache.clear()
        losses = []
        for index, (batch, labels) in enumerate(zip(inputs.chunk(2), target.chunk(2))):
            output = stage.forward_one_chunk(index, (batch,))
            assert stage.fwd_cache[index] is output
            assert len(stage.output_chunks) == (
                index + 1 if stage.retain_outputs else 0
            )
            if fail:
                raise RuntimeError("injected pipeline failure")
            with torch.autocast(
                "cpu",
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
                cache_enabled=False,
            ):
                losses.append(torch.nn.functional.mse_loss(output, labels) / 2)
        for index, loss in enumerate(losses):
            loss.backward()
            stage.fwd_cache.pop(index)
        return self._merge_outputs(stage.output_chunks)

    monkeypatch.setattr(ScheduleGPipe, "step", run)
    schedule = object.__new__(OutputAwareGPipe)
    schedule._stage = stage
    schedule._output_merge_spec = None
    return schedule, references


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_output_policy_preserves_updates_and_releases_chunks(monkeypatch, precision):
    from torch.distributed.pipelining import ScheduleGPipe

    schedule, references = schedule_fixture(monkeypatch, precision)
    stage = schedule._stage
    reference = copy.deepcopy(stage.submod)
    optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=0.01, foreach=False)
    baseline = torch.optim.AdamW(reference.parameters(), lr=0.01, foreach=False)
    merge = ScheduleGPipe._merge_outputs
    calls = []

    def checked_merge(self, chunks):
        assert self._stage.retain_outputs, "unused outputs must never be concatenated"
        calls.append(len(chunks))
        return merge(self, chunks)

    monkeypatch.setattr(ScheduleGPipe, "_merge_outputs", checked_merge)
    inputs, targets = torch.randn(4, 4), torch.randn(4, 4)
    for retain in (False, True, False, True):
        optimizer.zero_grad()
        baseline.zero_grad()
        expected_outputs, losses = [], []
        with torch.autocast(
            "cpu",
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
            cache_enabled=False,
        ):
            for batch, labels in zip(inputs.chunk(2), targets.chunk(2)):
                output = reference(batch)
                expected_outputs.append(output.detach())
                losses.append(torch.nn.functional.mse_loss(output, labels) / 2)
        for loss in losses:
            loss.backward()
        result = schedule.step(inputs, target=targets, return_outputs=retain)
        if retain:
            torch.testing.assert_close(
                result, torch.cat(expected_outputs), rtol=0, atol=0
            )
        else:
            assert result is None
        assert not stage.output_chunks and not stage.fwd_cache
        assert stage.retain_outputs is False
        assert all(reference() is None for reference in references)
        for actual, expected in zip(stage.submod.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        optimizer.step()
        baseline.step()
        for actual, expected in zip(stage.submod.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key in optimizer.state[actual]:
                torch.testing.assert_close(
                    optimizer.state[actual][key],
                    baseline.state[expected][key],
                    rtol=0,
                    atol=0,
                )
    assert calls == [2, 2]


@pytest.mark.parametrize("retain", [False, True])
def test_error_resets_only_auxiliary_output_policy(monkeypatch, retain):
    schedule, _ = schedule_fixture(monkeypatch, "fp32")
    with pytest.raises(RuntimeError, match="injected pipeline"):
        schedule.step(
            torch.randn(4, 4),
            target=torch.randn(4, 4),
            fail=True,
            return_outputs=retain,
        )
    assert schedule._stage.retain_outputs is False
    assert not schedule._stage.output_chunks
    # Native backward/communication state is not ours to discard after a failure.
    assert schedule._stage.fwd_cache


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("retain", [False, True])
def test_trainer_passes_output_policy_to_schedule(monkeypatch, stage, retain):
    from parascale.parallel import DistributedParallelConfig
    from parascale.parallel.execution import DistributedTrainer

    trainer = object.__new__(DistributedTrainer)
    trainer._precision = "fp32"
    trainer.device = torch.device("cpu")
    trainer.config = DistributedParallelConfig(pp_size=2)
    trainer.dp_rank = trainer.tp_rank = 0
    trainer.pp_rank = stage
    trainer._normalization_count = None
    trainer._prepare_step = lambda *args: None
    trainer.mesh = {"pp": SimpleNamespace(get_group=lambda: None)}
    seen = []

    def step(*args, **kwargs):
        seen.append(kwargs["return_outputs"])
        return torch.zeros(4, 4) if stage == 1 and retain else None

    trainer.schedule = SimpleNamespace(step=step)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    result = trainer.forward_backward(
        torch.randn(4, 4), torch.randn(4, 4), return_outputs=retain
    )
    assert seen == [retain]
    assert (result.output is not None) == (stage == 1 and retain)
