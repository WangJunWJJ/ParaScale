# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CPU lifecycle tests for the pinned 2.9.1 single-stage schedule contract."""

import weakref
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(params=["gpipe", "1f1b"])
def schedule_case(monkeypatch, request):
    from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe

    from parascale.parallel.pipeline_stage import pipeline_schedule_type

    native = ScheduleGPipe if request.param == "gpipe" else Schedule1F1B
    stage = SimpleNamespace(
        num_stages=2,
        stage_index=1,
        is_last=True,
        has_backward=False,
        retain_outputs=False,
        output_chunks=[],
        fwd_cache={},
        submod=torch.nn.Linear(4, 4),
    )

    def clear():
        stage.fwd_cache.clear()
        stage.output_chunks.clear()

    stage.clear_runtime_states = clear
    schedule = pipeline_schedule_type(request.param)(
        stage,
        2,
        loss_fn=torch.nn.functional.mse_loss,
        scale_grads=False,
    )
    # Local 2.15 changed initialization and forward caches. Model only the
    # pinned initialization/communication boundary; keep native eval/step/split.
    schedule._stage_initialized = False
    events = SimpleNamespace(initializations=[], modes=[], refs=[], fail=None)

    def initialize(self, args, kwargs):
        events.initializations.append((args, kwargs, stage.has_backward))
        stage.fwd_cache["initializing"] = torch.ones(1)
        self._internal_losses.append(torch.ones(()))
        if events.fail == "initialize":
            raise RuntimeError("injected initialize failure")
        stage.backward_ready = stage.has_backward and self._has_backward
        self._stage_initialized = True

    def microbatches(self, args, kwargs, targets, losses, *unused, **options):
        if not self._stage_initialized:
            self._initialize_stage(args[0], kwargs[0])
        self._internal_losses.clear()
        if self._has_backward:
            assert stage.backward_ready, "missing backward infrastructure"
        events.modes.append(
            (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )
        for index, (batch_args, batch_kwargs) in enumerate(zip(args, kwargs)):
            output = stage.submod(batch_args[0] + batch_kwargs["offset"])
            events.refs.append(weakref.ref(output))
            stage.fwd_cache[index] = output
            if stage.retain_outputs:
                stage.output_chunks.append(output)
            loss = self._loss_fn(output, targets[index])
            self._internal_losses.append(loss)
            if events.fail == "forward":
                raise RuntimeError("injected forward failure")
            if self._has_backward:
                loss.backward()
                stage.fwd_cache.pop(index)
        self._update_losses(stage, losses)
        # Pinned schedules scale existing parameter gradients even during eval.
        if self.scale_grads:
            for parameter in stage.submod.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(self._n_microbatches)

    monkeypatch.setattr(native, "_initialize_stage", initialize)
    monkeypatch.setattr(native, "_step_microbatches", microbatches)
    return schedule, stage, events


def batch():
    return torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)


def test_cold_eval_initializes_with_training_flag_and_first_microbatch(schedule_case):
    schedule, stage, events = schedule_case
    inputs, offset, targets = batch()
    losses = []
    assert schedule.eval(inputs, offset=offset, target=targets, losses=losses) is None
    assert len(events.initializations) == 1
    args, kwargs, backward = events.initializations[0]
    assert backward is True
    torch.testing.assert_close(args[0], inputs[:2])
    assert set(kwargs) == {"offset"}
    torch.testing.assert_close(kwargs["offset"], offset[:2])
    assert len(losses) == 2
    assert all(not loss.requires_grad for loss in losses)
    assert events.modes == [(False, False)]
    assert stage.has_backward is False
    assert schedule._has_backward is True
    assert not stage.fwd_cache and not stage.output_chunks
    assert not schedule._internal_losses
    schedule.step(inputs, offset=offset, target=targets)
    assert len(events.initializations) == 1
    assert all(parameter.grad is not None for parameter in stage.submod.parameters())


@pytest.mark.parametrize("retain", [False, True])
def test_warm_repeated_eval_preserves_gradients_and_output_policy(
    schedule_case, retain
):
    schedule, stage, events = schedule_case
    inputs, offset, targets = batch()
    schedule.step(inputs, offset=offset, target=targets)
    gradients = [(p.grad, p.grad.clone()) for p in stage.submod.parameters()]
    events.refs.clear()
    for _ in range(3):
        result = schedule.eval(
            inputs, offset=offset, target=targets, return_outputs=retain
        )
        if retain:
            torch.testing.assert_close(result, stage.submod(inputs + offset))
            assert not result.requires_grad and not result.is_inference()
        else:
            assert result is None
        assert stage.has_backward is True and schedule._has_backward is True
        assert stage.retain_outputs is False
        assert not stage.fwd_cache and not stage.output_chunks
        assert not schedule._internal_losses
        assert all(ref() is None for ref in events.refs)
        for parameter, (original, saved) in zip(stage.submod.parameters(), gradients):
            assert parameter.grad is original
            torch.testing.assert_close(parameter.grad, saved, rtol=0, atol=0)
    assert len(events.initializations) == 1
    schedule.step(inputs, offset=offset, target=targets)
    for parameter, (_, saved) in zip(stage.submod.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, saved * 2)


@pytest.mark.parametrize("failure", ["initialize", "split", "forward"])
@pytest.mark.parametrize("original_flag", [False, True])
def test_eval_failure_restores_flags_and_releases_caches(
    schedule_case,
    monkeypatch,
    failure,
    original_flag,
):
    schedule, stage, events = schedule_case
    stage.has_backward = original_flag
    stage.retain_outputs = True
    events.fail = failure
    if failure == "split":

        def fail_split(*args, **kwargs):
            stage.fwd_cache[0] = torch.ones(1)
            schedule._internal_losses.append(torch.ones(()))
            raise RuntimeError("injected split failure")

        monkeypatch.setattr(schedule, "_split_inputs", fail_split)
    inputs, offset, targets = batch()
    with torch.enable_grad():
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            schedule.eval(inputs, offset=offset, target=targets, return_outputs=True)
        assert torch.is_grad_enabled()
    assert stage.has_backward is original_flag
    assert schedule._has_backward is True
    assert stage.retain_outputs is True
    assert not stage.fwd_cache and not stage.output_chunks
    assert not schedule._internal_losses


def test_eval_rejects_gradient_scaling_before_native_execution(schedule_case):
    schedule, stage, events = schedule_case
    schedule.scale_grads = True
    for parameter in stage.submod.parameters():
        parameter.grad = torch.ones_like(parameter)
    inputs, offset, targets = batch()
    with pytest.raises(ValueError, match="scale_grads=False"):
        schedule.eval(inputs, offset=offset, target=targets)
    assert not events.initializations and not events.modes
    assert stage.has_backward is False and schedule._has_backward is True
    for parameter in stage.submod.parameters():
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


def test_eval_restores_callers_no_grad_context(schedule_case):
    schedule, _, events = schedule_case
    inputs, offset, targets = batch()
    with torch.no_grad():
        schedule.eval(inputs, offset=offset, target=targets)
        assert not torch.is_grad_enabled()
    assert events.modes == [(False, False)]


def test_cold_eval_without_root_inputs_initializes_empty_microbatch(
    schedule_case,
    monkeypatch,
):
    schedule, stage, events = schedule_case

    def receive_only(*args, **kwargs):
        assert stage.has_backward is False
        assert stage.backward_ready is True

    monkeypatch.setattr(schedule, "_step_microbatches", receive_only)
    assert schedule.eval(target=torch.randn(4, 4)) is None
    assert events.initializations == [((), {}, True)]
    assert stage.has_backward is False
    assert not stage.fwd_cache and not schedule._internal_losses
