# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Update commit boundaries independent of CUDA stage execution."""

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from parascale.data.text import TokenBatchCursor  # noqa: E402
from parascale.runtime.training.data_state import DataResumeController  # noqa: E402
from parascale.runtime.training.engine import TrainState  # noqa: E402
from tests.test_text_cursor import corpus  # noqa: E402


def setup_loop(tmp_path, *, steps=3, history=2, nan_gradient=False):
    from parascale.runtime.training.stage_loop import StageUpdateLoop

    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cursor = TokenBatchCursor(corpus(tmp_path), batch_size=2, sequence_length=4)
    progress = SimpleNamespace(state=TrainState(), scheduler=None)
    progress.data_resume = DataResumeController(progress)
    topology = SimpleNamespace(dp_size=2, microbatches=2)
    plan = SimpleNamespace(
        topology=topology,
        max_steps=steps,
        checkpoint_interval=2,
        metrics_history_limit=history,
    )

    def backward(inputs, labels, *, return_outputs):
        assert return_outputs is False
        loss = model(inputs.float()).square().mean()
        loss.backward()
        if nan_gradient:
            next(model.parameters()).grad.fill_(float("nan"))
        return SimpleNamespace(loss=loss.item(), normalization_count=24)

    trainer = SimpleNamespace(module=model, forward_backward=backward)
    saved = []
    loop = StageUpdateLoop(
        trainer,
        optimizer,
        progress,
        cursor,
        plan,
        torch.device("cpu"),
        lambda step: saved.append(step),
    )
    return loop, saved


def test_success_counts_dp_only_and_saves_final_once(tmp_path):
    loop, saved = setup_loop(tmp_path)
    result = loop.run()
    state = loop.progress.state
    assert (
        state.global_step,
        state.consumed_micro_batches,
        state.consumed_samples,
    ) == (3, 6, 12)
    assert saved == [2, 3]
    assert len(state.metrics_history) == 2
    assert state.metrics_history_dropped == 1
    assert result["completed_updates"] == 3
    assert result["samples"] == 24
    assert result["tokens"] == 96
    assert result["scored_tokens"] == 72


def test_periodic_final_checkpoint_is_not_duplicated(tmp_path):
    loop, saved = setup_loop(tmp_path, steps=2, history=0)
    loop.run()
    assert saved == [2]
    assert loop.progress.state.metrics_history == []
    assert loop.progress.state.metrics_history_dropped == 2


def test_nonfinite_gradient_rejects_before_optimizer_and_commit(tmp_path):
    loop, saved = setup_loop(tmp_path, nan_gradient=True)
    loop.optimizer.step = Mock(wraps=loop.optimizer.step)
    with pytest.raises(FloatingPointError, match="gradient"):
        loop.run()
    loop.optimizer.step.assert_not_called()
    assert loop.progress.state.global_step == 0
    assert loop.progress.state.consumed_micro_batches == 0
    assert loop.progress.data_resume.capture()["state"]["position"] == 0
    assert saved == []


@pytest.mark.parametrize("failure", ["optimizer", "scheduler"])
def test_mutation_failure_is_terminal_without_committed_progress(tmp_path, failure):
    loop, saved = setup_loop(tmp_path)
    failed_step = Mock(side_effect=RuntimeError("injected update failure"))
    if failure == "optimizer":
        loop.optimizer.step = failed_step
    else:
        loop.progress.scheduler = SimpleNamespace(step=failed_step)
    with pytest.raises(RuntimeError, match="injected update failure"):
        loop.run()
    assert loop.progress.state.global_step == 0
    assert loop.progress.state.consumed_samples == 0
    assert saved == []


def test_incomplete_window_does_not_execute_or_commit(tmp_path, monkeypatch):
    loop, saved = setup_loop(tmp_path)
    one = next(iter(loop.cursor))
    monkeypatch.setattr(
        loop.progress.data_resume, "prepare_iterator", lambda _: iter([one])
    )
    loop.trainer.forward_backward = Mock()
    with pytest.raises(RuntimeError, match="exhausted"):
        loop.run()
    loop.trainer.forward_backward.assert_not_called()
    assert loop.progress.state.global_step == 0
    assert saved == []


def test_consumption_failure_is_coordinated_before_save(tmp_path, monkeypatch):
    from parascale.runtime.training import stage_loop

    loop, saved = setup_loop(tmp_path)
    phases = []

    def coordinate(operation, *, phase):
        phases.append(phase)
        return operation()

    monkeypatch.setattr(stage_loop, "coordinated_call", coordinate)
    loop.progress.data_resume.record_consumption = Mock(
        side_effect=RuntimeError("commit failed")
    )
    with pytest.raises(RuntimeError, match="commit failed"):
        loop.run()
    assert phases[-1] == "consumption commit"
    assert saved == []


def _commit_failure_worker(rank, root):
    import torch.distributed as dist

    directory = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        local = directory / f"rank-{rank}"
        local.mkdir()
        loop, saved = setup_loop(local)
        if rank == 1:
            loop.progress.data_resume.record_consumption = Mock(
                side_effect=RuntimeError("injected rank-one commit failure")
            )
        with pytest.raises(RuntimeError, match="commit failure"):
            loop.run()
        assert saved == []
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("PARASCALE_DISTRIBUTED_CPU_TESTS") != "1",
    reason="explicit two-process Gloo acceptance gate",
)
def test_two_rank_commit_failure_stops_all_ranks_before_checkpoint(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_commit_failure_worker, args=(str(tmp_path),), nprocs=2, join=True)
