# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Complete stage progress recovery, with consumed rather than issued data."""

import copy
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from parascale.data.text import TokenBatchCursor  # noqa: E402
from parascale.optimizers.schedule import SchedulerSpec  # noqa: E402
from parascale.parallel import distributed_checkpoint as checkpoint  # noqa: E402
from parascale.runtime.training.data_state import DataResumeController  # noqa: E402
from parascale.runtime.training.engine import TrainState  # noqa: E402
from tests.test_distributed_checkpoint import (  # noqa: E402
    runtime as _checkpoint_runtime,
)
from tests.test_text_cursor import corpus  # noqa: E402

runtime = _checkpoint_runtime


def setup_runtime(tmp_path, optimizer, *, schedule=True):
    from parascale.runtime.training.stage_checkpoint import StageTrainingCheckpoint

    spec = SchedulerSpec("cosine", total_steps=8, warmup_steps=2) if schedule else None
    engine = SimpleNamespace(
        state=TrainState(),
        scheduler=spec.build(optimizer) if spec else None,
        scheduler_metadata=spec.to_metadata() if spec else {},
    )
    engine.data_resume = DataResumeController(engine)
    cursor = TokenBatchCursor(corpus(tmp_path), batch_size=2, sequence_length=4)
    participant = StageTrainingCheckpoint(
        engine, cursor, optimizer, microbatches=2, scheduler_spec=spec
    )
    iterator = engine.data_resume.prepare_iterator(cursor)
    return engine, cursor, participant, iterator


def update(trainer, optimizer, engine, iterator, first=None):
    batches = [next(iterator) if first is None else first, next(iterator)]
    optimizer.zero_grad(set_to_none=True)
    loss = sum(
        trainer.module(batch["input_ids"][:, :2].float() / 10).square().mean()
        * (1 + torch.rand(()) + random.random() + float(np.random.random()))
        / 2
        for batch in batches
    )
    loss.backward()
    optimizer.step()
    if engine.scheduler is not None:
        engine.scheduler.step()
    engine.state.global_step += 1
    engine.data_resume.record_consumption(
        {"gradient_accumulation_steps": 2, "batch_size": 4}
    )
    return loss.item()


@pytest.mark.parametrize("schedule", [False, True])
@pytest.mark.parametrize("step", [0, 1, 2, 4, 8])
def test_full_resume_matches_next_update_with_prefetched_cursor(
    tmp_path, runtime, schedule, step
):
    trainer, optimizer, _ = runtime
    engine, cursor, participant, iterator = setup_runtime(
        tmp_path, optimizer, schedule=schedule
    )
    for _ in range(step):
        update(trainer, optimizer, engine, iterator)
    ahead = next(iterator)
    assert cursor.position == step * 2 + 1
    directory = tmp_path / "checkpoint"
    checkpoint.save_checkpoint(
        trainer, directory, optimizer, step=step, runtime=participant
    )
    expected_loss = update(trainer, optimizer, engine, iterator, first=ahead)
    expected_model = copy.deepcopy(trainer.module.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_schedule = (
        copy.deepcopy(engine.scheduler.state_dict()) if schedule else None
    )
    checkpoint.load_checkpoint(trainer, directory, optimizer, runtime=participant)
    assert engine.state.global_step == step
    assert engine.state.consumed_micro_batches == step * 2
    assert engine.state.consumed_samples == step * 4
    with pytest.raises(RuntimeError, match="superseded"):
        next(iterator)
    restored = engine.data_resume.prepare_iterator(cursor)
    assert update(trainer, optimizer, engine, restored) == expected_loss
    for name, value in trainer.module.state_dict().items():
        assert torch.equal(value, expected_model[name])
    actual = optimizer.state_dict()
    assert actual["param_groups"] == expected_optimizer["param_groups"]
    for slot, state in actual["state"].items():
        for key, value in state.items():
            assert torch.equal(value, expected_optimizer["state"][slot][key])
    if schedule:
        assert engine.scheduler.state_dict() == expected_schedule


@pytest.mark.parametrize(
    "damage",
    [
        "step",
        "microbatches",
        "samples",
        "position",
        "cursor_identity",
        "python_rng",
        "numpy_rng",
        "scheduler_step",
        "scheduler_lr",
        "optimizer_lr",
    ],
)
def test_invalid_runtime_rejected_without_mutation(tmp_path, runtime, damage):
    trainer, optimizer, _ = runtime
    engine, cursor, participant, iterator = setup_runtime(tmp_path, optimizer)
    update(trainer, optimizer, engine, iterator)
    state = participant.capture()
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    if damage == "step":
        state["global_step"] = True
    elif damage == "microbatches":
        state["consumed_micro_batches"] += 1
    elif damage == "samples":
        state["consumed_samples"] += 1
    elif damage == "position":
        state["data_state"]["state"]["position"] += 1
    elif damage == "cursor_identity":
        state["data_state"]["state"]["identity"]["rank"] = 1
    elif damage == "python_rng":
        state["rng"]["python"] = ()
    elif damage == "numpy_rng":
        state["rng"]["numpy"] = ()
    elif damage == "scheduler_step":
        state["scheduler"]["last_epoch"] += 1
    elif damage == "scheduler_lr":
        state["scheduler"]["_last_lr"][0] = 0.3
    elif damage == "optimizer_lr":
        optimizer_state["param_groups"][0]["lr"] = 0.3
    before = participant.capture()
    with pytest.raises((ValueError, TypeError)):
        participant.validate(state, step=1, optimizer_state=optimizer_state)
    assert participant.capture() == before


def test_wrong_optimizer_binding_and_batch_policy_rejected(tmp_path, runtime):
    trainer, optimizer, _ = runtime
    engine, cursor, participant, iterator = setup_runtime(tmp_path, optimizer)
    with pytest.raises(ValueError, match="optimizer"):
        participant.identity(torch.optim.Adam(trainer.module.parameters()))
    checkpoint.save_checkpoint(
        trainer, tmp_path / "checkpoint", optimizer, step=0, runtime=participant
    )
    participant.microbatches = 4
    trainer.module.load_state_dict = Mock(side_effect=AssertionError("model mutation"))
    with pytest.raises(RuntimeError, match="runtime identity"):
        checkpoint.load_checkpoint(
            trainer, tmp_path / "checkpoint", optimizer, runtime=participant
        )
    trainer.module.load_state_dict.assert_not_called()


@pytest.mark.parametrize("replacement", ["initial", "scheduler", "lambda"])
def test_schedule_implementation_must_match_declared_spec(
    tmp_path, runtime, replacement
):
    trainer, optimizer, _ = runtime
    engine, cursor, participant, iterator = setup_runtime(tmp_path, optimizer)

    def bad(step):
        return 0.0 if step == 0 else 0.9

    if replacement == "initial":
        from parascale.runtime.training.stage_checkpoint import StageTrainingCheckpoint

        engine.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, bad)
        participant = StageTrainingCheckpoint(
            engine,
            cursor,
            optimizer,
            microbatches=2,
            scheduler_spec=participant.scheduler_spec,
        )
    elif replacement == "scheduler":
        engine.scheduler = participant.scheduler_spec.build(optimizer)
    else:
        engine.scheduler.lr_lambdas[0] = bad
    with pytest.raises(ValueError, match="scheduler"):
        participant.identity(optimizer)


@pytest.mark.parametrize("cached", ["bad", float("nan"), float("inf"), True])
def test_invalid_python_gaussian_cache_does_not_mutate_rng(tmp_path, runtime, cached):
    _, optimizer, _ = runtime
    engine, cursor, participant, iterator = setup_runtime(tmp_path, optimizer)
    state = participant.capture()
    version, internal, _ = state["rng"]["python"]
    state["rng"]["python"] = (version, internal, cached)
    before = random.getstate()
    with pytest.raises(ValueError, match="Python RNG"):
        participant.validate(state, step=0, optimizer_state=optimizer.state_dict())
    assert random.getstate() == before


def test_fresh_objects_restore_next_update(tmp_path, runtime):
    trainer, optimizer, _ = runtime
    engine, cursor, participant, iterator = setup_runtime(tmp_path, optimizer)
    for _ in range(3):
        update(trainer, optimizer, engine, iterator)
    checkpoint.save_checkpoint(
        trainer, tmp_path / "checkpoint", optimizer, step=3, runtime=participant
    )
    expected_loss = update(trainer, optimizer, engine, iterator)
    expected_model = copy.deepcopy(trainer.module.state_dict())
    trainer.module = torch.nn.Linear(2, 2)
    fresh_optimizer = torch.optim.Adam(trainer.module.parameters())
    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    fresh_engine, fresh_cursor, fresh_runtime, _ = setup_runtime(
        fresh_dir, fresh_optimizer
    )
    checkpoint.load_checkpoint(
        trainer, tmp_path / "checkpoint", fresh_optimizer, runtime=fresh_runtime
    )
    fresh_iterator = fresh_engine.data_resume.prepare_iterator(fresh_cursor)
    assert (
        update(trainer, fresh_optimizer, fresh_engine, fresh_iterator) == expected_loss
    )
    for name, value in trainer.module.state_dict().items():
        assert torch.equal(value, expected_model[name])
