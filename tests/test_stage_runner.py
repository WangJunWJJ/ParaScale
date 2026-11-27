# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Launch and recovery selection contracts for the production stage runner."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")


def launch_env(monkeypatch, **changes):
    values = dict(
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
    )
    values.update(changes)
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "changes",
    [
        {"LOCAL_RANK": None},
        {"LOCAL_WORLD_SIZE": "1"},
        {"RANK": "2"},
        {"LOCAL_RANK": "1"},
        {"WORLD_SIZE": "4"},
        {"MASTER_PORT": "0"},
    ],
)
def test_launch_rejects_incomplete_or_multinode_metadata(monkeypatch, changes):
    from parascale.runtime.stage_train_runner import _validate_stage_launch

    launch_env(monkeypatch, **changes)
    with pytest.raises(ValueError):
        _validate_stage_launch(SimpleNamespace(topology=SimpleNamespace(world_size=2)))


def test_launch_accepts_complete_single_node_metadata(monkeypatch):
    from parascale.runtime.stage_train_runner import _validate_stage_launch

    launch_env(monkeypatch, RANK="1", LOCAL_RANK="1")
    assert _validate_stage_launch(
        SimpleNamespace(topology=SimpleNamespace(world_size=2))
    ) == (1, 1)


@pytest.mark.parametrize(
    "step,latest", [(True, False), (-1, False), (1.5, False), ("1", False), (1, True)]
)
def test_resume_request_validation(step, latest):
    from parascale.runtime.stage_train_runner import _resume_request

    with pytest.raises(ValueError):
        _resume_request({}, step, latest)


def test_resume_request_honors_training_and_detects_conflict():
    from parascale.runtime.stage_train_runner import _resume_request

    assert _resume_request({"training": {"resume_step": 3}}, None, False) == 3
    with pytest.raises(ValueError):
        _resume_request({"training": {"resume_step": 3}}, None, True)


def test_rank_zero_operation_coordinates_errors_and_broadcasts(monkeypatch):
    from parascale.runtime import stage_train_runner as runner

    dist = SimpleNamespace(get_rank=lambda: 0, broadcast_object_list=Mock())
    monkeypatch.setattr(runner, "dist", dist)
    phases = []

    def coordinate(operation, *, phase, **kwargs):
        phases.append(phase)
        return operation()

    monkeypatch.setattr(runner, "coordinated_call", coordinate)
    assert runner._rank_zero(lambda: {"selected": 3}, phase="selection") == {
        "selected": 3
    }
    assert phases == ["selection"]
    assert dist.broadcast_object_list.call_count == 1
    dist.broadcast_object_list.reset_mock()
    with pytest.raises(OSError, match="disk full"):
        runner._rank_zero(Mock(side_effect=OSError("disk full")), phase="selection")
    dist.broadcast_object_list.assert_not_called()


def test_fresh_checkpoint_root_rejects_existing_steps(tmp_path):
    from parascale.runtime.stage_train_runner import _prepare_checkpoint_root

    root = tmp_path / "checkpoints"
    _prepare_checkpoint_root(root, resuming=False)
    (root / "step-00000001").mkdir()
    with pytest.raises(ValueError, match="resume"):
        _prepare_checkpoint_root(root, resuming=False)
    _prepare_checkpoint_root(root, resuming=True)


def test_checkpoint_root_rejects_parent_traversal_before_creation(tmp_path):
    from parascale.runtime.stage_train_runner import _prepare_checkpoint_root

    root = tmp_path / "new" / ".." / "snapshots"
    with pytest.raises(ValueError, match="traversal"):
        _prepare_checkpoint_root(root, resuming=False)
    assert not (tmp_path / "new").exists()
    assert not (tmp_path / "snapshots").exists()


def test_recovery_rejects_colliding_save_before_training(tmp_path):
    from parascale.runtime.stage_train_runner import _validate_save_targets

    (tmp_path / "step-00000003").mkdir()
    plan = SimpleNamespace(checkpoint_dir=tmp_path, max_steps=1, checkpoint_interval=10)
    with pytest.raises(ValueError, match="already exists"):
        _validate_save_targets(plan, initial_step=2)
    _validate_save_targets(plan, initial_step=3)


def test_recovery_rejects_newer_snapshot_beyond_update_window(tmp_path):
    from parascale.runtime.stage_train_runner import _validate_save_targets

    (tmp_path / "step-00000004").mkdir()
    plan = SimpleNamespace(checkpoint_dir=tmp_path, max_steps=1, checkpoint_interval=10)
    with pytest.raises(ValueError, match="already exists"):
        _validate_save_targets(plan, initial_step=2)


@pytest.mark.parametrize("checkpoint_format", ["rank", "dcp"])
@pytest.mark.parametrize("ddp_comm_hook", ["none", "bf16_compress"])
@pytest.mark.parametrize("evaluation", [False, True])
def test_runner_wires_real_cursor_optimizer_and_runtime_checkpoint(
    tmp_path, monkeypatch, checkpoint_format, ddp_comm_hook, evaluation
):
    import torch

    from parascale.data.text import TokenBatchCursor
    from parascale.runtime import stage_train_runner as runner
    from tests.test_stage_plan_no_torch import settings
    from tests.test_text_cursor import corpus

    config = settings()
    config["parascale"]["backend"]["ddp_comm_hook"] = ddp_comm_hook
    config["training"].update(max_steps=2, gradient_accumulation_steps=2)
    config["data"].update(sequence_length=4)
    config["parascale"]["training"] = {
        "checkpoint_save_path": str(tmp_path / "snapshots"),
        "checkpoint_save_interval": 2,
        "checkpoint_format": checkpoint_format,
    }
    world_size = 2 if checkpoint_format == "dcp" or ddp_comm_hook != "none" else 1
    config["parascale"]["parallel"] = {"data_parallel_size": world_size}
    stage = torch.nn.Linear(4, 1)
    stage.stage_index = 0
    stage.vocab_size = 64
    stage.tensor_parallel_plan = lambda: {}
    cursor = TokenBatchCursor(corpus(tmp_path), batch_size=2, sequence_length=4)
    if evaluation:
        config["evaluation"] = {"data_path": "/validation", "max_batches": 1}
        heldout = TokenBatchCursor(
            cursor.corpus,
            batch_size=2,
            sequence_length=4,
            world_size=world_size,
            repeat=False,
            shuffle=False,
        )
        monkeypatch.setattr(
            runner, "build_causal_lm_evaluation_cursor", lambda *args, **kwargs: heldout
        )
    monkeypatch.setattr(runner, "_initialize", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        runner,
        "dist",
        SimpleNamespace(
            get_rank=lambda: 0,
            get_world_size=lambda: world_size,
            broadcast_object_list=lambda *args, **kwargs: None,
            all_reduce=lambda *args, **kwargs: None,
            ReduceOp=SimpleNamespace(MAX="max"),
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_causal_lm_stage_components",
        lambda *args, **kwargs: (stage, cursor, None),
    )
    monkeypatch.setattr(runner, "build_stage_checkpoint_context", lambda *args: {})
    monkeypatch.setattr(
        runner, "prune_stage_checkpoints", lambda *args, **kwargs: {"removed": []}
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    captured = []
    expected_format = checkpoint_format

    class CpuExecutor:
        def __init__(self, module, *args, **kwargs):
            self.module = module
            self.config = args[0]
            self.device = torch.device("cpu")
            self.ddp_comm_hook = kwargs["ddp_comm_hook"]

        def evaluate(self, inputs, labels, **kwargs):
            with torch.no_grad():
                value = self.module(inputs.float()).square().mean()
            return SimpleNamespace(
                loss=value.item(), normalization_count=12 * world_size
            )

        def forward_backward(self, inputs, labels, **kwargs):
            value = self.module(inputs.float()).square().mean()
            value.backward()
            return SimpleNamespace(loss=value.item(), normalization_count=12)

        def save_checkpoint(self, path, optimizer, *, step, runtime, checkpoint_format):
            assert checkpoint_format == expected_format
            snapshot = runtime.capture()
            runtime.validate(
                snapshot, step=step, optimizer_state=optimizer.state_dict()
            )
            captured.append(snapshot)

    monkeypatch.setattr(runner, "DistributedTrainer", CpuExecutor)
    result = runner.run_stage_train_from_config(config)
    assert result["global_step"] == 2
    assert result["ddp_comm_hook"] == ddp_comm_hook
    assert result["samples"] == 8 * world_size
    assert result["tokens"] == 32 * world_size
    assert len(captured) == 1
    assert captured[0]["data_state"]["state"]["position"] == 4
    assert captured[0]["consumed_samples"] == 8
    assert result["checkpoint"]["step"] == 2
    assert result["checkpoint"]["format"] == expected_format
    if evaluation:
        assert result["evaluation"]["initial"]["global_step"] == 0
        assert result["evaluation"]["final"]["global_step"] == 2
        assert result["evaluation"]["final"]["scored_tokens"] == 12 * world_size
        assert heldout.position == 0
    else:
        assert result["evaluation"] is None
