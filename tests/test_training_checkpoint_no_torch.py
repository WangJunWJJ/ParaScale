# -*- coding: utf-8 -*-
# @Time : 2026/6/16 下午4:19
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest


def test_replay_resume_expands_only_component_training_window():
    from parascale.checkpoint import CheckpointManifest
    from parascale.runtime.train_runner import _resume_component_config

    config = {
        "training": {"max_steps": 2, "workload": "synthetic_regression"},
        "data": {"batch_size": 4},
    }
    manifest = CheckpointManifest(
        step=3,
        data_state={
            "resume_mode": "replay_skip",
            "consumed_micro_batches": 3,
        },
    )

    component_config = _resume_component_config(config, manifest, max_steps=2)

    assert component_config is not config
    assert component_config["training"]["max_steps"] == 5
    assert config["training"]["max_steps"] == 2

def test_stateful_resume_keeps_original_component_training_window():
    from parascale.checkpoint import CheckpointManifest
    from parascale.runtime.train_runner import _resume_component_config

    config = {"training": {"max_steps": 2}}
    manifest = CheckpointManifest(
        step=3,
        data_state={
            "resume_mode": "state_dict",
            "consumed_micro_batches": 3,
            "state": {"cursor": 3},
        },
    )

    component_config = _resume_component_config(config, manifest, max_steps=2)

    assert component_config is config

def test_checkpoint_optimizer_metadata_mismatch_fails_before_load():
    from parascale.runtime.training.checkpointing import CheckpointController

    manifest = type(
        "Manifest",
        (),
        {"metadata": {"optimizer": {"type": "four_bit_adamw"}}},
    )()
    optimizer = type(
        "Optimizer",
        (),
        {"_parascale_optimizer_metadata": {"type": "four_bit_sgd"}},
    )()

    with pytest.raises(ValueError, match="optimizer metadata mismatch.*type"):
        CheckpointController._validate_optimizer_metadata(manifest, optimizer)

def test_train_engine_exposes_checkpoint_collective_rank_and_barrier(monkeypatch):
    from parascale.core import MockCollectiveBackend
    from parascale.runtime.training import TrainEngine

    collective = MockCollectiveBackend(initialized=True, world_size=2, rank=1)
    engine = TrainEngine(config=object(), collective=collective)
    monkeypatch.setattr(engine, "_initialized_torch_distributed", lambda: None)
    monkeypatch.delenv("RANK", raising=False)

    assert engine._distributed_rank() == 1
    assert engine._distributed_world_size() == 2

    engine._distributed_barrier()

    assert collective.history[-1]["op"] == "barrier"

def test_train_engine_initialize_preserves_injected_collective_rank(monkeypatch):
    from parascale.core import MockCollectiveBackend
    from parascale.runtime.training import TrainEngine

    monkeypatch.delenv("RANK", raising=False)
    collective = MockCollectiveBackend(world_size=2, rank=1)
    plan = type("Plan", (), {"dp_size": 2, "tp_size": 1, "pp_size": 1})()
    engine = TrainEngine(
        config=object(),
        collective=collective,
        strategy_plan=plan,
        training_backend=object(),
    )

    engine.initialize()

    assert collective.rank == 1
    assert collective.history[-1]["kwargs"]["rank"] == 1

def test_train_engine_initialize_uses_launcher_rank(monkeypatch):
    from parascale.core import MockCollectiveBackend
    from parascale.runtime.training import TrainEngine

    monkeypatch.setenv("RANK", "3")
    collective = MockCollectiveBackend(world_size=4, rank=1)
    plan = type("Plan", (), {"dp_size": 4, "tp_size": 1, "pp_size": 1})()
    engine = TrainEngine(
        config=object(),
        collective=collective,
        strategy_plan=plan,
        training_backend=object(),
    )

    engine.initialize()

    assert collective.rank == 3
    assert collective.history[-1]["kwargs"]["rank"] == 3

def test_train_engine_initialize_rejects_invalid_launcher_rank(monkeypatch):
    from parascale.core import MockCollectiveBackend
    from parascale.runtime.training import TrainEngine

    monkeypatch.setenv("RANK", "not-an-integer")
    plan = type("Plan", (), {"dp_size": 2, "tp_size": 1, "pp_size": 1})()
    engine = TrainEngine(
        config=object(),
        collective=MockCollectiveBackend(world_size=2, rank=1),
        strategy_plan=plan,
        training_backend=object(),
    )

    with pytest.raises(ValueError):
        engine.initialize()

def test_nonzero_rank_checkpoint_result_skips_manifest_validation():
    from parascale.runtime.train_runner import _validate_final_checkpoint_result

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
