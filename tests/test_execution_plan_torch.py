# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Rank-local execution-plan errors must fail before training or recovery."""

import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_single_process_payload_and_checkpoint_ignore_inventory(tmp_path, monkeypatch):
    from parascale.checkpoint import CheckpointManager
    from parascale.runtime.train_runner import run_train_from_config

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint = tmp_path / "checkpoint"
    result = run_train_from_config({
        "parascale": {"training_backend": "native"},
        "hardware_profile": {"num_gpus": 4},
        "training": {"max_steps": 1, "checkpoint_dir": str(checkpoint)},
        "runtime": {"run_dir": str(tmp_path / "run")},
    })
    assert result["strategy_plan"]["dp_size"] == 1
    assert result["execution"][0]["world_size"] == 1
    assert CheckpointManager(str(checkpoint)).read_manifest(1).parallel_plan["dp_size"] == 1


def _failure_worker(rank, root, scenario):
    from parascale.config import ParaScaleConfig
    from parascale.runtime.train_runner import run_train_from_config
    from parascale.runtime.training import TrainEngine

    root = Path(root)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    torch.distributed.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=8),
    )
    try:
        if scenario == "resume_plan":
            from parascale.checkpoint import CheckpointManager
            from parascale.runtime.training.checkpointing import CheckpointController

            engine = TrainEngine(
                ParaScaleConfig(training_backend="native_ddp"),
                hardware_profile={"world_size": 2 + rank},
            )
            original = CheckpointController.load
            def forbidden(*args, **kwargs):
                raise AssertionError("must validate plan before any checkpoint application")
            CheckpointController.load = forbidden
            try:
                with pytest.raises((ValueError, RuntimeError), match="world_size|another rank"):
                    engine.load_checkpoint(CheckpointManager(str(root / "checkpoint")), 1)
            finally:
                CheckpointController.load = original
        else:
            config = {
                "parascale": {"training_backend": "auto"},
                "training": {
                    "max_steps": 1, "checkpoint_dir": str(root / "checkpoints"),
                    "checkpoint_interval": 0, "skip_final_checkpoint": True,
                },
                "runtime": {"run_dir": str(root / "run")},
                "hardware_profile": {"world_size": 2 + (rank if scenario == "auto" else 0)},
            }
            if scenario == "missing_launcher":
                os.environ.pop("RANK")
                os.environ.pop("WORLD_SIZE")
            with pytest.raises((ValueError, RuntimeError), match="world_size|launcher|another rank"):
                run_train_from_config(config)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("scenario", ["auto", "missing_launcher", "resume_plan"])
def test_rank_local_plan_errors_fail_all_ranks(tmp_path, monkeypatch, scenario):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = torch.multiprocessing.get_context("spawn")
    processes = [context.Process(target=_failure_worker, args=(rank, str(tmp_path), scenario))
                 for rank in range(2)]
    deadline = time.monotonic() + 45
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
