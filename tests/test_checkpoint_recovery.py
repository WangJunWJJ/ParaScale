# -*- coding: utf-8 -*-
# @Time : 2026/11/19 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit recovery selection, collective agreement, and real CPU resume."""

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from parascale.checkpoint import CheckpointManager, CheckpointManifest
from parascale.commands.run import register_run_parsers
from parascale.runtime import train_runner


def test_recovery_cli_flags_are_mutually_exclusive():
    parser = argparse.ArgumentParser()
    register_run_parsers(parser.add_subparsers())
    args = parser.parse_args(["train", "--config", "x.json", "--resume-latest"])
    assert args.resume_latest and args.resume_step is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["train", "--config", "x", "--resume-latest", "--resume-step", "1"]
        )


def test_recovery_rejects_configured_explicit_step():
    with pytest.raises(ValueError, match="resume.*exclusive"):
        train_runner.run_train_from_config(
            {"backend": {"training_backend": "native"}, "training": {"resume_step": 1}},
            resume_latest=True,
        )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("failed", [False, True])
def test_recovery_selection_is_broadcast_from_rank_zero(monkeypatch, rank, failed):
    manifest = CheckpointManifest(step=3)
    skipped = [{"step": 4, "reason": "incomplete"}]
    packet = {"manifest": manifest.to_dict(), "skipped": skipped, "error": None}
    if failed:
        packet = {"error": "no valid training checkpoint"}
    broadcasts = []

    def select():
        assert rank == 0, "only rank zero may scan checkpoints"
        if failed:
            raise FileNotFoundError(packet["error"])
        return manifest, skipped

    def broadcast(objects, src):
        assert src == 0
        if rank == 0:
            assert objects[0] == packet
        objects[0] = packet
        broadcasts.append(src)

    dist = SimpleNamespace(get_rank=lambda: rank, broadcast_object_list=broadcast)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    manager = SimpleNamespace(latest_valid_training_checkpoint=select)
    if failed:
        with pytest.raises(RuntimeError, match="no valid training checkpoint"):
            train_runner._select_latest_checkpoint(manager, distributed=True)
    else:
        selected, diagnostics = train_runner._select_latest_checkpoint(
            manager, distributed=True
        )
        assert selected.step == 3 and diagnostics == skipped
    assert broadcasts == [0]


def test_real_native_training_recovers_previous_complete_step(tmp_path):
    pytest.importorskip("torch")
    config = {
        "backend": {"training_backend": "native", "precision": "fp32"},
        "training": {
            "workload": "synthetic_regression",
            "max_steps": 2,
            "checkpoint_interval": 1,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "runtime": {"run_dir": str(tmp_path / "run")},
    }
    original = train_runner.run_train_from_config(config)
    assert original["global_step"] == 2
    manager = CheckpointManager(config["training"]["checkpoint_dir"])
    manager.payload_path(2, "backend_state.pt").write_bytes(b"interrupted write")
    with pytest.raises(RuntimeError, match="validation failed before resume"):
        train_runner.run_train_from_config(config, resume_step=2)
    resumed = train_runner.run_train_from_config(config, resume_latest=True)
    assert resumed["global_step"] == 3
    assert resumed["resumed_from"]["step"] == 1
    assert resumed["checkpoint_recovery"]["selected_step"] == 1
    assert resumed["checkpoint_recovery"]["skipped"][0]["step"] == 2
    assert resumed["checkpoint_validation"]["ok"]


def _distributed_recovery_worker(rank, root, fail_load):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        config = {
            "backend": {"training_backend": "native_ddp", "precision": "fp32"},
            "training": {
                "workload": "synthetic_regression",
                "max_steps": 2,
                "checkpoint_interval": 1,
                "checkpoint_dir": str(root / "checkpoints"),
            },
            "runtime": {"device": "cpu", "run_dir": str(root / "run")},
        }
        result = train_runner.run_train_from_config(config)
        assert result["global_step"] == 2
        if rank == 0:
            manager = CheckpointManager(config["training"]["checkpoint_dir"])
            manager.payload_path(2, "backend_state.pt").write_bytes(b"corrupt")
        dist.barrier()
        if fail_load:
            original_load = torch.load

            def load(path, *args, **kwargs):
                if rank == 1 and Path(path).name == "backend_state.pt":
                    raise OSError("rank-local read failure")
                return original_load(path, *args, **kwargs)

            torch.load = load
            try:
                with pytest.raises(
                    (OSError, RuntimeError), match="rank-local read failure"
                ):
                    train_runner.run_train_from_config(config, resume_latest=True)
            finally:
                torch.load = original_load
            return
        resumed = train_runner.run_train_from_config(config, resume_latest=True)
        assert resumed["global_step"] == 3
        assert resumed["checkpoint_recovery"]["selected_step"] == 1
        assert resumed["checkpoint_recovery"]["skipped"][0]["step"] == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_load", [False, True])
def test_real_two_rank_gloo_training_recovery(tmp_path, monkeypatch, fail_load):
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo is unavailable in this PyTorch build")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = torch.multiprocessing.get_context("spawn")
    workers = [
        context.Process(
            target=_distributed_recovery_worker, args=(rank, str(tmp_path), fail_load)
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + 120
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        assert [worker.exitcode for worker in workers] == [0, 0]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(timeout=10)
