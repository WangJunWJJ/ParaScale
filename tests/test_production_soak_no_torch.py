# -*- coding: utf-8 -*-
# @Time : 2026/11/20 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.benchmarks.scripts import production_soak
from tests.benchmarks.scripts.production_soak import (
    check_progress,
    check_training_record,
    hf_state_from_training,
)


def test_soak_maps_ddp_and_native_weights_without_accepting_mixed_keys():
    for prefix in ("module.wrapped_model.", "wrapped_model."):
        assert hf_state_from_training({prefix + "weight": 1}) == {"weight": 1}
    with pytest.raises(ValueError):
        hf_state_from_training({"weight": 1, "module.wrapped_model.weight": 2})


def test_soak_rejects_nonfinite_loss_and_unbounded_history():
    check_training_record({"loss": 0.1}, 1000, 1000)
    with pytest.raises(ValueError):
        check_training_record({"loss": float("nan")}, 1, 1000)
    with pytest.raises(ValueError):
        check_training_record({"loss": 0.1}, 1001, 1000)


def soak_args(tmp_path, monkeypatch):
    for key, value in {
        "PARASCALE_MODEL": "/models/clip",
        "PARASCALE_DATA": "/data/qualified",
        "PARASCALE_RUN_DIR": "/runs/source",
    }.items():
        monkeypatch.setenv(key, value)
    return SimpleNamespace(
        config=Path(__file__).resolve().parents[1]
        / "examples/gpu/example_015_clip_retrieval_quality/train.json",
        model="/models/clip",
        data="/data/qualified/train",
        output=tmp_path / "soak",
        hours=24.0,
    )


def test_soak_config_preserves_qualified_budget_and_resolves_controls(
    tmp_path, monkeypatch
):
    from parascale.configuration.resolver import normalize_execution_config

    args = soak_args(tmp_path, monkeypatch)
    config = production_soak.build_soak_config(args)
    _, runtime = normalize_execution_config(config)
    assert config["training"]["batch_size"] == 32
    assert config["data"]["cpu_prefetch"] is True
    assert config["data"]["text_length"] == 77
    assert config["optimizer"]["weight_decay"] == 0.01
    assert config["parascale"]["ddp_find_unused_parameters"] is True
    assert config["parascale"]["ddp_static_graph"] is False
    assert runtime.checkpoint_keep_last == 3
    assert runtime.metrics_history_limit == 1000
    assert config["training"]["checkpoint_interval"] == 5000
    assert "checkpoint_save_interval" not in config["parascale"]
    assert "checkpoint_keep_last" not in config["training"]
    args.hours = 1 / 30
    assert (
        production_soak.build_soak_config(args)["training"]["checkpoint_interval"]
        == 100
    )


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf")])
def test_soak_rejects_unbounded_or_invalid_duration(tmp_path, monkeypatch, hours):
    args = soak_args(tmp_path, monkeypatch)
    args.hours = hours
    with pytest.raises(ValueError, match="finite positive"):
        production_soak.build_soak_config(args)


@pytest.mark.parametrize(
    "bad", ["precision", "batch", "prefetch", "buckets", "retention_alias"]
)
def test_soak_rejects_unqualified_policy(tmp_path, monkeypatch, bad):
    from parascale.commands.common import load_config_file

    args = soak_args(tmp_path, monkeypatch)
    config = load_config_file(args.config)
    section, key, value = {
        "precision": ("parascale", "precision", "bf16"),
        "batch": ("training", "batch_size", 8),
        "prefetch": ("data", "cpu_prefetch", False),
        "buckets": ("parascale", "ddp_find_unused_parameters", False),
        "retention_alias": ("training", "checkpoint_keep_last", 99),
    }[bad]
    config[section][key] = value
    args.config = tmp_path / "bad.json"
    args.config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="qualified"):
        production_soak.build_soak_config(args)


@pytest.mark.parametrize("bad", [None, "same_mean", "nan", "shape"])
def test_soak_compares_complete_embeddings(bad):
    expected = {"image_embeddings": [[0.6, 0.8]], "text_embeddings": [[0.0, 1.0]]}
    output = {
        **expected,
        "image_embeddings_shape": [1, 2],
        "text_embeddings_shape": [1, 2],
    }
    if bad == "same_mean":
        output["image_embeddings"] = [[0.8, 0.6]]
    elif bad == "nan":
        output["text_embeddings"] = [[0.0, float("nan")]]
    elif bad == "shape":
        output["image_embeddings_shape"] = [2, 1]
    if bad:
        with pytest.raises(ValueError, match="embedding"):
            production_soak.check_embedding_output(output, expected)
    else:
        production_soak.check_embedding_output(output, expected)


def test_soak_retention_allows_one_inflight_but_not_unbounded_growth(tmp_path):
    for step in range(4):
        (tmp_path / f"step-{step:08d}").mkdir()
    production_soak.check_checkpoint_retention(tmp_path, final=False)
    with pytest.raises(RuntimeError, match="retention"):
        production_soak.check_checkpoint_retention(tmp_path, final=True)
    (tmp_path / "step-00000004").mkdir()
    with pytest.raises(RuntimeError, match="retention"):
        production_soak.check_checkpoint_retention(tmp_path, final=False)


def test_soak_retention_tracks_only_known_interruption_residue(tmp_path, monkeypatch):
    from parascale.checkpoint import CheckpointManager

    for step in range(3):
        (tmp_path / f"step-{step:08d}").mkdir()
    monkeypatch.setattr(
        CheckpointManager, "validate", lambda self, step: SimpleNamespace(ok=step < 3)
    )
    partial = tmp_path / "step-00000003"
    partial.mkdir()
    payload = partial / "partial.tmp"
    payload.write_bytes(b"unfinished")
    residue = production_soak.snapshot_checkpoint_residue(tmp_path)
    assert list(residue) == [partial.name]
    assert (
        production_soak.check_checkpoint_retention(
            tmp_path, final=True, interrupted=residue
        )
        == 4
    )
    payload.write_bytes(b"unfinished plus growth")
    with pytest.raises(RuntimeError, match="residue"):
        production_soak.check_checkpoint_retention(
            tmp_path, final=True, interrupted=residue
        )
    payload.write_bytes(b"unfinished")
    with pytest.raises(RuntimeError, match="residue"):
        production_soak.check_checkpoint_retention(tmp_path, final=True, interrupted={})


def test_soak_watchdog_requires_recent_progress_and_disk_headroom():
    check_progress({"time": 100}, 200, 30 * 1024**3)
    with pytest.raises(RuntimeError):
        check_progress({"time": 100}, 1000, 30 * 1024**3)
    with pytest.raises(RuntimeError):
        check_progress({"time": 100}, 200, 1024**3)


def test_soak_rejects_worker_exit_at_stop_boundary():
    with pytest.raises(RuntimeError, match="Unexpected worker exit"):
        production_soak.require_running([SimpleNamespace(poll=lambda: 1)])


@pytest.mark.parametrize("racing_exit", [False, True])
def test_soak_cleans_groups_even_when_leader_already_exited(monkeypatch, racing_exit):
    calls = []

    def killpg(pid, sig):
        calls.append((pid, sig))
        if racing_exit:
            raise ProcessLookupError()

    monkeypatch.setattr(production_soak.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(production_soak.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(production_soak, "group_alive", lambda pid: False)
    process = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda timeout: 0)
    production_soak.stop_process(process)
    assert len(calls) == 2
    production_soak.stop_process(process)
    assert len(calls) == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_soak_kills_surviving_child_after_leader_exit():
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
    parent = f'import subprocess,sys; subprocess.Popen([sys.executable,"-c",{child!r}]); print("ready",flush=True)'
    process = subprocess.Popen(
        [sys.executable, "-c", parent],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        process.wait(timeout=10)
        production_soak.stop_process(process)
        assert not production_soak.group_alive(process.pid)
    finally:
        production_soak.stop_process(process)
        process.stdout.close()
