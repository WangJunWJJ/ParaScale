# -*- coding: utf-8 -*-
# @Time : 2026/11/20 上午11:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from parascale import ParaScaleConfig
from parascale.checkpoint import CheckpointManager, CheckpointManifest
from parascale.data.wds_cursor import WdsCursorBatches
from parascale.runtime.training.engine import TrainEngine


def test_cursor_resume_seeks_without_replaying_transform(tmp_path, monkeypatch):
    path = tmp_path / "samples.tar"
    with tarfile.open(path, "w") as archive:
        for i in range(6):
            for suffix, blob in (("jpg", b"image"), ("txt", str(i).encode())):
                info = tarfile.TarInfo(f"{i}.{suffix}")
                info.size = len(blob)
                archive.addfile(info, io.BytesIO(blob))
    spec = SimpleNamespace(
        data_dir=str(tmp_path),
        batch_size=2,
        num_batches=10,
        pretrained_model_name_or_path="clip",
        text_length=64,
    )
    calls = []

    def transform(batch):
        calls.append(1)
        return [item["text"] for item in batch]

    loader = WdsCursorBatches(spec, transform)
    iterator = iter(loader)
    assert next(iterator) == ["0", "1"]
    state = loader.state_dict()
    expected = next(iterator)
    restored = WdsCursorBatches(spec, transform)
    restored.load_state_dict(state)
    before = len(calls)
    assert next(iter(restored)) == expected
    assert len(calls) == before + 1
    spec.num_samples = 5
    limited = WdsCursorBatches(spec, transform)
    assert [len(batch) for batch in limited] == [2, 2, 1]
    assert limited.position == 5
    changed = dict(state, identity={})
    with pytest.raises(ValueError, match="identity mismatch"):
        restored.load_state_dict(changed)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    other_rank = WdsCursorBatches(spec, transform)
    assert next(iter(other_rank)) == ["1", "3"]
    with pytest.raises(ValueError, match="identity mismatch"):
        other_rank.load_state_dict(state)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    with path.open("r+b") as stream:
        stream.seek(512)
        stream.write(b"other")
    modified = WdsCursorBatches(spec, transform)
    with pytest.raises(ValueError, match="identity mismatch"):
        modified.load_state_dict(state)


def test_replay_is_bounded():
    engine = TrainEngine(ParaScaleConfig(max_resume_replay_batches=2))
    engine.state.data_state = {
        "resume_mode": "replay_skip",
        "consumed_micro_batches": 3,
    }
    with pytest.raises(RuntimeError, match="exceeding limit"):
        engine.data_resume.prepare_iterator(range(10))


def test_metrics_history_is_bounded():
    engine = TrainEngine(ParaScaleConfig(metrics_history_limit=3))
    for i in range(10):
        engine.record_step({"loss": i})
    assert len(engine.state.metrics_history) == 3
    assert engine.state.metrics_history[0]["loss"] == 7
    assert engine.state.metrics_history_dropped == 7
    assert engine.state.global_step == 10


def test_training_applies_retention_and_reports_truncated_history(tmp_path):
    pytest.importorskip("torch")
    from parascale.runtime.benchmark_runner import run_benchmark_from_config

    config = {
        "backend": {"training_backend": "native", "precision": "fp32"},
        "training": {
            "workload": "synthetic_regression",
            "max_steps": 6,
            "checkpoint_interval": 1,
            "checkpoint_keep_last": 2,
            "metrics_history_limit": 2,
            "warmup_steps": 3,
            "checkpoint_dir": str(tmp_path / "state"),
        },
        "runtime": {"device": "cpu", "run_dir": str(tmp_path / "run")},
    }
    benchmark = run_benchmark_from_config(config)
    assert benchmark["metrics"]["measured_steps"] == 2
    result = benchmark["train"]
    assert result["global_step"] == 6
    assert result["metrics_history_dropped"] == 4
    assert len(result["metrics_history"]) == 2
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == [
        "step-00000005",
        "step-00000006",
    ]
    assert CheckpointManager(str(tmp_path / "state")).validate(5).ok


def test_retention_preserves_unmanaged_and_valid_fallbacks(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    for step in range(6):
        _checkpoint(manager, step, managed=step != 0)
    manager.payload_path(5, "state.bin").write_bytes(b"corrupt")
    manager.prune(keep_last=2)
    assert manager.manifest_path(0).exists()
    assert manager.manifest_path(3).exists()
    assert manager.manifest_path(4).exists()
    assert manager.manifest_path(5).exists()
    assert not manager.manifest_path(1).exists()
    assert not manager.manifest_path(2).exists()


def _checkpoint(manager, step, *, managed=True):
    for name in ("state.bin", "runtime.bin"):
        path = manager.payload_path(step, name)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"valid")
    manager.write_manifest(
        CheckpointManifest(
            step=step,
            backend="native",
            files=[
                {"path": "state.bin", "role": "backend_state"},
                {"path": "runtime.bin", "role": "runtime_state", "rank": 0},
            ],
            metadata={
                "retention_managed": managed,
                "backend_state_written": True,
                "runtime_state_version": 1,
                "rank_count": 1,
                "world_size": 1,
            },
        )
    )


def test_retention_preserves_incoming_references_and_incomplete_fallbacks(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    for step in range(1, 5):
        _checkpoint(manager, step)
    incomplete = manager.read_manifest(4)
    incomplete.files = incomplete.files[:1]
    manager.write_manifest(incomplete)
    manager.write_manifest(
        CheckpointManifest(step=5, files=[{"path": "../step-00000001/state.bin"}])
    )
    assert manager.prune(keep_last=2) == []
    assert manager.validate(5).ok
    assert manager.manifest_path(2).exists()


def test_stability_warmup_accounts_for_discarded_metrics(tmp_path):
    from parascale.commands.stability_report import collect_stability_results

    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "metrics_history": [{"loss": 1.0}] * 2,
                "metrics_history_dropped": 10,
            }
        ),
        encoding="utf-8",
    )
    rows = collect_stability_results(tmp_path, warmup_steps=5)
    assert rows[0]["measured_steps"] == 2


def test_retention_preserves_linked_checkpoint_target(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    for step in range(1, 4):
        _checkpoint(manager, step)
    try:
        (tmp_path / "step-00000099").symlink_to(
            tmp_path / "step-00000001", target_is_directory=True
        )
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this host")
    assert manager.prune(keep_last=2) == []
    assert (tmp_path / "step-00000099/state.bin").read_bytes() == b"valid"


def test_matrix_uses_isolated_artifact_directories(tmp_path):
    from parascale.commands.scenario import build_matrix_config

    base = {"runtime": {"run_dir": "old_run"}}
    paths = []
    for backend in ("native_ddp", "fsdp"):
        config = build_matrix_config(
            scenario="clip-datacomp-golden",
            base_config=base,
            run_spec={"run_id": "new_run"},
            backend=backend,
            output_dir=tmp_path,
            max_steps=10,
            warmup_steps=1,
            batch_size=None,
            num_samples=None,
        )
        paths.append(config["runtime"]["run_dir"])
    assert len(set(paths)) == 2
    assert all("old_run" not in path for path in paths)
    assert base["runtime"]["run_dir"] == "old_run"
