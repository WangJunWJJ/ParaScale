# -*- coding: utf-8 -*-
# @Time : 2026/11/19 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Fault-injection contracts for checkpoint files and recovery selection."""

import json
from pathlib import Path

import pytest

from parascale.checkpoint import CheckpointManager, CheckpointManifest


def _checkpoint(manager, step):
    entries = []
    for name, role in (
        ("backend_state.pt", "backend_state"),
        ("rank-00000/runtime_state.pt", "runtime_state"),
    ):
        path = manager.payload_path(step, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"state-{step}-{role}".encode())
        entries.append(
            {
                "path": name,
                "role": role,
                **({"rank": 0} if role == "runtime_state" else {}),
            }
        )
    manager.write_manifest(
        CheckpointManifest(
            step=step,
            backend="native",
            files=entries,
            metadata={
                "rank_count": 1,
                "world_size": 1,
                "runtime_state_version": 1,
                "backend_state_written": True,
            },
        )
    )


def test_atomic_file_failure_preserves_published_bytes(tmp_path):
    from parascale.checkpoint.io import atomic_binary_writer

    path = tmp_path / "state.pt"
    path.write_bytes(b"published")
    with pytest.raises(OSError, match="disk full"):
        with atomic_binary_writer(path) as stream:
            stream.write(b"partial")
            raise OSError("disk full")
    assert path.read_bytes() == b"published"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.pt"]


def test_atomic_file_fsync_failure_does_not_replace(tmp_path, monkeypatch):
    from parascale.checkpoint import io

    path = tmp_path / "state.pt"
    path.write_bytes(b"published")
    monkeypatch.setattr(
        io.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("fsync failed"))
    )
    with pytest.raises(OSError, match="fsync failed"):
        with io.atomic_binary_writer(path) as stream:
            stream.write(b"new")
    assert path.read_bytes() == b"published"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.pt"]


def test_manifest_failure_preserves_previous_commit_record(tmp_path, monkeypatch):
    from parascale.checkpoint import io

    manager = CheckpointManager(str(tmp_path))
    path = manager.write_manifest(CheckpointManifest(step=1))
    before = path.read_bytes()
    monkeypatch.setattr(
        io.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("rename failed"))
    )
    with pytest.raises(OSError, match="rename failed"):
        manager.write_manifest(CheckpointManifest(step=1, metadata={"new": True}))
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*.tmp"))


def test_latest_training_checkpoint_reports_corrupt_and_partial_candidates(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    _checkpoint(manager, 2)
    _checkpoint(manager, 10)
    manager.payload_path(10, "backend_state.pt").write_bytes(b"corrupt")
    manager.manifest_path(11).parent.mkdir()
    manager.manifest_path(12).parent.mkdir()
    manager.manifest_path(12).write_text("{", encoding="utf-8")
    manifest, skipped = manager.latest_valid_training_checkpoint()
    assert manifest.step == 2
    assert [entry["step"] for entry in skipped] == [12, 11, 10]
    assert all(entry["reason"] for entry in skipped)


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "global_step",
        "no_backend",
        "missing_rank",
        "duplicate_rank",
        "missing_hash",
        "bypass",
        "escape",
        "null_metadata",
        "string_rank",
    ],
)
def test_latest_selection_rejects_incomplete_training_contract(tmp_path, fault):
    manager = CheckpointManager(str(tmp_path))
    _checkpoint(manager, 1)
    _checkpoint(manager, 2)
    path = manager.manifest_path(2)
    data = json.loads(path.read_text(encoding="utf-8"))
    if fault == "identity":
        data["step"] = 999
    elif fault == "global_step":
        data["global_step"] = 999
    elif fault == "no_backend":
        data["files"] = data["files"][1:]
    elif fault == "missing_rank":
        data["metadata"]["rank_count"] = 2
    elif fault == "duplicate_rank":
        data["files"].append(data["files"][1])
    elif fault == "missing_hash":
        data["files"][0].pop("sha256")
    elif fault == "bypass":
        data["metadata"]["allow_checkpoint_error_for_benchmark"] = True
    elif fault == "escape":
        data["files"][0]["path"] = "../step-00000001/backend_state.pt"
    elif fault == "null_metadata":
        data["metadata"] = None
    elif fault == "string_rank":
        data["files"][1]["rank"] = "0"
    path.write_text(json.dumps(data), encoding="utf-8")
    selected, skipped = manager.latest_valid_training_checkpoint()
    assert selected.step == 1
    assert skipped[0]["step"] == 2


def test_latest_selection_requires_an_actual_training_checkpoint(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    manager.write_manifest(CheckpointManifest(step=3))
    with pytest.raises(FileNotFoundError, match="valid.*training checkpoint"):
        manager.latest_valid_training_checkpoint()


def test_latest_selection_uses_numeric_step_order(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    _checkpoint(manager, 99999999)
    _checkpoint(manager, 100000000)
    selected, skipped = manager.latest_valid_training_checkpoint()
    assert selected.step == 100000000
    assert skipped == []


def test_atomic_file_success_syncs_then_replaces(tmp_path, monkeypatch):
    from parascale.checkpoint import io

    path = tmp_path / "child" / "state.pt"
    events = []
    original_replace = io.os.replace
    monkeypatch.setattr(io.os, "fsync", lambda _: events.append("fsync"))

    def replace(source, target):
        events.append("replace")
        original_replace(source, target)

    monkeypatch.setattr(io.os, "replace", replace)
    with io.atomic_binary_writer(path) as stream:
        stream.write(b"complete")
    assert path.read_bytes() == b"complete"
    assert events[:2] == ["fsync", "replace"]
    assert not list(Path(path.parent).glob(".*.tmp"))
