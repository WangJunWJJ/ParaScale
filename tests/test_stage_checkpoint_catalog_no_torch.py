# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Filesystem-only recovery discovery and conservative stage retention."""

import hashlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from parascale.parallel.distributed import DistributedParallelConfig


def checkpoint(root, step, *, family="a", runtime=True, ranks=1):
    directory = root / f"step-{step:08d}"
    directory.mkdir(parents=True)
    files = []
    for rank in range(ranks):
        payload = directory / f"rank-{rank:05d}.pt"
        payload.write_bytes(f"opaque-test-state:{family}:{step}:{rank}".encode())
        files.append(
            {
                "path": payload.name,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        )
    metadata = {
        "schema_version": 5 if runtime else 4,
        "topology": asdict(DistributedParallelConfig(dp_size=ranks)),
        "step": step,
        "files": files,
        "context": {
            "model": family,
            "loss": "test",
            "data": "test",
            "tokenizer": "none",
            "precision": "fp32",
        },
    }
    if runtime:
        metadata["runtime"] = {"kind": "test", "version": 1}
    (directory / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def catalog():
    from parascale.parallel import checkpoint_catalog

    return checkpoint_catalog


def dcp_checkpoint(root, step, *, version="2.6.0+cpu"):
    directory = checkpoint(root, step, ranks=2)
    manifest = directory / "manifest.json"
    metadata = json.loads(manifest.read_text())
    metadata.update(
        schema_version=6, storage={"kind": "torch_dcp", "torch_version": version}
    )
    (directory / "shared").mkdir()
    (directory / "shared/.metadata").write_bytes(b"metadata")
    (directory / "shared/__1_0.distcp").touch()
    for path in sorted(directory.glob("rank-*.pt")) + sorted(
        (directory / "shared").iterdir()
    ):
        payload = path.read_bytes()
        entry = {
            "path": path.relative_to(directory).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        if path.parent == directory:
            metadata["files"][int(path.stem[5:])] = entry
        else:
            metadata["files"].append(entry)
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_schema6_valid_with_sparse_empty_data_shard(tmp_path):
    directory = dcp_checkpoint(tmp_path, 1)
    assert catalog().inspect_stage_checkpoint(directory)["schema_version"] == 6


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(extra=True),
        lambda m: m.pop("storage"),
        lambda m: m["storage"].update(kind="other"),
        lambda m: m["storage"].update(torch_version="../2.6"),
        lambda m: m["storage"].update(torch_version=""),
        lambda m: m["runtime"].clear(),
        lambda m: m["topology"].update(dp_size=1),
        lambda m: m["topology"].update(tp_size=2),
        lambda m: m["files"].pop(0),
        lambda m: m["files"].pop(2),
        lambda m: m["files"].pop(),
        lambda m: m["files"].append(m["files"][-1]),
        lambda m: m["files"].reverse(),
        lambda m: m["files"][0].update(size_bytes=False),
        lambda m: m["files"][0].update(size_bytes=0),
        lambda m: m["files"][2].update(size_bytes=0),
        lambda m: m["files"][-1].update(size_bytes=-1),
        lambda m: m["files"][-1].update(path="shared/__2_0.distcp"),
        lambda m: m["files"][-1].update(path="shared/__01_0.distcp"),
        lambda m: m["files"][-1].update(path="shared\\__1_0.distcp"),
        lambda m: m["files"][-1].update(path="shared/../__1_0.distcp"),
        lambda m: m["files"][-1].update(sha256="x" * 64),
        lambda m: m["files"][-1].update(extra=True),
    ],
)
def test_schema6_rejects_malformed_structure(tmp_path, mutate):
    from parascale.parallel.checkpoint_manifest import validate_manifest_structure

    directory = dcp_checkpoint(tmp_path, 1)
    metadata = json.loads((directory / "manifest.json").read_text())
    with pytest.raises(ValueError):
        mutate(metadata)
        validate_manifest_structure(metadata)


def test_schema6_null_runtime_is_structural_only(tmp_path):
    from parascale.parallel.checkpoint_manifest import validate_manifest_structure

    directory = dcp_checkpoint(tmp_path, 1)
    manifest = directory / "manifest.json"
    metadata = json.loads(manifest.read_text())
    metadata["runtime"] = None
    validate_manifest_structure(metadata)
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime"):
        catalog().inspect_stage_checkpoint(directory)


@pytest.mark.parametrize(
    ("payload", "error"), [(b"x", "size"), (b"METADATA", "checksum")]
)
def test_schema6_corrupt_payload_preserved(tmp_path, payload, error):
    old = dcp_checkpoint(tmp_path, 1)
    target = old / "shared/.metadata"
    target.write_bytes(payload)
    dcp_checkpoint(tmp_path, 2)
    dcp_checkpoint(tmp_path, 3)
    anchor = dcp_checkpoint(tmp_path, 4)
    with pytest.raises(ValueError, match=error):
        catalog().inspect_stage_checkpoint(old)
    report = catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert report["protected"] == [4, 3]
    assert report["removed"] == [2]
    assert target.read_bytes() == payload


def test_schema6_retention_separates_formats_and_storage_versions(tmp_path):
    legacy = checkpoint(tmp_path, 6, ranks=2)
    other_version = dcp_checkpoint(tmp_path, 5, version="2.7.0")
    old = dcp_checkpoint(tmp_path, 1)
    dcp_checkpoint(tmp_path, 2)
    anchor = dcp_checkpoint(tmp_path, 3)
    report = catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert report["removed"] == [1]
    assert report["protected"] == [3, 2]
    assert not old.exists()
    assert legacy.is_dir() and other_version.is_dir()


@pytest.mark.parametrize("name", ["manifest.json", "shared/.metadata"])
def test_schema6_rejects_hardlinked_files(tmp_path, name):
    directory = dcp_checkpoint(tmp_path, 1)
    outside = tmp_path / "outside"
    outside.hardlink_to(directory / name)
    with pytest.raises(ValueError, match="hard links"):
        catalog().inspect_stage_checkpoint(directory)


@pytest.mark.parametrize(
    "failure", [None, "manifest_sync", "shared_sync", "shared_rmdir"]
)
def test_schema6_retention_sync_order_and_interruptions(tmp_path, monkeypatch, failure):
    old = dcp_checkpoint(tmp_path, 1)
    dcp_checkpoint(tmp_path, 2)
    anchor = dcp_checkpoint(tmp_path, 3)
    shared = old / "shared"
    module = catalog()
    inventory = module.inspect_stage_checkpoint(old)["files"]
    events = []
    original_unlink, original_rmdir = Path.unlink, Path.rmdir

    def record(action, path):
        events.append((action, path.relative_to(tmp_path).as_posix()))

    def unlink(path, *args, **kwargs):
        record("unlink", path)
        return original_unlink(path, *args, **kwargs)

    def fsync(path):
        record("sync", path)
        if (path == old and failure == "manifest_sync") or (
            path == shared and failure == "shared_sync"
        ):
            raise OSError("injected directory sync failure")

    def rmdir(path):
        record("rmdir", path)
        if path == shared and failure == "shared_rmdir":
            raise OSError("injected shared removal failure")
        return original_rmdir(path)

    expected = [("unlink", f"{old.name}/manifest.json"), ("sync", old.name)]
    if failure != "manifest_sync":
        expected += [("unlink", f"{old.name}/{entry['path']}") for entry in inventory]
        expected.append(("sync", f"{old.name}/shared"))
        if failure != "shared_sync":
            expected.append(("rmdir", f"{old.name}/shared"))
    if failure is None:
        expected += [("sync", old.name), ("rmdir", old.name), ("sync", ".")]

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", unlink)
        patch.setattr(Path, "rmdir", rmdir)
        patch.setattr(module, "_fsync_directory", fsync)
        if failure is None:
            assert module.prune_stage_checkpoints(anchor, keep_last=2)["removed"] == [1]
        else:
            with pytest.raises(OSError, match="injected"):
                module.prune_stage_checkpoints(anchor, keep_last=2)

    assert events == expected
    assert not (old / "manifest.json").exists()
    if failure is None:
        assert not old.exists()
        return
    assert shared.is_dir()
    if failure == "manifest_sync":
        assert all((old / entry["path"]).is_file() for entry in inventory)
    else:
        assert not list(shared.iterdir())
    anchor = dcp_checkpoint(tmp_path, 4)
    report = module.prune_stage_checkpoints(anchor, keep_last=2)
    assert report["removed"] == [2]
    assert report["protected"] == [4, 3]
    assert report["skipped"] == [
        {"step": 1, "path": str(old), "reason": "Uncommitted/orphan directory"}
    ]
    assert shared.is_dir() and not (old / "manifest.json").exists()


@pytest.mark.parametrize(
    "kind", ["extra_root", "extra_shared", "shared_link", "file_link"]
)
def test_schema6_retention_preserves_extra_content_and_links(tmp_path, kind):
    old = dcp_checkpoint(tmp_path, 1)
    dcp_checkpoint(tmp_path, 2)
    anchor = dcp_checkpoint(tmp_path, 3)
    shared = old / "shared"
    outside = tmp_path / "outside"
    if kind == "extra_root":
        (old / "extra").mkdir()
    elif kind == "extra_shared":
        (shared / "extra").mkdir()
    else:
        path = shared if kind == "shared_link" else shared / ".metadata"
        path.rename(outside)
        try:
            path.symlink_to(outside, target_is_directory=kind == "shared_link")
        except OSError:
            pytest.skip("Symlink creation unavailable")
        with pytest.raises(ValueError, match="link"):
            catalog().inspect_stage_checkpoint(old)
    report = catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert report["removed"] == []
    assert report["skipped"][0]["step"] == 1
    assert (old / "manifest.json").is_file()
    if kind.endswith("_link"):
        assert outside.exists()


def test_inspection_and_latest_without_loading_torch_payload(tmp_path):
    old = checkpoint(tmp_path, 1)
    latest = checkpoint(tmp_path, 2)
    assert catalog().inspect_stage_checkpoint(old)["step"] == 1
    selected, skipped = catalog().latest_stage_checkpoint(tmp_path)
    assert selected == latest
    assert skipped == []


@pytest.mark.parametrize(
    "damage", ["missing", "checksum", "json", "duplicate", "schema4", "step"]
)
def test_latest_reports_rejected_newer_checkpoint_and_uses_fallback(tmp_path, damage):
    old = checkpoint(tmp_path, 1)
    newest = checkpoint(tmp_path, 2, runtime=damage != "schema4")
    manifest = newest / "manifest.json"
    if damage == "missing":
        (newest / "rank-00000.pt").unlink()
    elif damage == "checksum":
        (newest / "rank-00000.pt").write_bytes(b"damaged")
    elif damage == "json":
        manifest.write_text("{", encoding="utf-8")
    elif damage == "duplicate":
        manifest.write_text(
            manifest.read_text().replace('"step": 2', '"step": 2, "step": 2'),
            encoding="utf-8",
        )
    elif damage == "step":
        metadata = json.loads(manifest.read_text())
        metadata["step"] = 3
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
    selected, skipped = catalog().latest_stage_checkpoint(tmp_path)
    assert selected == old
    assert len(skipped) == 1 and skipped[0]["step"] == 2 and skipped[0]["reason"]


def test_no_valid_checkpoint_fails_and_noncanonical_names_are_not_selected(tmp_path):
    checkpoint(tmp_path, 2).rename(tmp_path / "step-2")
    with pytest.raises(FileNotFoundError, match="No valid"):
        catalog().latest_stage_checkpoint(tmp_path)


def test_latest_does_not_hide_semantic_changes_as_integrity_failures(tmp_path):
    checkpoint(tmp_path, 1, family="old-model")
    newest = checkpoint(tmp_path, 2, family="different-model")
    assert catalog().latest_stage_checkpoint(tmp_path) == (newest, [])


def test_retention_keeps_two_verified_fallbacks_and_other_families(tmp_path):
    for step in range(1, 5):
        checkpoint(tmp_path, step)
    other = checkpoint(tmp_path, 5, family="other")
    report = catalog().prune_stage_checkpoints(tmp_path / "step-00000004", keep_last=2)
    assert report["removed"] == [2, 1]
    assert report["protected"] == [4, 3]
    assert other.is_dir()
    assert not (tmp_path / "step-00000001").exists()
    assert catalog().latest_stage_checkpoint(tmp_path)[0] == other


def test_corrupt_state_does_not_count_as_fallback_and_is_not_deleted(tmp_path):
    for step in range(1, 6):
        checkpoint(tmp_path, step)
    corrupt = tmp_path / "step-00000004" / "rank-00000.pt"
    corrupt.write_bytes(b"corrupt")
    report = catalog().prune_stage_checkpoints(tmp_path / "step-00000005", keep_last=2)
    assert report["removed"] == [2, 1]
    assert report["protected"] == [5, 3]
    assert corrupt.read_bytes() == b"corrupt"
    assert any(item["step"] == 4 for item in report["skipped"])


@pytest.mark.parametrize(
    "kind", ["extra_file", "extra_directory", "schema4", "bad_manifest"]
)
def test_retention_never_deletes_unknown_content(tmp_path, kind):
    old = checkpoint(tmp_path, 1, runtime=kind != "schema4")
    checkpoint(tmp_path, 2)
    anchor = checkpoint(tmp_path, 3)
    if kind == "extra_file":
        (old / "notes.txt").write_text("keep", encoding="utf-8")
    elif kind == "extra_directory":
        (old / "user-data").mkdir()
    elif kind == "bad_manifest":
        (old / "manifest.json").write_text("{", encoding="utf-8")
    report = catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert report["removed"] == []
    assert old.is_dir() and (old / "rank-00000.pt").is_file()
    assert report["skipped"]


@pytest.mark.parametrize("keep_last", [0, 1, True, 2.5])
def test_retention_rejects_invalid_fallback_count_before_deletion(tmp_path, keep_last):
    old = checkpoint(tmp_path, 1)
    anchor = checkpoint(tmp_path, 2)
    with pytest.raises(ValueError, match="keep_last"):
        catalog().prune_stage_checkpoints(anchor, keep_last=keep_last)
    assert old.is_dir()


def test_corrupt_anchor_and_stale_anchor_cannot_authorize_cleanup(tmp_path):
    old = checkpoint(tmp_path, 1)
    anchor = checkpoint(tmp_path, 2)
    checkpoint(tmp_path, 3)
    with pytest.raises(ValueError, match="newest"):
        catalog().prune_stage_checkpoints(anchor, keep_last=2)
    (anchor / "rank-00000.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert old.is_dir()


def test_retention_unpublishes_manifest_before_removing_known_files(
    tmp_path, monkeypatch
):
    old = checkpoint(tmp_path, 1)
    checkpoint(tmp_path, 2)
    anchor = checkpoint(tmp_path, 3)
    original = Path.unlink
    removed = []

    def unlink(path, *args, **kwargs):
        removed.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert catalog().prune_stage_checkpoints(anchor, keep_last=2)["removed"] == [1]
    assert removed == ["manifest.json", "rank-00000.pt"]
    assert not old.exists()


def test_payload_link_cannot_escape_checkpoint_directory(tmp_path):
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"preserve")
    old = checkpoint(tmp_path, 1)
    (old / "rank-00000.pt").unlink()
    try:
        (old / "rank-00000.pt").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    checkpoint(tmp_path, 2)
    anchor = checkpoint(tmp_path, 3)
    with pytest.raises(ValueError, match="link"):
        catalog().inspect_stage_checkpoint(old)
    assert catalog().prune_stage_checkpoints(anchor, keep_last=2)["removed"] == []
    assert outside.read_bytes() == b"preserve"


def test_catalog_import_does_not_require_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Catalog must not import torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from parascale.parallel.checkpoint_catalog import inspect_stage_checkpoint
""",
        ],
        check=True,
    )


def test_unknown_manifest_blocks_all_deletion_before_any_unlink(tmp_path):
    unknown = checkpoint(tmp_path, 1)
    (unknown / "manifest.json").write_text("{}", encoding="utf-8")
    for step in range(2, 6):
        checkpoint(tmp_path, step)
    report = catalog().prune_stage_checkpoints(tmp_path / "step-00000005", keep_last=2)
    assert report["removed"] == []
    assert all(
        (tmp_path / f"step-{step:08d}" / "manifest.json").is_file()
        for step in range(1, 6)
    )


def test_newer_corrupt_payload_still_makes_anchor_stale(tmp_path):
    checkpoint(tmp_path, 1)
    anchor = checkpoint(tmp_path, 2)
    newest = checkpoint(tmp_path, 3)
    (newest / "rank-00000.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="newest"):
        catalog().prune_stage_checkpoints(anchor, keep_last=2)


def test_retention_family_comparison_distinguishes_bool_from_integer(tmp_path):
    old = checkpoint(tmp_path, 1)
    manifest = old / "manifest.json"
    metadata = json.loads(manifest.read_text())
    metadata["runtime"]["version"] = True
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    checkpoint(tmp_path, 2)
    checkpoint(tmp_path, 3)
    anchor = checkpoint(tmp_path, 4)
    report = catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert report["removed"] == [2]
    assert old.is_dir()


def test_interrupted_deletion_leaves_orphan_but_does_not_block_next_prune(
    tmp_path, monkeypatch
):
    old = checkpoint(tmp_path, 1)
    checkpoint(tmp_path, 2)
    anchor = checkpoint(tmp_path, 3)
    original = Path.unlink

    def fail_payload(path, *args, **kwargs):
        if path.name == "rank-00000.pt":
            raise OSError("injected interruption")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_payload)
    with pytest.raises(OSError, match="interruption"):
        catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert not (old / "manifest.json").exists()
    assert (old / "rank-00000.pt").is_file()
    monkeypatch.setattr(Path, "unlink", original)
    anchor = checkpoint(tmp_path, 4)
    report = catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert report["removed"] == [2]
    assert any(item["step"] == 1 for item in report["skipped"])
    assert old.is_dir()


@pytest.mark.parametrize("kind", ["ancestor", "directory", "manifest"])
def test_linked_checkpoint_paths_are_rejected_before_resolution(tmp_path, kind):
    root = tmp_path / "real"
    old = checkpoint(root, 1)
    target = tmp_path / "linked"
    try:
        if kind == "ancestor":
            target.symlink_to(root, target_is_directory=True)
            candidate = target / old.name
        elif kind == "directory":
            target.symlink_to(old, target_is_directory=True)
            candidate = target
        else:
            manifest = old / "manifest.json"
            manifest.rename(target)
            manifest.symlink_to(target)
            candidate = old
    except OSError:
        pytest.skip("Symlink creation unavailable")
    with pytest.raises(ValueError, match="link"):
        catalog().inspect_stage_checkpoint(candidate)


def test_anchor_payload_is_hashed_once(tmp_path, monkeypatch):
    checkpoint(tmp_path, 1)
    anchor = checkpoint(tmp_path, 2)
    module = catalog()
    original = module._digest
    paths = []

    def digest(path):
        paths.append(path)
        return original(path)

    monkeypatch.setattr(module, "_digest", digest)
    module.prune_stage_checkpoints(anchor, keep_last=2)
    assert paths.count(anchor / "rank-00000.pt") == 1


@pytest.mark.parametrize("ranks", [2, 4])
def test_rank_payloads_are_verified_concurrently(tmp_path, monkeypatch, ranks):
    anchor = checkpoint(tmp_path, 1, ranks=ranks)
    module = catalog()
    original = module._digest
    barrier = threading.Barrier(ranks, timeout=10)
    lock = threading.Lock()
    visited = []

    def digest(path):
        with lock:
            visited.append(path.name)
        barrier.wait()
        return original(path)

    monkeypatch.setattr(module, "_digest", digest)
    assert module.inspect_stage_checkpoint(anchor)["step"] == 1
    assert sorted(visited) == [f"rank-{rank:05d}.pt" for rank in range(ranks)]


def test_verification_drains_workers_and_preserves_error_order(tmp_path, monkeypatch):
    anchor = checkpoint(tmp_path, 1, ranks=4)
    module = catalog()
    original = module._digest
    barrier = threading.Barrier(4, timeout=10)
    lock = threading.Lock()
    finished = []

    def digest(path):
        barrier.wait()
        try:
            if path.name == "rank-00001.pt":
                time.sleep(0.04)
                return "incorrect"
            if path.name == "rank-00003.pt":
                return "incorrect"
            time.sleep(0.08)
            return original(path)
        finally:
            with lock:
                finished.append(path.name)

    monkeypatch.setattr(module, "_digest", digest)
    with pytest.raises(ValueError, match="rank-00001.pt"):
        module.inspect_stage_checkpoint(anchor)
    assert len(finished) == 4


def test_single_payload_verification_does_not_create_workers(tmp_path, monkeypatch):
    anchor = checkpoint(tmp_path, 1)
    module = catalog()

    def forbidden(*args, **kwargs):
        raise AssertionError("Single-file verification should stay inline")

    monkeypatch.setattr(module, "ThreadPoolExecutor", forbidden, raising=False)
    assert module.inspect_stage_checkpoint(anchor)["step"] == 1


def test_corrupt_multirank_anchor_never_prunes_older_snapshots(tmp_path):
    old = checkpoint(tmp_path, 1, ranks=4)
    checkpoint(tmp_path, 2, ranks=4)
    anchor = checkpoint(tmp_path, 3, ranks=4)
    (anchor / "rank-00002.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="rank-00002.pt"):
        catalog().prune_stage_checkpoints(anchor, keep_last=2)
    assert old.is_dir()


def test_parent_traversal_is_rejected_before_hash_or_unlink(tmp_path, monkeypatch):
    anchor = checkpoint(tmp_path, 3)
    child = tmp_path / "child"
    child.mkdir()
    module = catalog()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected filesystem operation")

    monkeypatch.setattr(module, "_digest", forbidden)
    monkeypatch.setattr(Path, "unlink", forbidden)
    with pytest.raises(ValueError, match="parent|traversal"):
        module.prune_stage_checkpoints(child / ".." / anchor.name, keep_last=2)


def test_reparse_attribute_is_rejected_without_is_junction(tmp_path, monkeypatch):
    anchor = checkpoint(tmp_path, 1)
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == tmp_path:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        catalog().inspect_stage_checkpoint(anchor)


@pytest.mark.parametrize("fail_sync", [False, True])
def test_manifest_unlink_is_synced_before_payload_deletion(
    tmp_path, monkeypatch, fail_sync
):
    old = checkpoint(tmp_path, 1)
    checkpoint(tmp_path, 2)
    anchor = checkpoint(tmp_path, 3)
    module = catalog()
    events = []
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        events.append(("unlink", path.name))
        return original(path, *args, **kwargs)

    def fsync(path):
        events.append(("sync", path.name))
        if fail_sync:
            raise OSError("injected sync failure")

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(module, "_fsync_directory", fsync)
    if fail_sync:
        with pytest.raises(OSError, match="sync failure"):
            module.prune_stage_checkpoints(anchor, keep_last=2)
        assert (old / "rank-00000.pt").is_file()
        assert events == [("unlink", "manifest.json"), ("sync", old.name)]
    else:
        module.prune_stage_checkpoints(anchor, keep_last=2)
        assert events == [
            ("unlink", "manifest.json"),
            ("sync", old.name),
            ("unlink", "rank-00000.pt"),
            ("sync", tmp_path.name),
        ]
