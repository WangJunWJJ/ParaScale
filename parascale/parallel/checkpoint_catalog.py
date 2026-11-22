# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Integrity discovery and conservative retention for complete stage snapshots.

Use a trusted, dedicated root with one writer and no concurrent mutations.
Integrity does not imply semantic recoverability: the all-rank loader must still
validate the payload and effective model/data/runtime contract before applying it.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from .checkpoint_manifest import unique_json_object, validate_manifest_structure
from .distributed_checkpoint import _digest, _fsync_directory


def _safe_path(path: Path) -> Path:
    """Check links/reparse points before resolution, including all ancestors."""
    if ".." in path.parts:
        raise ValueError("Checkpoint parent traversal is not allowed")
    path = Path(os.path.abspath(path))
    for current in (path, *path.parents):
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & (
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError(f"Checkpoint link/reparse path is not allowed: {current}")
    return path


def _step(path: Path) -> int:
    if not re.fullmatch(r"step-[0-9]{8,}", path.name):
        raise ValueError(f"Noncanonical checkpoint directory: {path.name}")
    step = int(path.name[5:])
    if path.name != f"step-{step:08d}":
        raise ValueError(f"Noncanonical checkpoint directory: {path.name}")
    return step


def _read_metadata(directory: Path) -> dict[str, Any]:
    directory = _safe_path(directory)
    step = _step(directory)
    manifest = _safe_path(directory / "manifest.json")
    if not manifest.is_file():
        raise ValueError("Checkpoint manifest must be a regular file")
    metadata = json.loads(
        manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_json_object
    )
    validate_manifest_structure(metadata)
    if metadata["schema_version"] != 5:
        raise ValueError("Stage catalog requires schema5 complete runtime checkpoints")
    if metadata["step"] != step:
        raise ValueError("Checkpoint directory step does not match manifest step")
    return metadata


def _check_payloads(directory: Path, metadata: dict[str, Any]) -> None:
    for entry in metadata["files"]:
        path = _safe_path(directory / entry["path"])
        if not path.is_file():
            raise ValueError(f"Checkpoint payload must be a regular file: {path}")
        if _digest(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path.name}")


def inspect_stage_checkpoint(directory: str | Path) -> dict[str, Any]:
    """Return schema5 metadata after validating local paths and all SHA256 hashes.

    Does not import Torch or deserialize payloads. Raises ValueError for invalid
    metadata, links or checksum mismatches, and OSError for filesystem failures.
    """
    directory = _safe_path(Path(directory))
    metadata = _read_metadata(directory)
    _check_payloads(directory, metadata)
    return metadata


def _candidates(root: Path) -> list[Path]:
    root = _safe_path(root)
    candidates = [path for path in root.iterdir() if path.name.startswith("step-")]

    def order(path: Path) -> int:
        try:
            return _step(path)
        except ValueError:
            return -1

    return sorted(candidates, key=order, reverse=True)


def _skip(path: Path, reason: str) -> dict[str, Any]:
    try:
        step = _step(path)
    except ValueError:
        step = None
    return {"step": step, "path": str(path), "reason": reason}


def latest_stage_checkpoint(root: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Find the newest structurally intact snapshot, reporting newer failures.

    Semantic incompatibility must fail in the loader, not silently select an
    older model. Raises FileNotFoundError when no intact snapshot is present.
    """
    skipped = []
    for directory in _candidates(Path(root)):
        try:
            inspect_stage_checkpoint(directory)
        except (OSError, ValueError) as error:
            skipped.append(_skip(directory, str(error)))
        else:
            return directory, skipped
    raise FileNotFoundError(f"No valid stage checkpoint in {root}: {skipped}")


def _family(metadata: dict[str, Any]) -> str:
    return json.dumps(
        {key: metadata[key] for key in ("topology", "context", "runtime")},
        sort_keys=True,
        allow_nan=False,
    )


def _remove_snapshot(directory: Path, metadata: dict[str, Any], root: Path) -> bool:
    """Unpublish then unlink only declared files; never recursively delete."""
    directory = _safe_path(directory)
    if directory.parent != root or _read_metadata(directory) != metadata:
        raise ValueError("Checkpoint changed before retention")
    names = {"manifest.json", *(entry["path"] for entry in metadata["files"])}
    if {path.name for path in directory.iterdir()} != names:
        return False
    paths = [_safe_path(directory / name) for name in sorted(names)]
    if any(not path.is_file() or path.parent != directory for path in paths):
        raise ValueError("Checkpoint contains unsafe retention paths")
    (directory / "manifest.json").unlink()
    _fsync_directory(directory)
    for entry in metadata["files"]:
        (directory / entry["path"]).unlink()
    directory.rmdir()
    _fsync_directory(root)
    return True


def prune_stage_checkpoints(anchor: str | Path, *, keep_last: int) -> dict[str, Any]:
    """Retain at least two checksum-verified snapshots in the anchor's family.

    keep_last includes the newest anchor and at least one older intact copy.
    Invalid/unknown manifests block all pruning; corrupt payloads, other families,
    extra content and manifest-less orphans are preserved and reported. Raises
    ValueError for a corrupt/stale anchor or invalid count, and OSError on I/O
    failure (including interrupted deletion). Invoke only after all-rank save.
    """
    if type(keep_last) is not int or keep_last < 2:
        raise ValueError("keep_last must be an integer >= 2")
    anchor = _safe_path(Path(anchor))
    metadata = inspect_stage_checkpoint(anchor)
    root, identity = anchor.parent, _family(metadata)
    report: dict[str, Any] = {"removed": [], "protected": [], "skipped": []}
    candidates, blocked = [], False
    # Validate every manifest before any deletion: unknown formats may refer to
    # files outside their own directory, unlike this self-contained stage schema.
    for directory in _candidates(root):
        try:
            _safe_path(directory)
            if (
                not (directory / "manifest.json").exists()
                and not (directory / "manifest.json").is_symlink()
            ):
                report["skipped"].append(
                    _skip(directory, "Uncommitted/orphan directory")
                )
                continue
            candidate = metadata if directory == anchor else _read_metadata(directory)
        except (OSError, ValueError) as error:
            blocked = True
            report["skipped"].append(_skip(directory, str(error)))
            continue
        if _family(candidate) != identity:
            report["skipped"].append(_skip(directory, "Different training family"))
        elif candidate["step"] > metadata["step"]:
            raise ValueError(
                "Retention anchor must be the newest checkpoint in its family"
            )
        else:
            candidates.append((directory, candidate))
    if blocked:
        return report
    verified = []
    for directory, candidate in candidates:
        try:
            if directory != anchor:
                _check_payloads(directory, candidate)
        except (OSError, ValueError) as error:
            report["skipped"].append(_skip(directory, str(error)))
        else:
            verified.append((directory, candidate))
    report["protected"] = [item[1]["step"] for item in verified[:keep_last]]
    for directory, candidate in verified[keep_last:]:
        if _remove_snapshot(directory, candidate, root):
            report["removed"].append(candidate["step"])
        else:
            report["skipped"].append(_skip(directory, "Extra content; retained"))
    return report
