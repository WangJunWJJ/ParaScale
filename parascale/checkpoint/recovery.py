# -*- coding: utf-8 -*-
# @Time : 2026/11/19 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Strict, opt-in selection of complete native training checkpoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parascale.checkpoint.manager import CheckpointManager, CheckpointManifest


def latest_valid_training_checkpoint(
    manager: CheckpointManager,
) -> tuple[CheckpointManifest, list[dict[str, Any]]]:
    """Return the newest intact native training record and rejected candidates.

    File and manifest validation is structural, not a model compatibility check.
    Raises FileNotFoundError if every candidate is missing or invalid.
    """
    candidates = []
    for directory in Path(manager.root).glob("step-*"):
        match = re.fullmatch(r"step-(\d+)", directory.name)
        if directory.is_dir() and match:
            step = int(match[1])
            if directory.name == f"step-{step:08d}":
                candidates.append(step)
    skipped = []
    for step in sorted(candidates, reverse=True):
        try:
            manifest = manager.read_manifest(step)
            _validate_training_contract(manager, manifest, step)
            report = manager.validate_manifest(manifest)
            if not report.ok:
                raise ValueError(f"checkpoint integrity failed: {report.to_dict()}")
            return manifest, skipped
        except (OSError, ValueError, TypeError, KeyError) as exc:
            skipped.append(
                {
                    "step": step,
                    "path": str(manager.manifest_path(step)),
                    "reason": str(exc),
                }
            )
    raise FileNotFoundError(
        f"No valid native training checkpoint in {manager.root}; skipped={skipped}"
    )


def _validate_training_contract(
    manager: CheckpointManager,
    manifest: CheckpointManifest,
    step: int,
) -> None:
    if manifest.step != step or manifest.global_step != step:
        raise ValueError("checkpoint step does not match its directory")
    if manifest.format != "parascale_manifest_v1" or manifest.schema_version != 2:
        raise ValueError("unsupported training checkpoint schema")
    if manifest.backend not in {"native", "native_ddp"}:
        raise ValueError("latest recovery requires a native/native_ddp checkpoint")
    metadata = manifest.metadata
    if (
        metadata.get("backend_state_written") is not True
        or metadata.get("runtime_state_version") != 1
    ):
        raise ValueError("backend or runtime state is incomplete")
    if any(
        metadata.get(key)
        for key in (
            "allow_checkpoint_error_for_benchmark",
            "skip_final_checkpoint",
            "backend_checkpoint_error",
        )
    ):
        raise ValueError("benchmark checkpoint bypass is not eligible for recovery")
    ranks = int(metadata.get("rank_count", 0))
    if ranks < 1 or ranks != int(metadata.get("world_size", 0)):
        raise ValueError("invalid checkpoint rank count")
    backend = [
        entry for entry in manifest.files if entry.get("role") == "backend_state"
    ]
    runtime = [
        entry for entry in manifest.files if entry.get("role") == "runtime_state"
    ]
    runtime_ranks = [entry.get("rank") for entry in runtime]
    if (
        len(backend) != 1
        or any(type(rank) is not int for rank in runtime_ranks)
        or sorted(runtime_ranks) != list(range(ranks))
    ):
        raise ValueError(
            "backend state and runtime state must cover every rank exactly once"
        )
    root = manager.manifest_path(step).parent.resolve()
    paths = set()
    for entry in manifest.files:
        path = Path(str(entry.get("path", "")))
        resolved = (root / path).resolve()
        if path.is_absolute() or not resolved.is_relative_to(root) or resolved in paths:
            raise ValueError(
                "checkpoint payload paths must be unique and inside the step directory"
            )
        paths.add(resolved)
        if (
            entry.get("error")
            or not resolved.is_file()
            or entry.get("entry_type") != "file"
        ):
            raise ValueError("checkpoint payload must be a complete file")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
            or int(entry.get("size_bytes", -1)) < 0
        ):
            raise ValueError("checkpoint payload requires size and SHA-256 metadata")
