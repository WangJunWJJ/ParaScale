# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free structural validation shared by stage loading and discovery."""

from __future__ import annotations

import json
import re
from dataclasses import fields
from typing import Any

from .checkpoint_context import canonical_checkpoint_context
from .distributed import DistributedParallelConfig


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous manifests instead of accepting the last duplicate key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Checkpoint manifest contains duplicate key: {key}")
        result[key] = value
    return result


def canonical_runtime_identity(identity: Any) -> str:
    """Validate a nonempty, finite JSON identity and return its canonical form."""
    serialized = json.dumps(identity, sort_keys=True, allow_nan=False)
    if (
        not isinstance(identity, dict)
        or not identity
        or json.loads(serialized) != identity
    ):
        raise ValueError("Checkpoint runtime identity must be a nonempty JSON object")
    return serialized


def validate_manifest_structure(metadata: Any) -> None:
    """Validate schema4/5/6 metadata, not payload contents or execution compatibility.

    Raises ValueError for malformed manifests. Rank files are self-contained,
    canonical local names; they cannot reference another checkpoint directory.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint manifest schema mismatch")
    schema = metadata.get("schema_version")
    if type(schema) is not int or schema not in (4, 5, 6):
        raise ValueError("Checkpoint topology/schema does not match execution")
    keys = {"schema_version", "topology", "step", "files", "context"}
    if schema >= 5:
        keys.add("runtime")
    if schema == 6:
        keys.add("storage")
    if set(metadata) != keys:
        raise ValueError("Checkpoint manifest schema mismatch")
    topology = metadata["topology"]
    try:
        if not isinstance(topology, dict) or set(topology) != {
            field.name for field in fields(DistributedParallelConfig)
        }:
            raise ValueError("Invalid topology fields")
        config = DistributedParallelConfig(**topology)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint topology/schema does not match execution"
        ) from error
    canonical_checkpoint_context(metadata["context"])
    if schema == 5:
        canonical_runtime_identity(metadata["runtime"])
    if type(metadata["step"]) is not int or metadata["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    entries = metadata["files"]
    if schema == 6:
        _validate_dcp(metadata, config)
        return
    if not isinstance(entries, list) or len(entries) != config.world_size:
        raise ValueError("Checkpoint rank count mismatch")
    for rank, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or entry["path"] != f"rank-{rank:05d}.pt"
            or not _valid_sha256(entry["sha256"])
        ):
            raise ValueError("Checkpoint rank entry schema mismatch")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_dcp(metadata: dict[str, Any], config: DistributedParallelConfig) -> None:
    if config.dp_size < 2 or config.tp_size != 1:
        raise ValueError("DCP checkpoint requires dp_size >= 2 and tp_size == 1")
    storage = metadata["storage"]
    if (
        not isinstance(storage, dict)
        or set(storage) != {"kind", "torch_version"}
        or storage["kind"] != "torch_dcp"
        or not isinstance(storage["torch_version"], str)
        or not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,127}", storage["torch_version"])
    ):
        raise ValueError("DCP checkpoint storage schema mismatch")
    if metadata["runtime"] is not None:
        try:
            canonical_runtime_identity(metadata["runtime"])
        except TypeError as error:
            raise ValueError("Invalid DCP checkpoint runtime identity") from error

    required = {f"rank-{rank:05d}.pt" for rank in range(config.world_size)}
    required.add("shared/.metadata")
    data = {f"shared/__{rank}_0.distcp" for rank in range(config.world_size)}
    entries = metadata["files"]
    if not isinstance(entries, list) or not (
        len(required) < len(entries) <= len(required) + len(data)
    ):
        raise ValueError("DCP checkpoint file count mismatch")
    paths = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "size_bytes"}
            or not isinstance(entry["path"], str)
            or not _valid_sha256(entry["sha256"])
            or type(entry["size_bytes"]) is not int
        ):
            raise ValueError("DCP checkpoint file entry schema mismatch")
        path = entry["path"]
        if path not in required and path not in data:
            raise ValueError(f"Invalid DCP checkpoint file path: {path!r}")
        if entry["size_bytes"] < (0 if path in data else 1):
            raise ValueError(f"Invalid DCP checkpoint file size: {path}")
        paths.append(path)
    if paths != sorted(set(paths)) or not required.issubset(paths):
        raise ValueError("DCP checkpoint paths must be unique, sorted and complete")
