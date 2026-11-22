# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free structural validation shared by stage loading and discovery."""

from __future__ import annotations

import json
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
    """Validate schema4/5 metadata, not payload contents or execution compatibility.

    Raises ValueError for malformed manifests. Rank files are self-contained,
    canonical local names; they cannot reference another checkpoint directory.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint manifest schema mismatch")
    schema = metadata.get("schema_version")
    if type(schema) is not int or schema not in (4, 5):
        raise ValueError("Checkpoint topology/schema does not match execution")
    keys = {"schema_version", "topology", "step", "files", "context"}
    if schema == 5:
        keys.add("runtime")
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
    if not isinstance(entries, list) or len(entries) != config.world_size:
        raise ValueError("Checkpoint rank count mismatch")
    for rank, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or entry["path"] != f"rank-{rank:05d}.pt"
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in entry["sha256"])
        ):
            raise ValueError("Checkpoint rank entry schema mismatch")
