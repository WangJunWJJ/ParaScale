# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午6:02
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Checkpoint command payload builders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from parascale.checkpoint import CheckpointManager
from parascale.commands.common import emit_json


def checkpoint_manager_for_path(checkpoint: str | Path) -> CheckpointManager:
    path = Path(checkpoint)
    if path.is_file() and path.name == "manifest.json":
        return CheckpointManager(str(path.parent.parent))
    if path.is_dir() and path.name.startswith("step-"):
        return CheckpointManager(str(path.parent))
    return CheckpointManager(str(path))


def build_checkpoint_validation_payload(checkpoint: str) -> Dict[str, Any]:
    manager = checkpoint_manager_for_path(checkpoint)
    manifest = manager.read_manifest_path(checkpoint)
    validation = manager.validate_manifest(manifest)
    return {
        "mode": "checkpoint_validate",
        "runtime_status": "diagnostic",
        "checkpoint": str(checkpoint),
        "manifest": manifest.to_dict(),
        "validation": validation.to_dict(),
    }


def cmd_checkpoint_validate(args: argparse.Namespace) -> int:
    payload = build_checkpoint_validation_payload(args.checkpoint)
    emit_json(payload, args.output)
    return 0
