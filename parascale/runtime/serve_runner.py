# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:04
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Serving execution runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from parascale.checkpoint import CheckpointManager
from parascale.runtime.inference.engine import InferenceEngine
from parascale.runtime.runner_common import _section
from parascale.workloads import build_serving_model_from_checkpoint


def run_serve_from_config(
    config_data: Dict[str, Any], checkpoint: str | None = None
) -> Dict[str, Any]:
    serving = _section(config_data, "serving")
    checkpoint = checkpoint or serving.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "parascale serve requires --checkpoint or serving.checkpoint for real execution."
        )
    manager = _checkpoint_manager_for_path(checkpoint)
    manifest = manager.read_manifest_path(checkpoint)
    checkpoint_validation = manager.validate_manifest(manifest)
    if not checkpoint_validation.ok:
        raise RuntimeError(
            f"checkpoint validation failed: {checkpoint_validation.to_dict()}"
        )
    mock = bool(serving.get("mock", False))
    if mock:
        engine = (
            InferenceEngine(config=config_data)
            .initialize(world_size=1)
            .load_model(checkpoint=manifest, mock=True)
        )
        runtime_status = "mock"
        capability_level = "manifest_load_validation"
    else:
        model = build_serving_model_from_checkpoint(config_data, manifest, manager)
        engine = (
            InferenceEngine(config=config_data)
            .initialize(world_size=1)
            .load_model(model=model)
        )
        runtime_status = "real_local"
        capability_level = "local_tiny_torch_checkpoint"
    requests = serving.get("requests", ["hello"])
    result = engine.generate(requests)
    return {
        "mode": "serve",
        "dry_run": False,
        "runtime_status": runtime_status,
        "capability_level": capability_level,
        "mock": mock,
        "checkpoint": str(checkpoint),
        "checkpoint_validation": checkpoint_validation.to_dict(),
        "manifest": manifest.to_dict(),
        "result": result,
    }

def _checkpoint_manager_for_path(checkpoint: str | Path) -> CheckpointManager:
    path = Path(checkpoint)
    if path.is_file() and path.name == "manifest.json":
        return CheckpointManager(str(path.parent.parent))
    if path.is_dir() and path.name.startswith("step-"):
        return CheckpointManager(str(path.parent))
    return CheckpointManager(str(path))
