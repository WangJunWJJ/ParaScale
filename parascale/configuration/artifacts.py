# -*- coding: utf-8 -*-
# @Time : 2026/6/27 上午11:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Persist auditable configuration artifacts for one runtime execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from .resolver import build_deepspeed_final_config, resolve_config


def write_config_artifacts(
    config_data: Dict[str, Any],
    run_dir: str | Path,
    *,
    cli_overrides: Dict[str, Any] | None = None,
    strategy_updates: Dict[str, Any] | None = None,
    emergency_overrides: Dict[str, Any] | None = None,
    dry_run: bool = False,
) -> Dict[str, str | None]:
    """Write the final resolved config and optional DeepSpeed config."""
    directory = Path(run_dir)
    resolved_path = directory / "config.resolved.json"
    deepspeed_path = directory / "backend.deepspeed.final.json"
    paths = {
        "run_dir": str(directory),
        "resolved_config": str(resolved_path),
        "deepspeed_final_config": None,
    }
    resolved = resolve_config(
        config_data,
        cli_overrides=cli_overrides,
        strategy_updates=strategy_updates,
        emergency_overrides=emergency_overrides,
        dry_run=dry_run,
    )
    if str(resolved.backend.get("training_backend")) == "deepspeed":
        paths["deepspeed_final_config"] = str(deepspeed_path)
    if _is_nonzero_distributed_rank():
        return paths

    directory.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(resolved_path, resolved.to_dict())
    if paths["deepspeed_final_config"]:
        _write_json_atomic(deepspeed_path, build_deepspeed_final_config(resolved))
    elif deepspeed_path.exists():
        deepspeed_path.unlink()
    return paths


def config_artifact_overrides(
    config_data: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Read provenance-aware patches carried by generated run configs."""
    metadata = config_data.get("_resolution", {})
    if not isinstance(metadata, Mapping):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key in ("cli_overrides", "strategy_updates", "emergency_overrides"):
        value = metadata.get(key, {})
        if isinstance(value, Mapping):
            result[key] = dict(value)
    return result


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_nonzero_distributed_rank() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    rank = int(os.environ.get("RANK", "0") or 0)
    return world_size > 1 and rank != 0
