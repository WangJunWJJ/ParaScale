# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Common CLI helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(text: str) -> Dict[str, Any]:
    if any(line.lstrip().startswith("- ") for line in text.splitlines()):
        raise RuntimeError(
            "Complex YAML lists require PyYAML. Install PyYAML>=6.0 or use JSON config."
        )
    data: Dict[str, Any] = {}
    current: Dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith(" ") and current is not None:
            key, _, value = line.strip().partition(":")
            current[key.strip()] = parse_scalar(value)
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not value:
            current = {}
            data[key] = current
        else:
            data[key] = parse_scalar(value)
            current = None
    return data


def load_config_file(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8-sig")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ImportError:
        return load_simple_yaml(text)


def emit_json(payload: Dict[str, Any], output_path: str | None = None) -> None:
    if _is_nonzero_distributed_rank():
        return
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_path:
        Path(output_path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def emit_error_json(payload: Dict[str, Any]) -> None:
    """Write one machine-readable command error to stderr on rank zero."""

    if _is_nonzero_distributed_rank():
        return
    sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _is_nonzero_distributed_rank() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    rank = int(os.environ.get("RANK", "0") or 0)
    return world_size > 1 and rank != 0
