# -*- coding: utf-8 -*-
# @Time : 2026/7/3 下午4:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Configuration validation and migration command handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parascale.commands.common import emit_json, load_config_file
from parascale.configuration import migrate_config_schema, validate_config_schema


def cmd_config_validate(args: argparse.Namespace) -> int:
    config = load_config_file(args.config)
    payload = {
        "mode": "config_validate",
        "config": str(args.config),
        "validation": validate_config_schema(config),
    }
    emit_json(payload, args.output)
    return 0


def cmd_config_migrate(args: argparse.Namespace) -> int:
    config = load_config_file(args.config)
    migrated = migrate_config_schema(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


__all__ = ["cmd_config_migrate", "cmd_config_validate"]
