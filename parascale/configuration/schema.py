# -*- coding: utf-8 -*-
# @Time : 2026/7/3 下午4:47
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Versioned user configuration schema validation and migration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

CURRENT_CONFIG_SCHEMA_VERSION = 1
LEGACY_CONFIG_SCHEMA_VERSION = 0


def validate_config_schema(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the top-level schema version before runtime construction."""

    if not isinstance(config, dict):
        raise TypeError("ParaScale configuration must be a mapping.")
    version = config.get("schema_version", LEGACY_CONFIG_SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("config schema_version must be an integer.")
    if version < LEGACY_CONFIG_SCHEMA_VERSION:
        raise ValueError("config schema_version must be >= 0.")
    if version > CURRENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "config schema_version is newer than supported: "
            f"received={version}, supported={CURRENT_CONFIG_SCHEMA_VERSION}."
        )
    legacy = version == LEGACY_CONFIG_SCHEMA_VERSION
    return {
        "ok": True,
        "schema_version": version,
        "current_schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
        "legacy": legacy,
        "migration_required": version < CURRENT_CONFIG_SCHEMA_VERSION,
    }


def migrate_config_schema(
    config: Dict[str, Any],
    *,
    target_version: int = CURRENT_CONFIG_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Return a migrated copy without mutating the caller's configuration."""

    report = validate_config_schema(config)
    if target_version != CURRENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "unsupported config migration target: "
            f"target={target_version}, supported={CURRENT_CONFIG_SCHEMA_VERSION}."
        )
    if report["schema_version"] > target_version:
        raise ValueError("config downgrade is not supported.")
    migrated = deepcopy(config)
    migrated["schema_version"] = target_version
    return migrated


__all__ = [
    "CURRENT_CONFIG_SCHEMA_VERSION",
    "LEGACY_CONFIG_SCHEMA_VERSION",
    "migrate_config_schema",
    "validate_config_schema",
]
