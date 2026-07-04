# -*- coding: utf-8 -*-
# @Time : 2026/6/26 下午12:07
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Auditable configuration resolution layer."""

from .artifacts import config_artifact_overrides, write_config_artifacts
from .environment import expand_environment_references
from .resolved import ConfigIssue, ResolvedConfig, ResolvedField
from .resolver import build_deepspeed_final_config, resolve_config
from .schema import (
    CURRENT_CONFIG_SCHEMA_VERSION,
    LEGACY_CONFIG_SCHEMA_VERSION,
    migrate_config_schema,
    validate_config_schema,
)

__all__ = [
    "ConfigIssue",
    "CURRENT_CONFIG_SCHEMA_VERSION",
    "LEGACY_CONFIG_SCHEMA_VERSION",
    "ResolvedConfig",
    "ResolvedField",
    "build_deepspeed_final_config",
    "config_artifact_overrides",
    "expand_environment_references",
    "migrate_config_schema",
    "resolve_config",
    "validate_config_schema",
    "write_config_artifacts",
]
