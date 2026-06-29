# -*- coding: utf-8 -*-
# @Time : 2026/6/26 下午12:07
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Auditable configuration resolution layer."""

from .artifacts import config_artifact_overrides, write_config_artifacts
from .resolved import ConfigIssue, ResolvedConfig, ResolvedField
from .resolver import build_deepspeed_final_config, resolve_config

__all__ = [
    "ConfigIssue",
    "ResolvedConfig",
    "ResolvedField",
    "build_deepspeed_final_config",
    "config_artifact_overrides",
    "resolve_config",
    "write_config_artifacts",
]
