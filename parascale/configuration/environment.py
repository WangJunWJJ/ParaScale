# -*- coding: utf-8 -*-
# @Time : 2026/7/2 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Expand explicit environment references in user configuration values."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

_ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_environment_references(value: Any) -> Any:
    """Recursively expand ``${NAME}`` references or reject missing variables."""
    if isinstance(value, str):
        missing = sorted(
            {
                name
                for name in _ENVIRONMENT_REFERENCE.findall(value)
                if name not in os.environ
            }
        )
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Undefined environment variable(s) in config: {names}")
        return _ENVIRONMENT_REFERENCE.sub(
            lambda match: os.environ[match.group(1)], value
        )
    if isinstance(value, Mapping):
        return {
            key: expand_environment_references(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_environment_references(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment_references(item) for item in value)
    return value


__all__ = ["expand_environment_references"]
