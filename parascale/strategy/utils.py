# -*- coding: utf-8 -*-
# @Time : 2026/5/3 下午9:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Small torch-free helpers for strategy modules."""

from __future__ import annotations

from typing import Any


def get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def largest_divisor_at_most(value: int, limit: int) -> int:
    value = max(1, int(value))
    limit = max(1, min(int(limit), value))
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1
