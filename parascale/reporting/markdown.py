# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:59
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Markdown report helpers."""

from __future__ import annotations

from typing import Any, Mapping


def key_value_markdown(title: str, payload: Mapping[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key in sorted(payload):
        lines.append(f"- `{key}`: {payload[key]}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["key_value_markdown"]
