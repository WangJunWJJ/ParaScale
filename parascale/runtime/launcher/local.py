# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Local launcher helpers."""

from __future__ import annotations

from typing import Any


def local_command(entrypoint_args: list[str], config_path: str) -> list[str]:
    return [*entrypoint_args, "--config", config_path]


def is_local_context(context: Any) -> bool:
    return max(1, int(getattr(context, "world_size", 1) or 1)) == 1


__all__ = ["is_local_context", "local_command"]
