# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:58
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Adapter-only checkpoint helpers."""

from __future__ import annotations

from typing import Any, Dict


def adapter_state_dict(model: Any) -> Dict[str, Any] | None:
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    if hasattr(unwrapped, "adapter_state_dict"):
        return unwrapped.adapter_state_dict()
    return None


def load_adapter_state_dict(model: Any, state: Dict[str, Any]) -> bool:
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    if hasattr(unwrapped, "load_adapter_state_dict"):
        unwrapped.load_adapter_state_dict(state)
        return True
    return False


__all__ = ["adapter_state_dict", "load_adapter_state_dict"]
