# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:59
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Profile report helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def build_profile_report(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "profile",
        "metrics": dict(metrics),
        "has_dataloader_wait": "dataloader_wait_ms" in metrics,
        "has_peak_memory": "peak_memory_bytes" in metrics,
    }


__all__ = ["build_profile_report"]
