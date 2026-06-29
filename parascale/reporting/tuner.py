# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:59
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Tuner evidence report helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def build_tuner_report(decisions: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(decision) for decision in decisions]
    return {
        "type": "tuner",
        "decisions": rows,
        "decision_count": len(rows),
    }


__all__ = ["build_tuner_report"]
