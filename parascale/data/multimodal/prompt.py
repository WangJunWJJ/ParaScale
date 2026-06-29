# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:57
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Prompt template helpers for multimodal workloads."""

from __future__ import annotations

from typing import Mapping


def default_prompt_from_sample(sample: Mapping[str, object]) -> str:
    text = sample.get("text") or sample.get("caption") or ""
    return str(text)


__all__ = ["default_prompt_from_sample"]
