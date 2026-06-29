# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:26
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inference sampling configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
