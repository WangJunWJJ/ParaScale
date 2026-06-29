# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:26
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Serving request and response schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ServeRequest:
    request_id: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServeResponse:
    request_id: str
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
