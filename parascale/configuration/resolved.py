# -*- coding: utf-8 -*-
# @Time : 2026/6/26 下午12:07
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Frozen resolved configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class ConfigIssue:
    code: str
    message: str
    path: str = ""
    severity: str = "warning"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ResolvedField:
    path: str
    value: Any
    source: str
    overridden_by: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "value": self.value,
            "source": self.source,
            "overridden_by": list(self.overridden_by),
            "history": [dict(item) for item in self.history],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ResolvedConfig:
    runtime: Mapping[str, Any]
    backend: Mapping[str, Any]
    workload: Mapping[str, Any]
    data: Mapping[str, Any]
    training: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    hardware: Mapping[str, Any]
    fields: Mapping[str, ResolvedField]
    warnings: List[ConfigIssue] = field(default_factory=list)
    errors: List[ConfigIssue] = field(default_factory=list)

    def field(self, path: str) -> ResolvedField:
        return self.fields[path]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime": dict(self.runtime),
            "backend": dict(self.backend),
            "workload": dict(self.workload),
            "data": dict(self.data),
            "training": dict(self.training),
            "optimizer": dict(self.optimizer),
            "hardware": dict(self.hardware),
            "fields": {
                path: field.to_dict() for path, field in sorted(self.fields.items())
            },
            "warnings": [issue.to_dict() for issue in self.warnings],
            "errors": [issue.to_dict() for issue in self.errors],
        }
