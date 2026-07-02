# -*- coding: utf-8 -*-
# @Time : 2026/7/2 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Stable ParaScale CLI failure types and exit codes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

EXIT_CONFIG = 2
EXIT_DEPENDENCY = 3
EXIT_RUNTIME = 4
EXIT_CHECKPOINT = 5
EXIT_BENCHMARK = 6
EXIT_INTERNAL = 70


@dataclass
class CliFailure(Exception):
    """Expected command failure with a stable process exit code."""

    message: str
    error_type: str
    exit_code: int
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self, command: str | None) -> Dict[str, Any]:
        return {
            "status": "error",
            "error_type": self.error_type,
            "message": self.message,
            "exit_code": self.exit_code,
            "command": command,
            "details": dict(self.details),
        }


def config_failure(message: str, **details: Any) -> CliFailure:
    return CliFailure(message, "config_error", EXIT_CONFIG, details)


def dependency_failure(message: str, **details: Any) -> CliFailure:
    return CliFailure(message, "dependency_error", EXIT_DEPENDENCY, details)


def runtime_failure(message: str, **details: Any) -> CliFailure:
    return CliFailure(message, "runtime_error", EXIT_RUNTIME, details)


def checkpoint_failure(message: str, **details: Any) -> CliFailure:
    return CliFailure(message, "checkpoint_error", EXIT_CHECKPOINT, details)


def benchmark_failure(message: str, **details: Any) -> CliFailure:
    return CliFailure(message, "benchmark_error", EXIT_BENCHMARK, details)


def internal_failure(message: str, **details: Any) -> CliFailure:
    return CliFailure(message, "internal_error", EXIT_INTERNAL, details)


def classify_exception(exc: Exception) -> CliFailure:
    """Map an exception to the public CLI failure taxonomy."""

    if isinstance(exc, CliFailure):
        return exc
    details = {"exception": type(exc).__name__}
    if isinstance(exc, (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError)):
        return config_failure(str(exc), **details)
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return dependency_failure(str(exc), **details)
    if isinstance(exc, ValueError):
        return config_failure(str(exc), **details)
    if isinstance(exc, RuntimeError):
        return runtime_failure(str(exc), **details)
    return internal_failure(str(exc), **details)


__all__ = [
    "EXIT_BENCHMARK",
    "EXIT_CHECKPOINT",
    "EXIT_CONFIG",
    "EXIT_DEPENDENCY",
    "EXIT_INTERNAL",
    "EXIT_RUNTIME",
    "CliFailure",
    "benchmark_failure",
    "checkpoint_failure",
    "classify_exception",
    "config_failure",
    "dependency_failure",
    "internal_failure",
    "runtime_failure",
]
