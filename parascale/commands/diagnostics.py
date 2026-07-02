# -*- coding: utf-8 -*-
# @Time : 2026/7/2 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pure runtime requirement evaluation for the doctor command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

CheckFunction = Callable[[Dict[str, Any]], bool]


@dataclass(frozen=True)
class DiagnosticCheck:
    """Result of evaluating one requested runtime capability."""

    name: str
    ok: bool
    message: str
    evidence: Dict[str, Any]
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "required": self.required,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """Ordered collection of requested runtime capability checks."""

    requirements: tuple[str, ...]
    checks: tuple[DiagnosticCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.ok)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "requirements": list(self.requirements),
            "checks": {check.name: check.to_dict() for check in self.checks},
        }


def _python_version(payload: Dict[str, Any]) -> tuple[int, int]:
    text = str(payload.get("python", {}).get("version", "0.0"))
    parts = text.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def _core(payload: Dict[str, Any]) -> bool:
    return _python_version(payload) >= (3, 10) and bool(
        payload.get("dependencies", {}).get("yaml")
    )


def _torch(payload: Dict[str, Any]) -> bool:
    runtime = payload.get("torch_runtime", {})
    return (
        bool(payload.get("dependencies", {}).get("torch"))
        and bool(runtime.get("available"))
        and not runtime.get("error")
    )


def _distributed(payload: Dict[str, Any]) -> bool:
    return bool(payload.get("distributed_runtime", {}).get("available"))


def _cuda(payload: Dict[str, Any]) -> bool:
    runtime = payload.get("torch_runtime", {})
    return bool(runtime.get("cuda_available")) and int(
        runtime.get("cuda_device_count", 0)
    ) > 0


def _deepspeed(payload: Dict[str, Any]) -> bool:
    runtime = payload.get("deepspeed_runtime", {})
    return (
        bool(payload.get("dependencies", {}).get("deepspeed"))
        and bool(runtime.get("available"))
        and not runtime.get("error")
    )


def _npu(payload: Dict[str, Any]) -> bool:
    dependencies = payload.get("dependencies", {})
    runtime = payload.get("ascend_runtime", {})
    return (
        bool(dependencies.get("torch_npu"))
        and bool(runtime.get("available"))
        and int(runtime.get("device_count", 0)) > 0
    )


CHECKS: Dict[str, CheckFunction] = {
    "core": _core,
    "torch": _torch,
    "distributed": _distributed,
    "cuda": _cuda,
    "deepspeed": _deepspeed,
    "npu": _npu,
}


def _evidence(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if name == "core":
        return {
            "python_version": payload.get("python", {}).get("version"),
            "yaml": payload.get("dependencies", {}).get("yaml", False),
        }
    if name in {"torch", "cuda"}:
        return dict(payload.get("torch_runtime", {}))
    if name == "distributed":
        return dict(payload.get("distributed_runtime", {}))
    if name == "deepspeed":
        return dict(payload.get("deepspeed_runtime", {}))
    return dict(payload.get("ascend_runtime", {}))


def evaluate_diagnostics(
    payload: Dict[str, Any], requirements: Iterable[str]
) -> DiagnosticReport:
    """Evaluate requested capabilities against a doctor inspection payload."""

    ordered = tuple(dict.fromkeys(str(item).lower() for item in requirements))
    unknown = tuple(name for name in ordered if name not in CHECKS)
    if unknown:
        raise ValueError(f"unknown doctor requirements: {', '.join(unknown)}")
    checks = tuple(
        DiagnosticCheck(
            name=name,
            ok=bool(CHECKS[name](payload)),
            message=(
                f"required capability '{name}' is available"
                if CHECKS[name](payload)
                else f"required capability '{name}' is unavailable"
            ),
            evidence=_evidence(name, payload),
        )
        for name in ordered
    )
    return DiagnosticReport(requirements=ordered, checks=checks)


__all__ = [
    "CHECKS",
    "DiagnosticCheck",
    "DiagnosticReport",
    "evaluate_diagnostics",
]
