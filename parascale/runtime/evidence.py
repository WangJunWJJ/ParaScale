# -*- coding: utf-8 -*-
# @Time : 2026/7/24 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Low-frequency runtime evidence helpers for command payloads."""

from __future__ import annotations

from typing import Any, Dict


def attach_runtime_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a compact, uniform evidence summary to a command payload."""

    payload["evidence"] = build_runtime_evidence(payload)
    return payload


def build_runtime_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a stable evidence envelope from an already assembled payload."""

    evidence: Dict[str, Any] = {
        "mode": payload.get("mode"),
        "dry_run": bool(payload.get("dry_run", False)),
        "runtime_status": payload.get("runtime_status", "unknown"),
        "capability_level": payload.get("capability_level", "unknown"),
        "mock": bool(payload.get("mock", False)),
        "synthetic": bool(payload.get("synthetic", False)),
    }
    if "strict_errors" in payload:
        evidence["strict_errors"] = bool(payload.get("strict_errors"))
    if "measurement_window" in payload:
        evidence["measurement_window"] = dict(payload.get("measurement_window") or {})
    elif isinstance(payload.get("metrics"), dict):
        window = _measurement_window_from_metrics(payload["metrics"])
        if window:
            evidence["measurement_window"] = window
    if "resolved_config" in payload:
        resolved = payload.get("resolved_config") or {}
        warnings = resolved.get("warnings", []) if isinstance(resolved, dict) else []
        errors = resolved.get("errors", []) if isinstance(resolved, dict) else []
        evidence["resolved_config"] = {
            "available": isinstance(resolved, dict),
            "warning_count": len(warnings) if isinstance(warnings, list) else 0,
            "error_count": len(errors) if isinstance(errors, list) else 0,
        }
    if "config_artifacts" in payload:
        artifacts = payload.get("config_artifacts") or {}
        if isinstance(artifacts, dict):
            evidence["config_artifacts"] = {
                "run_dir": artifacts.get("run_dir"),
                "resolved_config": artifacts.get("resolved_config"),
                "deepspeed_final_config": artifacts.get("deepspeed_final_config"),
            }
    return evidence


def _measurement_window_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any] | None:
    if "warmup_steps" not in metrics and "measured_steps" not in metrics:
        return None
    return {
        "warmup_steps_effective": int(metrics.get("warmup_steps", 0) or 0),
        "measured_steps": int(metrics.get("measured_steps", 0) or 0),
        "warmup_excluded_from_metrics": int(metrics.get("warmup_steps", 0) or 0) > 0,
    }


__all__ = ["attach_runtime_evidence", "build_runtime_evidence"]
