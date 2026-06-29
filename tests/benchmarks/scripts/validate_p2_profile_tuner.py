# -*- coding: utf-8 -*-
# @Time : 2026/6/12 下午4:07
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Validate P2 profile-driven tuner explanations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from parascale.cli import build_plan_payload

GB = 1024**3


def _base_config() -> Dict[str, Any]:
    return {
        "parascale": {
            "task_type": "multimodal",
            "model_family": "clip",
            "training_backend": "auto",
            "optimize_for": "throughput",
            "batching_strategy": "sample",
            "dataloader_num_workers": 2,
            "dataloader_prefetch_factor": 2,
        },
        "model_profile": {
            "total_params": 150_000_000,
            "total_memory": 1_200_000_000,
            "num_layers": 18,
            "model_type": "clip_medium",
        },
        "hardware_profile": {
            "num_gpus": 2,
            "gpus_per_node": 2,
            "gpu_memory": 24 * GB,
            "available_memory": 20 * GB,
        },
    }


def _scenario(name: str, runtime_profile: Dict[str, Any]) -> Dict[str, Any]:
    config = _base_config()
    config["runtime_profile"] = runtime_profile
    payload = build_plan_payload(config)
    tuning = payload.get("runtime_tuning", {})
    decisions = tuning.get("decisions", [])
    explain = payload.get("explain", {})
    return {
        "name": name,
        "ok": bool(decisions)
        and bool(explain.get("summary"))
        and bool(tuning.get("observed_profile"))
        and bool(tuning.get("thresholds")),
        "selected_backend": payload.get("strategy_plan", {}).get("backend"),
        "actions": tuning.get("actions", []),
        "decisions": decisions,
        "config_updates": tuning.get("config_updates", {}),
        "explain": explain,
    }


def build_report() -> Dict[str, Any]:
    scenarios = [
        _scenario(
            "memory_pressure",
            {
                "peak_memory_per_gpu": 22 * GB,
                "padding_ratio": 0.1,
                "batch_tokens": 8192,
            },
        ),
        _scenario(
            "padding_and_dataloader_wait",
            {
                "peak_memory_per_gpu": 8 * GB,
                "padding_ratio": 0.5,
                "batch_tokens": 8192,
                "dataloader_wait_ms": 60.0,
                "images_per_second": 80.0,
            },
        ),
    ]
    return {
        "mode": "p2_profile_tuner_validation",
        "passed": all(item["ok"] for item in scenarios),
        "scenarios": scenarios,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = [
        "# P2 Profile/Tuner Validation Report",
        "",
        f"- Passed: {report['passed']}",
        "",
        "| Scenario | OK | Backend | Actions | Config updates |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["scenarios"]:
        lines.append(
            "| {name} | {ok} | {backend} | {actions} | {updates} |".format(
                name=item["name"],
                ok=item["ok"],
                backend=item["selected_backend"],
                actions=", ".join(item["actions"]),
                updates=", ".join(sorted(item["config_updates"].keys())),
            )
        )
    lines.extend(["", "## Decision Details", ""])
    for item in report["scenarios"]:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append(item["explain"].get("summary", ""))
        lines.append("")
        for decision in item["decisions"]:
            lines.append(
                "- `{action}`: {reason} evidence={evidence} threshold={threshold}".format(
                    action=decision.get("action"),
                    reason=decision.get("reason"),
                    evidence=decision.get("evidence"),
                    threshold=decision.get("threshold"),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(report, Path(args.markdown))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
