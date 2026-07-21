# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:01
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Plan command payload builders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from parascale.commands.common import load_config_file
from parascale.config import ParaScaleConfig
from parascale.configuration import resolve_config
from parascale.data import build_dataloader_plan
from parascale.parallel import build_parallel_plan
from parascale.runtime import (
    build_benchmark_plan,
    build_launch_plan,
    build_runtime_context,
)
from parascale.runtime.profiles import BenchmarkProfileStore
from parascale.strategy import (
    RuntimeProfile,
    build_strategy_plan,
    tune_strategy_from_runtime,
)

PLAN_EXAMPLES = """examples:
  python -m parascale.cli plan --config configs/quickstart/tiny_torch.yaml
  python -m parascale.cli plan --config configs/quickstart/vision_synthetic.json --json
  python -m parascale.cli plan --config configs/quickstart/vision_synthetic.json --output runs/plan.json
"""


def register_plan_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "plan",
        help="Build an auto strategy and dataloader plan.",
        epilog=PLAN_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, help="Path to a JSON/YAML planning config."
    )
    parser.add_argument(
        "--output", help="Optional path to write the generated plan JSON."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable plan JSON instead of the summary.",
    )
    parser.set_defaults(func=cmd_plan)


def section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def build_plan_payload(config_data: Dict[str, Any]) -> Dict[str, Any]:
    resolved_config = resolve_config(config_data, dry_run=True)
    context = build_runtime_context(config_data, mode="plan")
    parascale_config = context.config
    hardware_profile = section(config_data, "hardware_profile")
    runtime_profile = BenchmarkProfileStore().runtime_profile_from_config(config_data)

    strategy_plan = context.strategy_plan
    tuning = None
    payload = {
        "mode": "plan",
        "runtime_status": "plan_only",
        "runtime_context": context.to_dict(),
        "launch_plan": build_launch_plan(
            context, **_launch_kwargs(section(config_data, "launch"))
        ).to_dict(),
        "benchmark_plan": build_benchmark_plan(context).to_dict(),
        "parallel_plan": build_parallel_plan(parascale_config, strategy_plan).to_dict(),
        "strategy_plan": strategy_plan.to_dict(),
        "communication_plan": dict(
            getattr(strategy_plan, "communication_plan", {}) or {}
        ),
        "dataloader_plan": build_dataloader_plan(
            parascale_config,
            world_size=context.world_size,
        ).to_dict(),
        "resolved_config": resolved_config.to_dict(),
    }
    if runtime_profile:
        tuning = tune_strategy_from_runtime(
            strategy_plan,
            RuntimeProfile(**runtime_profile),
            hardware_profile,
            parascale_config,
        )
        payload["runtime_tuning"] = tuning.to_dict()
        payload["recommended_strategy_plan"] = tuning.plan.to_dict()
        payload["recommended_config_updates"] = dict(tuning.config_updates)
        payload["runtime_profile_source"] = runtime_profile_source(config_data)
    payload["explain"] = build_plan_explain(strategy_plan, tuning)
    return payload


def _launch_kwargs(launch: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if "nnodes" in launch:
        kwargs["nnodes"] = int(launch["nnodes"])
    if "node_rank" in launch:
        kwargs["node_rank"] = int(launch["node_rank"])
    if "master_addr" in launch:
        kwargs["master_addr"] = str(launch["master_addr"])
    if "master_port" in launch:
        kwargs["master_port"] = int(launch["master_port"])
    return kwargs


def runtime_profile_source(config_data: Dict[str, Any]) -> str:
    if section(config_data, "runtime_profile"):
        return "runtime_profile"
    benchmark_profile = config_data.get("benchmark_profile")
    if isinstance(benchmark_profile, dict):
        if benchmark_profile.get("path") or benchmark_profile.get("result_path"):
            return "benchmark_profile.path"
        return "benchmark_profile.metrics"
    if config_data.get("benchmark_result_path"):
        return "benchmark_result_path"
    if config_data.get("benchmark_profile_path"):
        return "benchmark_profile_path"
    return "unknown"


def build_plan_explain(strategy_plan: Any, tuning: Any = None) -> Dict[str, Any]:
    static_reasons = list(getattr(strategy_plan, "reasons", []) or [])
    static_warnings = list(getattr(strategy_plan, "warnings", []) or [])
    explanation: Dict[str, Any] = {
        "selected_backend": getattr(strategy_plan, "backend", "unknown"),
        "strategy_type": getattr(strategy_plan, "strategy_type", "unknown"),
        "communication_plan": dict(
            getattr(strategy_plan, "communication_plan", {}) or {}
        ),
        "static_reasons": static_reasons,
        "static_warnings": static_warnings,
        "runtime_decisions": [],
        "recommended_config_updates": {},
        "summary": (
            static_reasons[0] if static_reasons else "No strategy reason recorded."
        ),
    }
    if tuning is not None:
        tuning_dict = tuning.to_dict()
        explanation["runtime_decisions"] = tuning_dict.get("decisions", [])
        explanation["recommended_config_updates"] = tuning_dict.get(
            "config_updates", {}
        )
        if explanation["runtime_decisions"]:
            first = explanation["runtime_decisions"][0]
            explanation["summary"] = (
                f"{explanation['summary']} Runtime tuner recommends "
                f"{first.get('action')} because {first.get('reason')}"
            )
    return explanation


def format_plan_summary(payload: Dict[str, Any]) -> str:
    explain = section(payload, "explain")
    strategy = section(payload, "strategy_plan")
    dataloader = section(payload, "dataloader_plan")
    launch = section(payload, "launch_plan")
    benchmark = section(payload, "benchmark_plan")

    selected_backend = explain.get("selected_backend") or strategy.get(
        "backend", "unknown"
    )
    precision = strategy.get("precision", "unknown")
    batch_policy = strategy.get(
        "batch_policy", dataloader.get("batch_sampler", "unknown")
    )
    launcher = launch.get("launcher", "local")
    world_size = launch.get(
        "world_size", payload.get("runtime_context", {}).get("world_size", 1)
    )
    scenarios = benchmark.get("scenarios", [])
    scenario_names = [
        str(item.get("name", item)) if isinstance(item, Mapping) else str(item)
        for item in scenarios
    ]

    lines = [
        "ParaScale plan",
        f"- backend: {selected_backend}",
        f"- precision: {precision}",
        f"- batch policy: {batch_policy}",
        f"- launcher: {launcher} (world_size={world_size})",
    ]
    if scenario_names:
        lines.append(f"- benchmark scenarios: {', '.join(scenario_names[:3])}")
    summary = explain.get("summary")
    if summary:
        lines.append(f"- why: {summary}")

    reasons = explain.get("static_reasons") or []
    for reason in reasons[:3]:
        if reason != summary:
            lines.append(f"  reason: {reason}")

    decisions = explain.get("runtime_decisions") or []
    if decisions:
        lines.append("- runtime tuner:")
        for decision in decisions[:3]:
            action = decision.get("action", "update")
            reason = decision.get("reason", "no reason recorded")
            lines.append(f"  - {action}: {reason}")

    updates = explain.get("recommended_config_updates") or payload.get(
        "recommended_config_updates", {}
    )
    if updates:
        lines.append("- recommended config updates:")
        for key, value in list(updates.items())[:8]:
            lines.append(f"  - {key}: {value}")

    warnings = explain.get("static_warnings") or strategy.get("warnings") or []
    if warnings:
        lines.append("- warnings:")
        for warning in warnings[:5]:
            lines.append(f"  - {warning}")

    lines.append("")
    lines.append("Use --json for the full machine-readable plan.")
    return "\n".join(lines)


def cmd_plan(args: argparse.Namespace) -> int:
    payload = build_plan_payload(load_config_file(args.config))
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    if args.json or not args.output:
        if args.json:
            print(output)
        elif not args.output:
            print(format_plan_summary(payload))
    return 0


def build_static_strategy_plan(config_data: Dict[str, Any]) -> Dict[str, Any]:
    model_profile = section(config_data, "model_profile")
    hardware_profile = section(config_data, "hardware_profile")
    parascale_config = ParaScaleConfig.from_dict(config_data.get("parascale", {}))
    return build_strategy_plan(
        model_profile,
        hardware_profile,
        parascale_config,
    ).to_dict()
