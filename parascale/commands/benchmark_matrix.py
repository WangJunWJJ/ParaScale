# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark matrix command workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from parascale.commands.common import load_config_file
from parascale.commands.launcher import (
    benchmark_matrix_command,
    benchmark_matrix_env,
    matrix_result_is_oom,
    oom_retry_policy_payload,
    run_matrix_command,
    run_oom_retry_sequence,
)
from parascale.commands.scenario import (
    apply_pipeline_cache_args,
    benchmark_matrix_scenario_config,
    build_matrix_config,
    matrix_batch_sizes,
)
from parascale.configuration import write_config_artifacts
from parascale.reporting import (
    build_backend_matrix_report,
    write_backend_matrix_markdown,
)
from parascale.runtime.profiles import BenchmarkProfileStore


def run_benchmark_matrix_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    scenario = str(args.scenario)
    scenario_config = benchmark_matrix_scenario_config(scenario, args)
    output_dir = Path(args.output_dir or scenario_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = Path(args.markdown or scenario_config["markdown"])
    summary_path = Path(args.summary or output_dir / "summary.json")
    default_backends = ["native_ddp", "fsdp", "deepspeed"]
    if scenario == "vlm-lora-real":
        default_backends = [
            "native_ddp",
            "fsdp",
            "deepspeed_zero2",
            "deepspeed_zero3",
        ]
    backends = list(args.backends or default_backends)
    env = benchmark_matrix_env()
    commands = []
    run_results = []
    retry_results = []
    batch_sizes = matrix_batch_sizes(args)
    for run_spec in scenario_config["runs"]:
        for batch_size in batch_sizes:
            matrix_run_spec = dict(run_spec)
            if batch_size is not None and len(batch_sizes) > 1:
                matrix_run_spec["run_id"] = f"{run_spec['run_id']}_b{batch_size}"
            for backend in backends:
                config_path = (
                    output_dir / f"{matrix_run_spec['run_id']}_{backend}.config.json"
                )
                result_path = output_dir / f"{matrix_run_spec['run_id']}_{backend}.json"
                error_path = (
                    output_dir / f"{matrix_run_spec['run_id']}_{backend}.error.json"
                )
                log_path = output_dir / f"{matrix_run_spec['run_id']}_{backend}.log"
                config_data = build_matrix_config(
                    scenario=scenario,
                    base_config=load_config_file(scenario_config["base_config"]),
                    run_spec=matrix_run_spec,
                    backend=backend,
                    output_dir=output_dir,
                    max_steps=args.max_steps,
                    warmup_steps=args.warmup_steps,
                    batch_size=batch_size,
                    num_samples=args.num_samples,
                )
                apply_pipeline_cache_args(config_data, args)
                artifact_run_id = f"{matrix_run_spec['run_id']}_{backend}"
                artifact_dir = output_dir / artifact_run_id
                config_data.setdefault("runtime", {})["run_dir"] = str(artifact_dir)
                artifacts = write_config_artifacts(
                    config_data,
                    artifact_dir,
                    dry_run=bool(args.dry_run),
                )
                config_path.write_text(
                    json.dumps(config_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                command = benchmark_matrix_command(
                    backend=backend,
                    config_path=config_path,
                    result_path=result_path,
                    nproc_per_node=args.nproc_per_node,
                    master_port=int(args.master_port) + len(commands),
                )
                command_record = {
                    "run_id": matrix_run_spec["run_id"],
                    "backend": backend,
                    "batch_size": batch_size,
                    "config": str(config_path),
                    "output": str(result_path),
                    "log": str(log_path),
                    "command": command,
                    "config_artifacts": artifacts,
                }
                commands.append(command_record)
                if args.dry_run:
                    continue
                result = run_matrix_command(
                    command,
                    env=env,
                    backend=backend,
                    run_id=matrix_run_spec["run_id"],
                    error_path=error_path,
                    log_path=log_path,
                )
                run_results.append(result)
                if bool(args.oom_retry) and matrix_result_is_oom(result):
                    retry_results.extend(
                        run_oom_retry_sequence(
                            scenario=scenario,
                            scenario_config=scenario_config,
                            base_run_spec=matrix_run_spec,
                            failed_backend=backend,
                            failed_batch_size=batch_size,
                            output_dir=output_dir,
                            env=env,
                            args=args,
                            commands=commands,
                        )
                    )
    if args.dry_run:
        return {
            "mode": "benchmark_matrix",
            "dry_run": True,
            "scenario": scenario,
            "commands": commands,
            "batch_size_sweep": [size for size in batch_sizes if size is not None],
            "oom_retry": bool(args.oom_retry),
            "oom_retry_policy": oom_retry_policy_payload(),
            "output_dir": str(output_dir),
            "summary": str(summary_path),
            "markdown": str(markdown_path),
        }
    report = build_backend_matrix_report(
        output_dir,
        title=scenario_config["title"],
        workload_label=scenario_config["workload_label"],
        optimize_for=args.optimize_for,
        throughput_tolerance=float(args.throughput_tolerance),
    )
    attach_matrix_tuner_explanations(report, output_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    write_backend_matrix_markdown(report, markdown_path)
    return {
        "mode": "benchmark_matrix",
        "dry_run": False,
        "scenario": scenario,
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
        "commands": commands,
        "run_results": run_results,
        "retry_results": retry_results,
        "report": report,
    }


def attach_matrix_tuner_explanations(
    report: Dict[str, Any],
    output_dir: Path,
) -> None:
    explanations = []
    for row in report.get("results", []):
        if row.get("status") != "ok":
            continue
        result_path = Path(str(row.get("path", "")))
        if not result_path.exists():
            continue
        config_path = output_dir / f"{result_path.stem}.config.json"
        if not config_path.exists():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = dict(payload.get("metrics", {}))
            runtime_profile = runtime_profile_from_metrics(metrics)
            if not runtime_profile:
                continue
            config_data["runtime_profile"] = runtime_profile
            explain_payload = build_plan_explain_payload(config_data)
            explanations.append(
                {
                    "run_id": row.get("run_id"),
                    "backend": row.get("backend"),
                    "runtime_profile": runtime_profile,
                    "runtime_tuning": explain_payload.get("runtime_tuning", {}),
                    "explain": explain_payload.get("explain", {}),
                }
            )
        except Exception as exc:
            explanations.append(
                {
                    "run_id": row.get("run_id"),
                    "backend": row.get("backend"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    report["tuner_explanations"] = explanations


def runtime_profile_from_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return BenchmarkProfileStore().runtime_profile_from_metrics(metrics)


def build_plan_explain_payload(config_data: Dict[str, Any]) -> Dict[str, Any]:
    from parascale.commands.plan import build_plan_payload

    return build_plan_payload(config_data)
