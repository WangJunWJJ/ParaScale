# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:17
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Long-window stability benchmark workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from parascale.commands.common import load_config_file
from parascale.commands.launcher import (
    benchmark_matrix_env,
    run_matrix_command,
    train_matrix_command,
)
from parascale.commands.scenario import (
    apply_pipeline_cache_args,
    benchmark_matrix_scenario_config,
    build_matrix_config,
    section,
)
from parascale.commands.stability_report import (
    collect_stability_results,
    write_stability_markdown,
)
from parascale.commands.stability_resume import (
    append_stability_resume_command,
    run_stability_resume_phase,
)


def run_benchmark_stability_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    scenario = str(args.scenario)
    scenario_config = benchmark_matrix_scenario_config(scenario, args)
    output_dir = Path(args.output_dir or "runs/benchmarks/stability")
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = Path(args.markdown or output_dir / "stability_report.md")
    summary_path = Path(args.summary or output_dir / "summary.json")
    default_backends = ["native_ddp"]
    if scenario == "vlm-lora-real":
        default_backends = ["deepspeed_zero2", "fsdp"]
    backends = list(args.backends or default_backends)
    workers_values = stability_workers(args)
    dataloader_sweeps = stability_dataloader_sweeps(args)
    env = benchmark_matrix_env()
    commands: list[Dict[str, Any]] = []
    results: list[Dict[str, Any]] = []
    for run_spec in scenario_config["runs"]:
        for workers in workers_values:
            for dataloader_options in dataloader_sweeps:
                suffix = stability_dataloader_suffix(workers, dataloader_options)
                for backend in backends:
                    run_id = f"{run_spec['run_id']}_{suffix}_{backend}"
                    config_path = output_dir / f"{run_id}.config.json"
                    result_path = output_dir / f"{run_id}.json"
                    error_path = output_dir / f"{run_id}.error.json"
                    log_path = output_dir / f"{run_id}.log"
                    config_data = build_matrix_config(
                        scenario=scenario,
                        base_config=load_config_file(scenario_config["base_config"]),
                        run_spec={**run_spec, "run_id": run_id},
                        backend=backend,
                        output_dir=output_dir,
                        max_steps=args.max_steps,
                        warmup_steps=args.warmup_steps,
                        batch_size=args.batch_size,
                        num_samples=args.num_samples,
                    )
                    apply_stability_config(
                        config_data, args, workers, dataloader_options
                    )
                    config_path.write_text(
                        json.dumps(config_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    command = train_matrix_command(
                        backend=backend,
                        config_path=config_path,
                        result_path=result_path,
                        nproc_per_node=args.nproc_per_node,
                        master_port=int(args.master_port) + len(commands),
                    )
                    commands.append(
                        {
                            "run_id": run_id,
                            "phase": "train",
                            "backend": backend,
                            "dataloader_workers": workers,
                            "dataloader_options": dict(dataloader_options),
                            "config": str(config_path),
                            "output": str(result_path),
                            "log": str(log_path),
                            "command": command,
                        }
                    )
                    if args.dry_run:
                        if bool(args.resume_stress):
                            append_stability_resume_command(
                                args=args,
                                commands=commands,
                                output_dir=output_dir,
                                run_id=run_id,
                                backend=backend,
                                workers=workers,
                                dataloader_options=dataloader_options,
                                config_data=config_data,
                                checkpoint_step=int(
                                    args.kill_step or max(1, int(args.max_steps) // 2)
                                ),
                            )
                        continue
                    result = run_matrix_command(
                        command,
                        env=env,
                        backend=backend,
                        run_id=run_id,
                        error_path=error_path,
                        log_path=log_path,
                    )
                    results.append(result)
                    if bool(args.resume_stress):
                        results.extend(
                            run_stability_resume_phase(
                                args=args,
                                env=env,
                                commands=commands,
                                output_dir=output_dir,
                                run_id=run_id,
                                backend=backend,
                                workers=workers,
                                dataloader_options=dataloader_options,
                                config_data=config_data,
                                checkpoint_step=int(
                                    args.kill_step or max(1, int(args.max_steps) // 2)
                                ),
                            )
                        )
    payload: Dict[str, Any] = {
        "mode": "benchmark_stability",
        "dry_run": bool(args.dry_run),
        "scenario": scenario,
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
        "commands": commands,
        "results": results,
        "stability": collect_stability_results(
            output_dir, warmup_steps=int(args.warmup_steps or 0)
        ),
    }
    if not args.dry_run:
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_stability_markdown(payload, markdown_path)
    return payload


def stability_workers(args: argparse.Namespace) -> list[int]:
    values = getattr(args, "dataloader_workers_sweep", None)
    if values:
        return sorted({max(0, int(value)) for value in values})
    return [max(0, int(getattr(args, "dataloader_workers", 0) or 0))]


def bool_sweep(values: Any, default: bool | None = None) -> list[bool | None]:
    if not values:
        return [default]
    normalized = []
    for value in values:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            normalized.append(True)
        elif text in {"0", "false", "no", "n", "off"}:
            normalized.append(False)
        else:
            raise ValueError(f"Invalid boolean sweep value: {value}")
    return sorted(set(normalized), key=lambda item: str(item))


def stability_dataloader_sweeps(args: argparse.Namespace) -> list[Dict[str, Any]]:
    persistent_values = bool_sweep(
        getattr(args, "persistent_workers_sweep", None), default=None
    )
    prefetch_values = getattr(args, "prefetch_factor_sweep", None) or [None]
    pin_values = bool_sweep(getattr(args, "pin_memory_sweep", None), default=None)
    sweeps: list[Dict[str, Any]] = []
    for persistent in persistent_values:
        for prefetch in prefetch_values:
            for pin_memory in pin_values:
                options: Dict[str, Any] = {}
                if persistent is not None:
                    options["persistent_workers"] = bool(persistent)
                if prefetch is not None:
                    options["prefetch_factor"] = int(prefetch)
                if pin_memory is not None:
                    options["pin_memory"] = bool(pin_memory)
                sweeps.append(options)
    return sweeps or [{}]


def stability_dataloader_suffix(
    workers: int, dataloader_options: Mapping[str, Any]
) -> str:
    parts = [f"w{workers}"]
    if "persistent_workers" in dataloader_options:
        parts.append(f"pw{int(bool(dataloader_options['persistent_workers']))}")
    if "prefetch_factor" in dataloader_options:
        parts.append(f"pf{int(dataloader_options['prefetch_factor'])}")
    if "pin_memory" in dataloader_options:
        parts.append(f"pin{int(bool(dataloader_options['pin_memory']))}")
    return "_".join(parts)


def apply_stability_config(
    config_data: Dict[str, Any],
    args: argparse.Namespace,
    workers: int,
    dataloader_options: Mapping[str, Any] | None = None,
) -> None:
    parascale = section(config_data, "parascale")
    training = section(config_data, "training")
    data = section(config_data, "data")
    parascale["dataloader_num_workers"] = int(workers)
    data["num_workers"] = int(workers)
    dataloader_options = dict(dataloader_options or {})
    if "persistent_workers" in dataloader_options:
        value = bool(dataloader_options["persistent_workers"])
        parascale["dataloader_persistent_workers"] = value
        data["persistent_workers"] = value
    if "prefetch_factor" in dataloader_options:
        value = int(dataloader_options["prefetch_factor"])
        parascale["dataloader_prefetch_factor"] = value
        data["prefetch_factor"] = value
    if "pin_memory" in dataloader_options:
        value = bool(dataloader_options["pin_memory"])
        parascale["dataloader_pin_memory"] = value
        data["pin_memory"] = value
    if bool(getattr(args, "pipeline_cache", False)):
        parascale["pipeline_cache"] = True
        data["pipeline_cache"] = True
    apply_pipeline_cache_args(config_data, args)
    if bool(getattr(args, "prompt_template_cache", False)):
        parascale["prompt_template_cache"] = True
        data["prompt_template_cache"] = True
    if bool(getattr(args, "preprocess_in_workers", False)):
        parascale["preprocess_in_workers"] = True
        data["preprocess_in_workers"] = True
    if workers <= 0:
        parascale["dataloader_persistent_workers"] = False
        data["persistent_workers"] = False
    if args.checkpoint_interval is not None:
        parascale["checkpoint_save_interval"] = int(args.checkpoint_interval)
        training["checkpoint_interval"] = int(args.checkpoint_interval)
    training["skip_final_checkpoint"] = False
    if args.resume_stress:
        kill_step = int(args.kill_step or max(1, int(args.max_steps) // 2))
        training["max_steps"] = kill_step
        training["benchmark_steps"] = kill_step
