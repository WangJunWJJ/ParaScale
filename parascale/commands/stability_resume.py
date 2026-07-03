# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:23
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Resume-stress helpers for stability benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from parascale.commands.launcher import run_matrix_command, train_matrix_command
from parascale.commands.scenario import section


def can_run_resume_phase(training_result: Mapping[str, Any]) -> bool:
    """Return whether a completed training phase is eligible for resume."""
    completed = (
        training_result.get("status") == "ok"
        and int(training_result.get("returncode", 0) or 0) == 0
    )
    interrupted_with_checkpoint = (
        training_result.get("status") == "interrupted"
        and bool(training_result.get("intentional_kill"))
        and bool(training_result.get("checkpoint_ok"))
    )
    return completed or interrupted_with_checkpoint


def skipped_resume_result(
    *,
    run_id: str,
    backend: str,
    training_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Describe a resume phase skipped because its training dependency failed."""
    return {
        "run_id": f"{run_id}_resume",
        "phase": "resume",
        "backend": backend,
        "status": "skipped",
        "reason": "upstream_failed",
        "depends_on": run_id,
        "upstream_returncode": training_result.get("returncode"),
        "upstream_error": training_result.get("error"),
    }


def run_stability_resume_phase(
    *,
    args: argparse.Namespace,
    env: Dict[str, str],
    commands: list[Dict[str, Any]],
    output_dir: Path,
    run_id: str,
    backend: str,
    workers: int,
    dataloader_options: Mapping[str, Any] | None = None,
    config_data: Dict[str, Any],
    checkpoint_step: int,
) -> list[Dict[str, Any]]:
    record = append_stability_resume_command(
        args=args,
        commands=commands,
        output_dir=output_dir,
        run_id=run_id,
        backend=backend,
        workers=workers,
        dataloader_options=dataloader_options,
        config_data=config_data,
        checkpoint_step=checkpoint_step,
    )
    result = run_matrix_command(
        record["command"],
        env=env,
        backend=backend,
        run_id=str(record["run_id"]),
        error_path=Path(str(record["output"])).with_suffix(".error.json"),
        log_path=Path(str(record["log"])),
    )
    return [result]


def append_stability_resume_command(
    *,
    args: argparse.Namespace,
    commands: list[Dict[str, Any]],
    output_dir: Path,
    run_id: str,
    backend: str,
    workers: int,
    dataloader_options: Mapping[str, Any] | None = None,
    config_data: Dict[str, Any],
    checkpoint_step: int,
) -> Dict[str, Any]:
    resume_run_id = f"{run_id}_resume"
    resume_config = json.loads(json.dumps(config_data))
    training = section(resume_config, "training")
    resume_steps = int(
        args.resume_steps or max(1, int(args.max_steps) - checkpoint_step)
    )
    training["max_steps"] = resume_steps
    training["benchmark_steps"] = resume_steps
    resume_config_path = output_dir / f"{resume_run_id}.config.json"
    resume_result_path = output_dir / f"{resume_run_id}.json"
    resume_log_path = output_dir / f"{resume_run_id}.log"
    resume_config_path.write_text(
        json.dumps(resume_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    command = train_matrix_command(
        backend=backend,
        config_path=resume_config_path,
        result_path=resume_result_path,
        nproc_per_node=args.nproc_per_node,
        master_port=int(args.master_port) + len(commands),
        resume_step=checkpoint_step,
    )
    commands.append(
        record := {
            "run_id": resume_run_id,
            "phase": "resume",
            "resume_step": checkpoint_step,
            "backend": backend,
            "dataloader_workers": workers,
            "dataloader_options": dict(dataloader_options or {}),
            "config": str(resume_config_path),
            "output": str(resume_result_path),
            "log": str(resume_log_path),
            "command": command,
        }
    )
    return record
