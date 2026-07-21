# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:02
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Smoke command implementation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict

from parascale.commands.checkpoint import build_checkpoint_validation_payload
from parascale.commands.common import load_config_file
from parascale.commands.doctor import build_doctor_payload
from parascale.commands.plan import build_plan_payload
from parascale.commands.run import run_serve_from_config, run_train_from_config

SMOKE_EXAMPLES = """examples:
  python -m parascale.cli smoke --config configs/quickstart/tiny_torch.yaml --skip-real
  python -m parascale.cli smoke --config configs/quickstart/tiny_torch.yaml
"""


def register_smoke_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "smoke",
        help="Run the compact server smoke flow and write a JSON report.",
        epilog=SMOKE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/server_tiny_torch.json",
        help="Path to a JSON/YAML smoke config.",
    )
    parser.add_argument(
        "--output",
        default="runs/server_smoke_report.json",
        help="Path to write the smoke JSON report.",
    )
    parser.add_argument(
        "--skip-real", action="store_true", help="Only run doctor and plan."
    )
    parser.set_defaults(func=cmd_smoke)


def build_smoke_report(config_path: str, skip_real: bool = False) -> Dict[str, Any]:
    config = load_config_file(config_path)
    report: Dict[str, Any] = {
        "config": config_path,
        "started_at_unix": time.time(),
        "steps": {},
    }
    report["steps"]["doctor"] = capture_step(build_doctor_payload)
    report["steps"]["plan"] = capture_step(lambda: build_plan_payload(config))

    if not skip_real:
        train_step = capture_step(lambda: run_train_from_config(config))
        report["steps"]["train"] = train_step
        if train_step["ok"]:
            trained = train_step["result"]
            resume_step = int(trained.get("global_step", 0))
            checkpoint = str(trained.get("checkpoint"))
            report["steps"]["checkpoint_validate"] = capture_step(
                lambda: build_checkpoint_validation_payload(checkpoint)
            )
            report["steps"]["resume"] = capture_step(
                lambda: run_train_from_config(config, resume_step=resume_step)
            )
            report["steps"]["serve"] = capture_step(
                lambda: run_serve_from_config(config, checkpoint=checkpoint)
            )

    report["finished_at_unix"] = time.time()
    return report


def capture_step(fn: Callable[[], Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        result = fn()
        return {
            "ok": True,
            "elapsed_seconds": time.perf_counter() - start,
            "result": result,
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_seconds": time.perf_counter() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def cmd_smoke(args: argparse.Namespace) -> int:
    report = build_smoke_report(args.config, skip_real=args.skip_real)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(str(output_path))
    return 0
