# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Unified ParaScale command line entrypoint."""

from __future__ import annotations

import argparse
import os

from parascale._version import __version__
from parascale.commands import register_command_parsers
from parascale.commands.benchmark import (
    add_pipeline_cache_arguments as add_pipeline_cache_arguments,
)
from parascale.commands.benchmark import (
    cmd_benchmark_matrix as cmd_benchmark_matrix,
)
from parascale.commands.benchmark import (
    cmd_benchmark_stability as cmd_benchmark_stability,
)
from parascale.commands.checkpoint import (
    build_checkpoint_validation_payload as build_checkpoint_validation_payload,
)
from parascale.commands.checkpoint import (
    cmd_checkpoint_validate as cmd_checkpoint_validate,
)
from parascale.commands.common import emit_error_json
from parascale.commands.common import load_config_file as load_config_file
from parascale.commands.configuration import (
    cmd_config_migrate as cmd_config_migrate,
)
from parascale.commands.configuration import (
    cmd_config_validate as cmd_config_validate,
)
from parascale.commands.doctor import cmd_doctor as cmd_doctor
from parascale.commands.errors import classify_exception
from parascale.commands.plan import (
    build_plan_payload as build_plan_payload,
)
from parascale.commands.plan import cmd_plan as cmd_plan
from parascale.commands.run import (
    build_benchmark_dry_run_payload as build_benchmark_dry_run_payload,
)
from parascale.commands.run import (
    build_serve_dry_run_payload as build_serve_dry_run_payload,
)
from parascale.commands.run import (
    build_train_dry_run_payload as build_train_dry_run_payload,
)
from parascale.commands.run import cmd_benchmark as cmd_benchmark
from parascale.commands.run import cmd_infer as cmd_infer
from parascale.commands.run import cmd_serve as cmd_serve
from parascale.commands.run import cmd_train as cmd_train
from parascale.commands.run import (
    run_benchmark_from_config as run_benchmark_from_config,
)
from parascale.commands.run import (
    run_inference_from_config as run_inference_from_config,
)
from parascale.commands.run import run_serve_from_config as run_serve_from_config
from parascale.commands.run import run_train_from_config as run_train_from_config
from parascale.commands.smoke import build_smoke_report as build_smoke_report
from parascale.commands.smoke import cmd_smoke as cmd_smoke
from parascale.commands.vision import cmd_vision_profile as cmd_vision_profile

__all__ = [
    "add_pipeline_cache_arguments",
    "build_benchmark_dry_run_payload",
    "build_checkpoint_validation_payload",
    "build_plan_payload",
    "build_serve_dry_run_payload",
    "build_smoke_report",
    "build_train_dry_run_payload",
    "cmd_benchmark",
    "cmd_benchmark_matrix",
    "cmd_benchmark_stability",
    "cmd_checkpoint_validate",
    "cmd_doctor",
    "cmd_infer",
    "cmd_plan",
    "cmd_serve",
    "cmd_smoke",
    "cmd_train",
    "cmd_vision_profile",
    "load_config_file",
    "main",
    "run_benchmark_from_config",
    "run_inference_from_config",
    "run_serve_from_config",
    "run_train_from_config",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="parascale",
        description="ParaScale training utilities.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=None,
        help="Local rank injected by distributed launchers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_command_parsers(subparsers)

    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    try:
        return int(args.func(args))
    except Exception as exc:
        if os.environ.get("PARASCALE_DEBUG") == "1":
            raise
        failure = classify_exception(exc)
        emit_error_json(failure.to_dict(command))
        return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
