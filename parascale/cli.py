# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Unified ParaScale command line entrypoint."""

from __future__ import annotations

import argparse
import os

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
from parascale.commands.doctor import cmd_doctor as cmd_doctor
from parascale.commands.errors import classify_exception
from parascale.commands.plan import (
    build_plan_payload as build_plan_payload,
)
from parascale.commands.plan import (
    cmd_plan as cmd_plan,
)
from parascale.commands.run import (
    build_benchmark_dry_run_payload as build_benchmark_dry_run_payload,
)
from parascale.commands.run import (
    build_serve_dry_run_payload as build_serve_dry_run_payload,
)
from parascale.commands.run import (
    build_train_dry_run_payload as build_train_dry_run_payload,
)
from parascale.commands.run import (
    cmd_benchmark as cmd_benchmark,
)
from parascale.commands.run import (
    cmd_infer as cmd_infer,
)
from parascale.commands.run import (
    cmd_serve as cmd_serve,
)
from parascale.commands.run import (
    cmd_train as cmd_train,
)
from parascale.commands.run import (
    run_benchmark_from_config as run_benchmark_from_config,
)
from parascale.commands.run import (
    run_inference_from_config as run_inference_from_config,
)
from parascale.commands.run import (
    run_serve_from_config as run_serve_from_config,
)
from parascale.commands.run import (
    run_train_from_config as run_train_from_config,
)
from parascale.commands.smoke import (
    build_smoke_report as build_smoke_report,
)
from parascale.commands.smoke import (
    cmd_smoke as cmd_smoke,
)
from parascale.commands.vision import (
    cmd_vision_profile as cmd_vision_profile,
)

ROOT_EXAMPLES = """examples:
  python -m parascale.cli doctor
  python -m parascale.cli plan --config configs/quickstart/tiny_torch.yaml
  python -m parascale.cli train --config configs/quickstart/tiny_torch.yaml --dry-run
  python -m parascale.cli smoke --config configs/quickstart/tiny_torch.yaml --skip-real
"""

PLAN_EXAMPLES = """examples:
  python -m parascale.cli plan --config configs/quickstart/tiny_torch.yaml
  python -m parascale.cli plan --config configs/quickstart/vision_synthetic.json --json
  python -m parascale.cli plan --config configs/quickstart/vision_synthetic.json --output runs/plan.json
"""

TRAIN_EXAMPLES = """examples:
  python -m parascale.cli train --config configs/quickstart/tiny_torch.yaml --dry-run
  python -m parascale.cli train --config configs/quickstart/tiny_torch.yaml
"""

SMOKE_EXAMPLES = """examples:
  python -m parascale.cli smoke --config configs/quickstart/tiny_torch.yaml --skip-real
  python -m parascale.cli smoke --config configs/quickstart/tiny_torch.yaml
"""

BENCHMARK_MATRIX_EXAMPLES = """examples:
  python -m parascale.cli benchmark-matrix --scenario yolo-world-large --variants m --dry-run
  python -m parascale.cli benchmark-matrix --scenario vlm-lora-hf-clip --backends native_ddp fsdp deepspeed --dry-run
"""

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
        epilog=ROOT_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    plan_parser = subparsers.add_parser(
        "plan",
        help="Build an auto strategy and dataloader plan.",
        epilog=PLAN_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plan_parser.add_argument(
        "--config", required=True, help="Path to a JSON/YAML planning config."
    )
    plan_parser.add_argument(
        "--output", help="Optional path to write the generated plan JSON."
    )
    plan_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable plan JSON instead of the summary.",
    )
    plan_parser.set_defaults(func=cmd_plan)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Diagnose local ParaScale runtime dependencies and devices."
    )
    doctor_parser.add_argument(
        "--output", help="Optional path to write the doctor JSON."
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when required runtime capabilities are unavailable.",
    )
    doctor_parser.add_argument(
        "--require",
        action="append",
        choices=["core", "torch", "distributed", "cuda", "deepspeed", "npu"],
        default=[],
        help="Capability required by this environment. Repeat for multiple checks.",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run the compact server smoke flow and write a JSON report.",
        epilog=SMOKE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    smoke_parser.add_argument(
        "--config",
        default="configs/server_tiny_torch.json",
        help="Path to a JSON/YAML smoke config.",
    )
    smoke_parser.add_argument(
        "--output",
        default="runs/server_smoke_report.json",
        help="Path to write the smoke JSON report.",
    )
    smoke_parser.add_argument(
        "--skip-real", action="store_true", help="Only run doctor and plan."
    )
    smoke_parser.set_defaults(func=cmd_smoke)

    train_parser = subparsers.add_parser(
        "train",
        help="Validate and launch a ParaScale training run.",
        epilog=TRAIN_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--config", required=True, help="Path to a JSON/YAML training config."
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the execution plan.",
    )
    train_parser.add_argument(
        "--resume-step", type=int, help="Resume from a ParaScale manifest step."
    )
    train_parser.add_argument(
        "--output", help="Optional path to write the dry-run JSON."
    )
    train_parser.set_defaults(func=cmd_train)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run a ParaScale inference workload through the unified runtime.",
    )
    infer_parser.add_argument(
        "--config", required=True, help="Path to a JSON/YAML inference config."
    )
    infer_parser.add_argument(
        "--output", help="Optional path to write the inference JSON."
    )
    infer_parser.set_defaults(func=cmd_infer)

    serve_parser = subparsers.add_parser(
        "serve", help="Validate and launch a ParaScale serving runtime."
    )
    serve_parser.add_argument(
        "--config", help="Optional path to a JSON/YAML serving config."
    )
    serve_parser.add_argument(
        "--checkpoint", help="Optional checkpoint or manifest path."
    )
    serve_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate serving inputs without starting a server.",
    )
    serve_parser.add_argument(
        "--output", help="Optional path to write the dry-run JSON."
    )
    serve_parser.set_defaults(func=cmd_serve)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Validate and launch a ParaScale benchmark run."
    )
    benchmark_parser.add_argument(
        "--config", required=True, help="Path to a JSON/YAML benchmark config."
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print benchmark plan.",
    )
    benchmark_parser.add_argument(
        "--output", help="Optional path to write the dry-run JSON."
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    matrix_parser = subparsers.add_parser(
        "benchmark-matrix",
        help="Run a unified native-DDP/FSDP/DeepSpeed benchmark matrix.",
        epilog=BENCHMARK_MATRIX_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    matrix_parser.add_argument(
        "--scenario",
        required=True,
        choices=["vlm-lora-hf-clip", "vlm-lora-real", "yolo-world-large"],
        help="Validated benchmark scenario to execute.",
    )
    matrix_parser.add_argument(
        "--backends",
        nargs="+",
        choices=[
            "native_ddp",
            "fsdp",
            "deepspeed",
            "deepspeed_zero2",
            "deepspeed_zero3",
        ],
        help="Backends to run. Defaults to all validated matrix backends.",
    )
    matrix_parser.add_argument(
        "--variants",
        nargs="+",
        choices=["s", "m", "l", "x"],
        help="YOLO-World variants for yolo-world-large.",
    )
    matrix_parser.add_argument("--base-config", help="Override scenario base config.")
    matrix_parser.add_argument("--run-id", help="Override single-run id.")
    matrix_parser.add_argument("--output-dir", help="Directory for matrix JSON files.")
    matrix_parser.add_argument("--summary", help="Path to write matrix summary JSON.")
    matrix_parser.add_argument("--markdown", help="Path to write Markdown report.")
    matrix_parser.add_argument(
        "--output", help="Optional path to write the command payload JSON."
    )
    matrix_parser.add_argument("--max-steps", type=int, help="Override max steps.")
    matrix_parser.add_argument(
        "--warmup-steps", type=int, help="Override benchmark warmup steps."
    )
    matrix_parser.add_argument("--batch-size", type=int, help="Override batch size.")
    matrix_parser.add_argument(
        "--batch-size-sweep",
        nargs="+",
        type=int,
        help="Run one matrix per listed batch size, for example: 1 2 4.",
    )
    matrix_parser.add_argument("--num-samples", type=int, help="Override sample limit.")
    matrix_parser.add_argument("--nproc-per-node", type=int, default=2)
    matrix_parser.add_argument("--master-port", type=int, default=29710)
    matrix_parser.add_argument(
        "--optimize-for",
        choices=["throughput", "memory", "balanced"],
        default="balanced",
        help="Recommendation policy for the generated report.",
    )
    matrix_parser.add_argument("--throughput-tolerance", type=float, default=0.05)
    matrix_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and commands without launching benchmarks.",
    )
    matrix_parser.add_argument(
        "--oom-retry",
        action="store_true",
        help=(
            "On OOM-like failures, retry with a smaller batch, activation "
            "checkpointing, and safer FSDP/DeepSpeed fallbacks."
        ),
    )
    add_pipeline_cache_arguments(matrix_parser)
    matrix_parser.set_defaults(func=cmd_benchmark_matrix)

    stability_parser = subparsers.add_parser(
        "benchmark-stability",
        help="Run long-window stability and resume stress benchmarks.",
    )
    stability_parser.add_argument(
        "--scenario",
        required=True,
        choices=["vlm-lora-real", "yolo-world-large"],
        help="Validated stability scenario to execute.",
    )
    stability_parser.add_argument(
        "--backends",
        nargs="+",
        choices=[
            "native_ddp",
            "fsdp",
            "deepspeed",
            "deepspeed_zero2",
            "deepspeed_zero3",
        ],
        help="Backends to run.",
    )
    stability_parser.add_argument(
        "--variants",
        nargs="+",
        choices=["s", "m", "l", "x"],
        help="YOLO-World variants for yolo-world-large.",
    )
    stability_parser.add_argument(
        "--base-config", help="Override scenario base config."
    )
    stability_parser.add_argument("--run-id", help="Override single-run id.")
    stability_parser.add_argument("--output-dir", help="Directory for stability files.")
    stability_parser.add_argument("--summary", help="Path to write stability summary.")
    stability_parser.add_argument("--markdown", help="Path to write Markdown report.")
    stability_parser.add_argument(
        "--output", help="Optional path to write the command payload JSON."
    )
    stability_parser.add_argument("--max-steps", type=int, default=500)
    stability_parser.add_argument("--warmup-steps", type=int, default=20)
    stability_parser.add_argument("--batch-size", type=int)
    stability_parser.add_argument("--num-samples", type=int)
    stability_parser.add_argument("--nproc-per-node", type=int, default=2)
    stability_parser.add_argument("--master-port", type=int, default=29810)
    stability_parser.add_argument("--dataloader-workers", type=int, default=0)
    stability_parser.add_argument(
        "--dataloader-workers-sweep",
        nargs="+",
        type=int,
        help="Run stability windows for each dataloader worker count.",
    )
    stability_parser.add_argument(
        "--persistent-workers-sweep",
        nargs="+",
        help="Run stability windows for persistent_workers true/false values.",
    )
    stability_parser.add_argument(
        "--prefetch-factor-sweep",
        nargs="+",
        type=int,
        help="Run stability windows for dataloader prefetch factors.",
    )
    stability_parser.add_argument(
        "--pin-memory-sweep",
        nargs="+",
        help="Run stability windows for pin_memory true/false values.",
    )
    add_pipeline_cache_arguments(stability_parser)
    stability_parser.add_argument("--checkpoint-interval", type=int, default=100)
    stability_parser.add_argument(
        "--resume-stress",
        action="store_true",
        help="Run an initial train phase, then restart a fresh launcher from checkpoint.",
    )
    stability_parser.add_argument(
        "--kill-step",
        type=int,
        help="Checkpoint step used as the simulated kill/restart boundary.",
    )
    stability_parser.add_argument(
        "--resume-steps",
        type=int,
        help="Number of steps to run after restart. Defaults to max_steps-kill_step.",
    )
    stability_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and commands without launching training.",
    )
    stability_parser.set_defaults(func=cmd_benchmark_stability)

    vision_profile_parser = subparsers.add_parser(
        "vision-profile", help="Profile a real image folder data pipeline."
    )
    vision_profile_parser.add_argument(
        "--data-dir", required=True, help="Directory containing real image files."
    )
    vision_profile_parser.add_argument(
        "--batch-size", type=int, default=8, help="Images per profiling batch."
    )
    vision_profile_parser.add_argument(
        "--max-batches", type=int, default=4, help="Maximum batches to profile."
    )
    vision_profile_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize image edge used for tensor batches.",
    )
    vision_profile_parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size used for patch-token accounting.",
    )
    vision_profile_parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, etc."
    )
    vision_profile_parser.add_argument(
        "--output", help="Optional path to write the profile JSON."
    )
    vision_profile_parser.set_defaults(func=cmd_vision_profile)

    checkpoint_parser = subparsers.add_parser(
        "checkpoint", help="Checkpoint utility commands."
    )
    checkpoint_subparsers = checkpoint_parser.add_subparsers(
        dest="checkpoint_command", required=True
    )
    validate_parser = checkpoint_subparsers.add_parser(
        "validate", help="Validate a checkpoint manifest and payload files."
    )
    validate_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint root, step directory, or manifest path.",
    )
    validate_parser.add_argument(
        "--output", help="Optional path to write the validation JSON."
    )
    validate_parser.set_defaults(func=cmd_checkpoint_validate)

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
