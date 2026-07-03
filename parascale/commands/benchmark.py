# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:17
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark command entrypoints and shared CLI arguments."""

from __future__ import annotations

import argparse

from parascale.commands.benchmark_matrix import run_benchmark_matrix_from_args
from parascale.commands.common import emit_json
from parascale.commands.errors import EXIT_BENCHMARK
from parascale.commands.scenario import (
    apply_pipeline_cache_args,
    benchmark_matrix_scenario_config,
)
from parascale.commands.stability import run_benchmark_stability_from_args

__all__ = [
    "add_pipeline_cache_arguments",
    "apply_pipeline_cache_args",
    "benchmark_matrix_scenario_config",
    "cmd_benchmark_matrix",
    "cmd_benchmark_stability",
    "run_benchmark_matrix_from_args",
    "run_benchmark_stability_from_args",
]


def cmd_benchmark_matrix(args: argparse.Namespace) -> int:
    payload = run_benchmark_matrix_from_args(args)
    emit_json(payload, args.output)
    return EXIT_BENCHMARK if benchmark_payload_failed(payload) else 0


def cmd_benchmark_stability(args: argparse.Namespace) -> int:
    payload = run_benchmark_stability_from_args(args)
    emit_json(payload, args.output)
    return EXIT_BENCHMARK if benchmark_payload_failed(payload) else 0


def benchmark_payload_failed(payload: dict) -> bool:
    """Return whether a benchmark command contains a failed launcher result."""

    for key in ("run_results", "retry_results", "results"):
        for result in payload.get(key, []):
            status = str(result.get("status", "")).lower()
            intentional_interrupt = (
                status == "interrupted"
                and bool(result.get("intentional_kill"))
                and bool(result.get("checkpoint_ok"))
            )
            if intentional_interrupt:
                continue
            if status in {"error", "failed"}:
                return True
            returncode = result.get("returncode")
            if returncode is not None and int(returncode) != 0:
                return True
    return False


def add_pipeline_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pipeline-cache",
        action="store_true",
        help="Enable VLM processor output cache.",
    )
    parser.add_argument(
        "--pipeline-cache-dir",
        help="Directory used by the VLM processor output cache.",
    )
    parser.add_argument(
        "--pipeline-cache-max-entries",
        type=int,
        help="Maximum processor/prompt cache files before pruning old entries.",
    )
    parser.add_argument(
        "--pipeline-cache-max-bytes",
        type=int,
        help="Maximum processor/prompt cache bytes before pruning old entries.",
    )
    parser.add_argument(
        "--pipeline-cache-ttl-seconds",
        type=float,
        help="Cache TTL in seconds. A value <= 0 disables TTL expiry.",
    )
    parser.add_argument(
        "--prompt-template-cache",
        action="store_true",
        help="Enable disk-backed prompt template cache.",
    )
    parser.add_argument(
        "--prompt-template-cache-dir",
        help="Optional directory for prompt template cache files.",
    )
    parser.add_argument(
        "--preprocess-in-workers",
        action="store_true",
        help=(
            "Run VLM processor preprocessing inside dataloader workers when "
            "supported."
        ),
    )
    parser.add_argument(
        "--dataset-local-cache-dir",
        help="Optional local directory used to cache WDS tar shards before reading.",
    )
    parser.add_argument(
        "--cuda-prefetch",
        action="store_true",
        help="Enable CUDA stream batch prefetch and non-blocking H2D in TrainEngine.",
    )
    parser.add_argument(
        "--cuda-prefetch-device",
        help="Optional CUDA device for stream prefetch, for example cuda:0.",
    )
