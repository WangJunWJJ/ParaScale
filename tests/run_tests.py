#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""One-command test runner for ParaScale.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py --distributed
    python tests/run_tests.py --backend deepspeed
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, env=None):
    print(f"\n[ParaScale test] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def syntax_check(paths):
    for path in paths:
        source_path = ROOT / path
        try:
            source = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = source_path.read_text(encoding="utf-8-sig")
        compile(source, str(source_path), "exec")


def run_distributed_smoke(backend, env):
    if importlib.util.find_spec("torch") is None:
        print("\n[ParaScale distributed smoke] skipped: torch is not installed")
        return
    if backend == "all":
        backends = ["fsdp", "deepspeed"]
    else:
        backends = [backend]
    for name in backends:
        run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "tests/distributed_runtime_smoke.py",
                "--backend",
                name,
            ],
            env=env,
        )


def main():
    parser = argparse.ArgumentParser(description="Run ParaScale test suites.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed torchrun smoke tests when torch is installed.",
    )
    parser.add_argument(
        "--backend",
        choices=["all", "native", "fsdp", "deepspeed"],
        default="all",
        help="Select backend smoke tests.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    temp_root = Path(tempfile.gettempdir()) / "parascale-test-runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["OMP_NUM_THREADS"] = "1"
    env["NCCL_DEBUG"] = "WARN"
    os.environ["OMP_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    os.environ["NCCL_DEBUG"] = env["NCCL_DEBUG"]
    env.setdefault("PYTHONPYCACHEPREFIX", str(temp_root / "pycache"))
    env.setdefault(
        "PYTEST_ADDOPTS", f"--basetemp={temp_root / 'pytest'} -p no:cacheprovider"
    )
    if args.backend != "all":
        env["PARASCALE_TEST_BACKEND"] = args.backend

    print("\n[ParaScale test] syntax check")
    syntax_check(
        [
            "setup.py",
            "parascale/config.py",
            "parascale/configuration/__init__.py",
            "parascale/configuration/artifacts.py",
            "parascale/configuration/resolved.py",
            "parascale/configuration/resolver.py",
            "parascale/commands/__init__.py",
            "parascale/commands/benchmark.py",
            "parascale/commands/benchmark_matrix.py",
            "parascale/commands/checkpoint.py",
            "parascale/commands/common.py",
            "parascale/commands/doctor.py",
            "parascale/commands/launcher.py",
            "parascale/commands/plan.py",
            "parascale/commands/run.py",
            "parascale/commands/scenario.py",
            "parascale/commands/smoke.py",
            "parascale/commands/stability.py",
            "parascale/commands/stability_report.py",
            "parascale/commands/stability_resume.py",
            "parascale/commands/vision.py",
            "parascale/contracts/__init__.py",
            "parascale/contracts/backend.py",
            "parascale/contracts/batch.py",
            "parascale/contracts/checkpoint.py",
            "parascale/contracts/metrics.py",
            "parascale/contracts/plan.py",
            "parascale/contracts/workload.py",
            "parascale/communication/__init__.py",
            "parascale/communication/hooks.py",
            "parascale/communication/plan.py",
            "parascale/communication/profiler.py",
            "parascale/data/__init__.py",
            "parascale/data/schema.py",
            "parascale/data/estimators.py",
            "parascale/data/multimodal/__init__.py",
            "parascale/data/multimodal/batch.py",
            "parascale/data/multimodal/cache.py",
            "parascale/data/multimodal/processor.py",
            "parascale/data/multimodal/profiler.py",
            "parascale/data/multimodal/prompt.py",
            "parascale/data/sampler.py",
            "parascale/data/collator.py",
            "parascale/data/plan.py",
            "parascale/data/vision/__init__.py",
            "parascale/data/vision/batch.py",
            "parascale/data/vision/cache.py",
            "parascale/data/vision/collator.py",
            "parascale/data/vision/image_folder.py",
            "parascale/data/vision/preprocessor.py",
            "parascale/data/vision/profiler.py",
            "parascale/data/vision/sampler.py",
            "parascale/data/vision/transforms.py",
            "parascale/data/text/__init__.py",
            "parascale/strategy/__init__.py",
            "parascale/strategy/plan.py",
            "parascale/strategy/planner.py",
            "parascale/strategy/profiler.py",
            "parascale/strategy/tuner.py",
            "parascale/strategy/hetero.py",
            "parascale/strategy/backend_plan.py",
            "parascale/strategy/communication_plan.py",
            "parascale/strategy/data_plan.py",
            "parascale/strategy/device_plan.py",
            "parascale/strategy/inference_plan.py",
            "parascale/core/__init__.py",
            "parascale/core/device/__init__.py",
            "parascale/core/device/ascend.py",
            "parascale/core/device/base.py",
            "parascale/core/device/cpu.py",
            "parascale/core/device/cuda.py",
            "parascale/core/device/registry.py",
            "parascale/core/distributed/__init__.py",
            "parascale/core/distributed/collective.py",
            "parascale/core/distributed/process_group.py",
            "parascale/core/distributed/registry.py",
            "parascale/core/cluster.py",
            "parascale/runtime/__init__.py",
            "parascale/runtime/backend.py",
            "parascale/runtime/backends/__init__.py",
            "parascale/runtime/backends/ascend_native.py",
            "parascale/runtime/backends/base.py",
            "parascale/runtime/backends/deepspeed.py",
            "parascale/runtime/backends/fsdp.py",
            "parascale/runtime/backends/native.py",
            "parascale/runtime/backends/registry.py",
            "parascale/runtime/training/__init__.py",
            "parascale/runtime/training/accumulation.py",
            "parascale/runtime/training/checkpointing.py",
            "parascale/runtime/training/engine.py",
            "parascale/runtime/training/fit_loop.py",
            "parascale/runtime/training/memory.py",
            "parascale/runtime/training/metrics.py",
            "parascale/runtime/training/precision.py",
            "parascale/runtime/training/prefetch.py",
            "parascale/runtime/training/step.py",
            "parascale/runtime/inference/__init__.py",
            "parascale/runtime/inference/batcher.py",
            "parascale/runtime/inference/engine.py",
            "parascale/runtime/inference/memory.py",
            "parascale/runtime/inference/scheduler.py",
            "parascale/runtime/inference/tasks/__init__.py",
            "parascale/runtime/inference/tasks/embedding.py",
            "parascale/runtime/inference/tasks/multimodal.py",
            "parascale/runtime/inference/tasks/vision.py",
            "parascale/runtime/context.py",
            "parascale/runtime/factory.py",
            "parascale/runtime/launcher/__init__.py",
            "parascale/runtime/launcher/deepspeed.py",
            "parascale/runtime/launcher/local.py",
            "parascale/runtime/launcher/torchrun.py",
            "parascale/reporting/__init__.py",
            "parascale/reporting/benchmark.py",
            "parascale/reporting/markdown.py",
            "parascale/reporting/matrix.py",
            "parascale/reporting/profile.py",
            "parascale/reporting/tuner.py",
            "parascale/workloads/__init__.py",
            "parascale/workloads/adapters/__init__.py",
            "parascale/workloads/adapters/yolo.py",
            "parascale/workloads/common.py",
            "parascale/workloads/registry.py",
            "parascale/workloads/serving.py",
            "parascale/workloads/clip.py",
            "parascale/workloads/tiny.py",
            "parascale/workloads/datacomp.py",
            "parascale/workloads/vision.py",
            "parascale/workloads/vlm_cache.py",
            "parascale/workloads/vlm_lora.py",
            "parascale/workloads/yolo.py",
            "parascale/checkpoint/__init__.py",
            "parascale/checkpoint/adapter.py",
            "parascale/checkpoint/manager.py",
            "parascale/checkpoint/manifest.py",
            "parascale/checkpoint/converter.py",
            "parascale/checkpoint/validator.py",
            "parascale/serving/__init__.py",
            "parascale/serving/api.py",
            "parascale/serving/engine.py",
            "parascale/serving/kv_cache.py",
            "parascale/serving/sampler.py",
            "parascale/serving/scheduler.py",
            "parascale/cli.py",
            "parascale/parallel/__init__.py",
            "parascale/parallel/communication.py",
            "parascale/parallel/pipeline.py",
            "parascale/parallel/plan.py",
            "parascale/parallel/sequence.py",
            "parascale/parallel/tensor.py",
            "parascale/optimizers/optimizers.py",
            "parascale/optimizers/zero.py",
            "parascale/quantization/base.py",
            "parascale/quantization/fake_quantize.py",
            "parascale/quantization/observers.py",
            "parascale/quantization/ptq.py",
            "parascale/quantization/qat.py",
            "parascale/quantization/quantized_layers.py",
            "parascale/quantization/utils.py",
            "tests/conftest.py",
            "tests/test_config_no_torch.py",
            "tests/test_strategy_no_torch.py",
            "tests/test_strategy_feedback_no_torch.py",
            "tests/test_data_no_torch.py",
            "tests/test_cli_no_torch.py",
            "tests/test_cli_output_no_torch.py",
            "tests/test_backend_smoke.py",
            "tests/test_train_no_torch.py",
            "tests/test_yolo_no_torch.py",
            "tests/test_architecture_reset_no_torch.py",
            "tests/test_core_architecture_no_torch.py",
            "tests/test_contracts_communication_no_torch.py",
            "tests/test_experimental_parallel_assets.py",
            "tests/test_parallel_plan_no_torch.py",
            "tests/test_v1_runtime_architecture.py",
            "tests/test_vision_synthetic_workload.py",
            "tests/test_vision_profile_real_images.py",
            "tests/test_fourbit_optimizer.py",
            "tests/test_multi_node.py",
            "tests/test_ptq.py",
            "tests/test_quantization.py",
            "tests/distributed_runtime_smoke.py",
            "tests/server_smoke_report.py",
        ]
    )
    run([sys.executable, "-m", "pytest", "tests", "-q"], env=env)
    if args.distributed:
        run_distributed_smoke(args.backend, env)


if __name__ == "__main__":
    main()
