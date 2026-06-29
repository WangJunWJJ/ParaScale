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


def python_sources():
    """Return the current Python source set without a duplicate file manifest."""
    yield ROOT / "setup.py"
    for source_root in (ROOT / "parascale", ROOT / "tests"):
        yield from sorted(source_root.rglob("*.py"))


def syntax_check():
    for source_path in python_sources():
        try:
            source = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = source_path.read_text(encoding="utf-8-sig")
        compile(source, str(source_path), "exec")


def run_distributed_smoke(backend, env):
    if importlib.util.find_spec("torch") is None:
        print("\n[ParaScale distributed smoke] skipped: torch is not installed")
        return
    backends = ["fsdp", "deepspeed"] if backend == "all" else [backend]
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
    syntax_check()
    run([sys.executable, "-m", "pytest", "tests", "-q"], env=env)
    if args.distributed:
        run_distributed_smoke(args.backend, env)


if __name__ == "__main__":
    main()
