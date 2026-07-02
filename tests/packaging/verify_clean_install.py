# -*- coding: utf-8 -*-
# @Time : 2026/7/2 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Verify an installed ParaScale wheel through public CLI entrypoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()

    with tempfile.TemporaryDirectory(prefix="parascale-clean-install-") as temp:
        workdir = Path(temp)
        config = root / "configs" / "quickstart" / "tiny_torch.yaml"
        run(["parascale", "doctor"], workdir)
        run(["parascale", "plan", "--config", str(config), "--json"], workdir)
        run(["parascale", "train", "--config", str(config)], workdir)
        validation = run(
            [
                "parascale",
                "checkpoint",
                "validate",
                "--checkpoint",
                str(workdir / "runs" / "quickstart" / "tiny_torch"),
            ],
            workdir,
        )
        payload = json.loads(validation.stdout)
        if payload["validation"]["ok"] is not True:
            raise RuntimeError("clean-install checkpoint validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
