# -*- coding: utf-8 -*-
# @Time : 2026/7/2 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Verify an installed ParaScale wheel through public CLI entrypoints."""

from __future__ import annotations

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
    with tempfile.TemporaryDirectory(prefix="parascale-clean-install-") as temp:
        workdir = Path(temp)
        config = workdir / "tiny_torch.json"
        config.write_text(
            json.dumps(
                {
                    "parascale": {
                        "training_backend": "native",
                        "precision": "fp32",
                        "checkpoint_save_path": "./runs/tiny/checkpoints",
                        "checkpoint_save_interval": 1,
                    },
                    "runtime": {"device": "cpu"},
                    "model": {
                        "type": "tiny_mlp",
                        "input_dim": 4,
                        "hidden_dim": 8,
                        "output_dim": 2,
                    },
                    "data": {
                        "type": "tensor_random",
                        "batch_size": 2,
                        "input_dim": 4,
                        "output_dim": 2,
                    },
                    "optimizer": {"type": "adamw", "lr": 0.001},
                    "training": {
                        "workload": "torch_tiny_mlp",
                        "max_steps": 2,
                        "checkpoint_dir": "./runs/tiny/checkpoints",
                        "checkpoint_interval": 1,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        version = run(["parascale", "--version"], workdir)
        if "0.1.0" not in version.stdout:
            raise RuntimeError(
                "clean-install version mismatch: " f"{version.stdout.strip()}"
            )
        run(
            ["parascale", "doctor", "--strict", "--require", "torch"],
            workdir,
        )
        run(
            ["parascale", "config", "validate", "--config", str(config)],
            workdir,
        )
        migrated_config = workdir / "tiny_torch.v1.json"
        run(
            [
                "parascale",
                "config",
                "migrate",
                "--config",
                str(config),
                "--output",
                str(migrated_config),
            ],
            workdir,
        )
        run(
            ["parascale", "plan", "--config", str(migrated_config), "--json"],
            workdir,
        )
        run(["parascale", "train", "--config", str(config)], workdir)
        validation = run(
            [
                "parascale",
                "checkpoint",
                "validate",
                "--checkpoint",
                str(workdir / "runs" / "tiny" / "checkpoints"),
            ],
            workdir,
        )
        payload = json.loads(validation.stdout)
        if payload["validation"]["ok"] is not True:
            raise RuntimeError("clean-install checkpoint validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
