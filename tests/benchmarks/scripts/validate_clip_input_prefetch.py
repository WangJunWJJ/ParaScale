# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Fixed-budget paired real-data product CLI timing; no checkpoint speed claims."""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
from pathlib import Path

from parascale.commands.common import load_config_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = load_config_file(args.config)
    if (
        base["parascale"].get("training_backend") != "native_ddp"
        or base["parascale"].get("precision") != "fp32"
        or base["parascale"].get("data_parallel_size") != 4
        or base["training"].get("batch_size") != 32
        or base["training"].get("gradient_accumulation_steps", 1) != 1
        or base["data"].get("num_workers", 0) != 0
        or not base["data"].get("streaming")
        or base["data"].get("type") != "datacomp_wds"
        or base["model"].get("type") != "hf_clip"
    ):
        parser.error("requires Example015 FP32 native-DDP4 WDS budget")
    results = []
    for pair, order in enumerate(((False, True), (True, False), (False, True))):
        runs = {}
        for enabled in order:
            name = f"pair-{pair}-prefetch-{int(enabled)}"
            config = json.loads(json.dumps(base))
            config["data"]["cpu_prefetch"] = enabled
            config["runtime"]["run_dir"] = str(args.output / name)
            config["parascale"]["metrics_history_limit"] = 50
            config["training"].update(
                max_steps=50,
                checkpoint_interval=0,
                skip_final_checkpoint=True,
                checkpoint_dir=str(args.output / name / "checkpoints"),
            )
            path = args.output / f"{name}.config.json"
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            output = args.output / f"{name}.json"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc_per_node=4",
                "--master_addr=127.0.0.1",
                "--master_port=29500",
                "-m",
                "parascale.cli",
                "train",
                "--config",
                str(path),
                "--output",
                str(output),
            ]
            print(f"Starting {name}", flush=True)
            with (args.output / f"{name}.log").open("w") as log:
                with subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process:
                    try:
                        process.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
                    if process.returncode:
                        raise RuntimeError(f"{name} failed: inspect retained log")
            result = json.loads(output.read_text())
            assert result["global_step"] == 50 and len(result["metrics_history"]) == 50
            assert (
                not result["mock"] and not result["synthetic"] and not result["dry_run"]
            )
            measured = result["metrics_history"][10:]
            samples = sum(row["batch_size"] for row in measured)
            seconds = sum(
                row["step_time_seconds"] + row["dataloader_wait_ms"] / 1000
                for row in measured
            )
            runs[str(enabled)] = {
                "steady_images_per_second": samples / seconds,
                "fit_images_per_second": 50 * 128 / result["elapsed_seconds"],
                "mean_wait_ms": statistics.mean(
                    row["dataloader_wait_ms"] for row in measured
                ),
                "losses": [row["loss"] for row in result["metrics_history"]],
            }
        ratio = (
            runs["True"]["steady_images_per_second"]
            / runs["False"]["steady_images_per_second"]
        )
        results.append({"pair": pair, "runs": runs, "speedup": ratio})
        (args.output / "progress.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    speedups = [row["speedup"] for row in results]
    passed = statistics.median(speedups) >= 1.1 and min(speedups) > 1.0
    summary = {
        "status": "passed" if passed else "failed",
        "pairs": results,
        "median_speedup": statistics.median(speedups),
        "scope": "real-data native-DDP4 FP32 global128; 10 warmup +40 measured; correctness/recovery are separate gates",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
