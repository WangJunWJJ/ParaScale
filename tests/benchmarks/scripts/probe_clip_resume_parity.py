# -*- coding: utf-8 -*-
# @Time : 2026/11/20 上午11:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Compare independent continuous CLIP runs against checkpoint continuation."""

import argparse
import json
import math
import shutil
from pathlib import Path

from validate_clip_recovery import run_case, weight_digest, write_json


def compare_state(left, right):
    import torch

    rows = []

    def visit(a, b, name):
        if isinstance(a, torch.Tensor):
            assert (
                isinstance(b, torch.Tensor)
                and a.shape == b.shape
                and a.dtype == b.dtype
            )
            delta = a.double() - b.double()
            rows.append(
                {
                    "name": name,
                    "elements": a.numel(),
                    "different": int((a != b).sum()),
                    "max_abs": float(delta.abs().max()) if a.numel() else 0,
                    "squared_error": float(delta.square().sum()),
                    "squared_reference": float(a.double().square().sum()),
                }
            )
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), name
            for key in a:
                visit(a[key], b[key], f"{name}/{key}")
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), name
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{name}/{index}")
        else:
            assert a == b, f"non-tensor state differs: {name}"

    visit(left, right, "state")
    return {
        "different_elements": sum(row["different"] for row in rows),
        "max_abs": max((row["max_abs"] for row in rows), default=0),
        "relative_l2": math.sqrt(
            sum(row["squared_error"] for row in rows)
            / max(sum(row["squared_reference"] for row in rows), 1e-300)
        ),
        "largest_differences": sorted(
            rows, key=lambda row: row["max_abs"], reverse=True
        )[:12],
    }


def compare_checkpoints(left, right):
    import torch

    a = torch.load(left, map_location="cpu", weights_only=True)["backend_state"]
    b = torch.load(right, map_location="cpu", weights_only=True)["backend_state"]
    return {
        key: compare_state(a[key], b[key])
        for key in ("model_state_dict", "optimizer_state_dict")
    }


def main():
    import torch

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--bucket-cap-mb", type=int)
    parser.add_argument("--find-unused-parameters", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.iterdir()):
        raise ValueError("output must be empty")
    base = json.loads(args.config.read_text())
    if args.bucket_cap_mb:
        base["parascale"]["ddp_bucket_cap_mb"] = args.bucket_cap_mb
    if args.find_unused_parameters:
        base["parascale"]["ddp_find_unused_parameters"] = True
    results = {}
    for name in ("continuous_a", "continuous_b", "resumed"):
        config = json.loads(json.dumps(base))
        folder = args.output / name
        checkpoint = folder / "checkpoints"
        config["runtime"]["run_dir"] = str(folder)
        config["training"].update(
            checkpoint_dir=str(checkpoint), max_steps=20 if name == "resumed" else 40
        )
        config["parascale"]["checkpoint_save_path"] = str(checkpoint)
        if name == "resumed":
            shutil.copytree(
                args.output / "continuous_a/checkpoints/step-00000020",
                checkpoint / "step-00000020",
            )
        config_path = args.output / f"{name}.config.json"
        write_json(config_path, config)
        print(f"Starting {name}", flush=True)
        payload, elapsed = run_case(
            config_path,
            args.output / f"{name}.result.json",
            "0,1,3,4",
            resume=name == "resumed",
        )
        results[name] = {
            "wall_seconds": elapsed,
            "weight_hash": weight_digest(checkpoint / "step-00000040/backend_state.pt"),
            "losses": [row["loss"] for row in payload["metrics_history"]],
        }
        write_json(args.output / "progress.json", results)

    def path(name, step):
        return (
            args.output / name / "checkpoints" / f"step-{step:08d}" / "backend_state.pt"
        )

    comparisons = {}
    for label, left, right in (
        ("continuous_repeat_20", path("continuous_a", 20), path("continuous_b", 20)),
        ("continuous_repeat_40", path("continuous_a", 40), path("continuous_b", 40)),
        ("continuous_vs_resumed_40", path("continuous_a", 40), path("resumed", 40)),
        (
            "previous_resumed_vs_new_continuous_40",
            args.original / "step-00000040/backend_state.pt",
            path("continuous_a", 40),
        ),
    ):
        comparisons[label] = compare_checkpoints(left, right)
    write_json(
        args.output / "comparison.json",
        {
            "runs": results,
            "comparisons": comparisons,
            "bucket_cap_mb_override": args.bucket_cap_mb,
            "find_unused_parameters_override": args.find_unused_parameters,
        },
    )
    print(
        json.dumps(
            {
                key: value["model_state_dict"]["max_abs"]
                for key, value in comparisons.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
