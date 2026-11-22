# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Qualify CLIP CPU lookahead through fresh-process recovery and retrieval."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from parascale.checkpoint import CheckpointManager
from parascale.commands.common import load_config_file
from tests.benchmarks.scripts.validate_clip_recovery import execute_training
from tests.benchmarks.scripts.validate_clip_retrieval import (
    compare_quality,
    validate_manifest,
)
from tests.benchmarks.tools.prepare_flickr8k_retrieval import file_hash


def compare_values(left, right, *, exact=False):
    """Compare nested checkpoint values, preserving failure diagnostics."""
    differing = 0

    def visit(a, b):
        nonlocal differing
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(
                a,
                b,
                atol=0 if exact else 1e-5,
                rtol=0 if exact else 1e-4,
                equal_nan=False,
            )
            differing += int((a != b).sum())
        elif isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys()
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b):
                visit(x, y)
        else:
            assert type(a) is type(b) and a == b, f"metadata differs: {a!r} != {b!r}"

    try:
        visit(left, right)
    except (AssertionError, TypeError, ValueError) as exc:
        return {"passed": False, "error": str(exc)}
    return {"passed": True, "exact": exact, "differing_elements": differing}


def run(command, log):
    with log.open("w", encoding="utf-8") as stream:
        result = execute_training([sys.executable, *command], None, stream)
    if result.returncode:
        raise RuntimeError(f"command failed; inspect {log}")


def validate_input_binding(config, data, model):
    """Bind training to the evaluator's immutable model and train-only shards."""
    if (
        Path(config["model"]["pretrained_model_name_or_path"]).resolve()
        != model.resolve()
        or Path(config["data"]["data_dir"]).resolve() != (data / "train").resolve()
        or config["model"].get("type") != "hf_clip"
        or config["data"].get("type") != "datacomp_wds"
        or not config["data"].get("streaming")
        or config["data"].get("num_workers", 0) != 0
        or config["data"].get("dataset_local_cache_dir")
    ):
        raise ValueError("training model/data binding differs from evaluation inputs")
    manifest = json.loads((data / "manifest.json").read_text())
    validate_manifest(manifest)
    expected = {record["shard"] for record in manifest["splits"]["train"]}
    actual = {
        path.relative_to(data).as_posix() for path in (data / "train").glob("*.tar")
    }
    if actual != expected:
        raise ValueError("training shard set differs")
    for name in actual:
        if file_hash(data / name) != manifest["shards_sha256"][name]:
            raise ValueError("training shard checksum differs")
    assets = {
        path.name: file_hash(path)
        for path in model.iterdir()
        if path.is_file()
        and path.suffix in {".bin", ".safetensors", ".json", ".txt", ".model"}
    }
    if not any(name.endswith((".bin", ".safetensors")) for name in assets):
        raise ValueError("no local pretrained weights")
    return {
        "manifest_sha256": file_hash(data / "manifest.json"),
        "model_assets": assets,
    }


def validate_optimizer_budget(state):
    try:
        optimizer = state["optimizer_state_dict"]
        ids = [key for group in optimizer["param_groups"] for key in group["params"]]
        weights = list(state["model_state_dict"].values())
        assert (
            ids and len(ids) == len(weights) and set(ids) == optimizer["state"].keys()
        )
        for key, weight in zip(ids, weights):
            slot = optimizer["state"][key]
            assert slot["step"].numel() == 1 and slot["step"].item() == 50
            for name in ("exp_avg", "exp_avg_sq"):
                assert (
                    slot[name].shape == weight.shape
                    and torch.isfinite(slot[name]).all()
                )
    except (AssertionError, KeyError, TypeError, AttributeError) as exc:
        return {"passed": False, "error": f"AdamW budget/moments invalid: {exc}"}
    return {"passed": True, "parameters": len(ids), "steps": 50}


def validate_runtime_budget(runtime, rank):
    try:
        data = runtime["data_state"]
        identity = data["state"]["identity"]
        assert (
            runtime["version"] == 1
            and runtime["rank"] == rank
            and runtime["world_size"] == 4
        )
        assert (
            data["consumed_micro_batches"] == 50 and data["state"]["position"] == 1600
        )
        assert data["resume_mode"] == "state_dict" and data["target"] == "dataloader"
        assert (
            identity["rank"] == rank
            and identity["world_size"] == 4
            and identity["batch_size"] == 32
        )
    except (AssertionError, KeyError, TypeError) as exc:
        return {"passed": False, "error": f"rank progress invalid: {exc}"}
    return {"passed": True, "micro_batches": 50, "position": 1600}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Recheck existing artifacts without retraining or reevaluation",
    )
    args = parser.parse_args()
    if not args.check_only:
        args.output.mkdir(parents=True, exist_ok=False)
    base = load_config_file(args.config)
    if (
        base["parascale"]["training_backend"] != "native_ddp"
        or base["parascale"]["precision"] != "fp32"
        or base["training"]["batch_size"] != 32
        or base["training"].get("gradient_accumulation_steps", 1) != 1
    ):
        parser.error("requires fixed FP32 native-DDP4 global128 budget")
    if shutil.disk_usage(args.output).free < 20 * 1024**3:
        raise ValueError("at least20GiB free space required")
    binding = validate_input_binding(base, args.data, args.model)
    if args.check_only:
        return compare_completed(args, binding)
    baseline = args.output / "pretrained.json"
    evaluation = [
        "-m",
        "tests.benchmarks.scripts.validate_clip_retrieval",
        "--data",
        str(args.data),
        "--model",
        str(args.model),
    ]
    print("Evaluating original weights", flush=True)
    run([*evaluation, "--output", str(baseline)], args.output / "pretrained.log")
    for name, steps in (("continuous", 50), ("resumed", 25)):
        folder = args.output / name
        checkpoints = folder / "checkpoints"
        if name == "resumed":
            source = args.output / "continuous/checkpoints/step-00000025"
            shutil.copytree(source, checkpoints / source.name)
            assert file_hash(source / "backend_state.pt") == file_hash(
                checkpoints / source.name / "backend_state.pt"
            )
        config = json.loads(json.dumps(base))
        config["data"]["cpu_prefetch"] = True
        config["parascale"].update(checkpoint_keep_last=2, metrics_history_limit=50)
        config["runtime"]["run_dir"] = str(folder)
        config["training"].update(
            max_steps=steps,
            checkpoint_interval=25,
            checkpoint_dir=str(checkpoints),
            skip_final_checkpoint=False,
        )
        path = args.output / f"{name}.config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        command = [
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
            str(args.output / f"{name}.json"),
        ]
        if name == "resumed":
            command += ["--resume-step", "25"]
        print(f"Training {name}", flush=True)
        run(command, args.output / f"{name}.log")
        result = json.loads((args.output / f"{name}.json").read_text())
        assert result["global_step"] == 50 and result["checkpoint_validation"]["ok"]
        if name == "resumed":
            assert result["resumed_from"]["global_step"] == 25
        manager = CheckpointManager(str(checkpoints))
        assert manager.validate(50).ok
        print(f"Evaluating {name}", flush=True)
        run(
            [
                *evaluation,
                "--checkpoint-root",
                str(checkpoints),
                "--checkpoint-step",
                "50",
                "--baseline",
                str(baseline),
                "--output",
                str(args.output / f"{name}-retrieval.json"),
            ],
            args.output / f"{name}-retrieval.log",
        )
    assert validate_input_binding(base, args.data, args.model) == binding
    return compare_completed(args, binding)


def compare_completed(args, binding):
    resumed_result = json.loads((args.output / "resumed.json").read_text())
    initial = []
    for name in ("continuous", "resumed"):
        config = json.loads((args.output / f"{name}.config.json").read_text())
        assert validate_input_binding(config, args.data, args.model) == binding
        assert config["data"]["cpu_prefetch"] is True
        manager = CheckpointManager(str(args.output / name / "checkpoints"))
        assert manager.validate(50).ok
        assert manager.validate(25).ok
        initial.append(manager.read_manifest(25))
        manifest = manager.read_manifest(50)
        assert manifest.global_step == 50 and manifest.consumed_samples == 6400
        result = json.loads((args.output / f"{name}.json").read_text())
        assert (
            result["global_step"] == 50 and not result["mock"] and not result["dry_run"]
        )
        assert {row["rank"] for row in result["execution"]} == {0, 1, 2, 3}
        assert all(
            row["world_size"] == 4
            and row["optimizer_class"] == "torch.optim.adamw.AdamW"
            for row in result["execution"]
        )
    for key in ("step", "global_step", "consumed_samples", "data_state", "files"):
        assert initial[0].to_dict()[key] == initial[1].to_dict()[key]
        assert resumed_result["resumed_from"][key] == initial[0].to_dict()[key]
    roots = [
        args.output / name / "checkpoints/step-00000050"
        for name in ("continuous", "resumed")
    ]
    payloads = [
        torch.load(root / "backend_state.pt", map_location="cpu", weights_only=True)
        for root in roots
    ]
    checks = {
        name: compare_values(
            payloads[0]["backend_state"][name], payloads[1]["backend_state"][name]
        )
        for name in ("model_state_dict", "optimizer_state_dict")
    }
    for index, payload in enumerate(payloads):
        checks[f"arm{index}_optimizer_budget"] = validate_optimizer_budget(
            payload["backend_state"]
        )
    del payloads
    for rank in range(4):
        runtime = [
            torch.load(
                root / f"rank-{rank:05d}/runtime_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            for root in roots
        ]
        for index, state in enumerate(runtime):
            checks[f"arm{index}_rank{rank}_budget"] = validate_runtime_budget(
                state, rank
            )
        for key in ("data_state", "rng_state"):
            checks[f"rank{rank}_{key}"] = compare_values(
                runtime[0][key], runtime[1][key], exact=True
            )
    quality = [
        json.loads((args.output / f"{name}-retrieval.json").read_text())
        for name in ("continuous", "resumed")
    ]
    baseline = json.loads((args.output / "pretrained.json").read_text())
    for index, result in enumerate(quality):
        expected_checkpoint = {
            "step": 50,
            "payload_sha256": file_hash(roots[index] / "backend_state.pt"),
            "manifest_sha256": file_hash(roots[index] / "manifest.json"),
        }
        assert result["checkpoint"] == expected_checkpoint
        assert result["manifest_sha256"] == binding["manifest_sha256"]
        recomputed = compare_quality(
            result["metrics"],
            baseline,
            binding["manifest_sha256"],
            result["evaluation_identity"],
            result["initial_model_sha256"],
        )
        checks[f"arm{index}_pretrained_quality"] = {
            "passed": all(check["passed"] for check in recomputed.values())
        }
    for key in ("image_to_text_r1", "text_to_image_r1"):
        checks[key] = {
            "passed": quality[1]["metrics"][key] + 0.01
            >= quality[0]["metrics"][key] - 1e-7,
            "continuous": quality[0]["metrics"][key],
            "resumed": quality[1]["metrics"][key],
        }
    passed = all(check["passed"] for check in checks.values())
    result = {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "input_binding": binding,
        "scope": "fixed4rank FP32 global128; floating state atol1e-5/rtol1e-4, exact rank cursor/RNG; not24h certification",
    }
    (
        args.output / ("summary-verified.json" if args.check_only else "summary.json")
    ).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
