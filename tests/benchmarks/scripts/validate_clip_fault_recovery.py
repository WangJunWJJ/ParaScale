# -*- coding: utf-8 -*-
# @Time : 2026/11/23 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded real CLIP SIGKILL/latest-valid recovery; not a quality/speed gate."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from parascale.checkpoint import CheckpointManager
from parascale.commands.common import load_config_file
from parascale.commands.launcher import run_matrix_command_until_checkpoint
from tests.benchmarks.scripts.validate_clip_prefetch_recovery import (
    compare_values,
    validate_input_binding,
    validate_optimizer_budget,
)
from tests.benchmarks.scripts.validate_clip_recovery import execute_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 15 * 1024**3:
        raise ValueError("at least 15GiB free space required")
    base = load_config_file(args.config)
    policy = base["parascale"]
    assert policy["training_backend"] == "native_ddp" and policy["precision"] == "fp32"
    assert policy["ddp_find_unused_parameters"] and not policy["ddp_static_graph"]
    assert base["data"]["cpu_prefetch"] is True
    assert base["training"]["batch_size"] == 32
    assert base["training"].get("gradient_accumulation_steps", 1) == 1
    data = Path(base["data"]["data_dir"]).parent
    model = Path(base["model"]["pretrained_model_name_or_path"])
    binding = validate_input_binding(base, data, model)

    def command(name, steps, checkpoint_root, interval=0, resume=False, mismatch=False):
        config = json.loads(json.dumps(base))
        config["parascale"].update(checkpoint_keep_last=2, metrics_history_limit=100)
        if mismatch:
            config["parascale"]["ddp_find_unused_parameters"] = False
        config["training"].update(
            max_steps=steps,
            checkpoint_interval=interval,
            checkpoint_dir=str(checkpoint_root),
            skip_final_checkpoint=False,
        )
        config["runtime"]["run_dir"] = str(args.output / name)
        path = args.output / f"{name}.config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        result = [
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
            str(args.output / f"{name}.json"),
        ]
        return result + (["--resume-latest"] if resume else [])

    def run(name, argv, *, expected_success=True):
        print(f"Running {name}", flush=True)
        with (args.output / f"{name}.log").open("w") as stream:
            process = execute_training(argv, None, stream)
        assert (process.returncode == 0) is expected_success, name

    reference = args.output / "continuous/checkpoints"
    interrupted = args.output / "interrupted/checkpoints"
    run("continuous", command("continuous", 4, reference, interval=2))
    # Fork the same process RNG state, not independently initialized Python/NumPy.
    shared_step = "step-00000002"
    shutil.copytree(reference / shared_step, interrupted / shared_step)
    manager = CheckpointManager(str(interrupted))
    assert manager.validate(2).ok
    assert manager.read_manifest(2).to_dict() == CheckpointManager(
        str(reference)
    ).read_manifest(2).to_dict()
    print("Interrupting after complete step3 from shared step2", flush=True)
    killed = run_matrix_command_until_checkpoint(
        command("interrupted", 100, interrupted, interval=1, resume=True),
        env=dict(os.environ),
        backend="native_ddp",
        run_id="clip_fault",
        error_path=args.output / "kill.error.json",
        log_path=args.output / "kill.log",
        checkpoint_root=interrupted,
        checkpoint_step=3,
        timeout_seconds=300,
    )
    (args.output / "interruption.json").write_text(json.dumps(killed, indent=2))
    assert killed["status"] == "interrupted" and killed["intentional_kill"]
    assert killed["checkpoint_ok"] and killed["returncode"] == -9
    origin = manager.read_manifest(3).to_dict()
    assert manager.latest_valid_training_checkpoint()[0].step == 3
    # A separate, explicit incomplete-publication injection tests latest fallback.
    partial = interrupted / "step-00000999"
    partial.mkdir(exist_ok=False)
    (partial / "manifest.json").write_text('{"step":999}', encoding="utf-8")
    run(
        "mismatch",
        command("mismatch", 1, interrupted, resume=True, mismatch=True),
        expected_success=False,
    )
    assert (
        "native training policy mismatch" in (args.output / "mismatch.log").read_text()
    )
    assert not (args.output / "mismatch.json").exists()
    run("resumed", command("resumed", 1, interrupted, resume=True))
    result = json.loads((args.output / "resumed.json").read_text())
    assert result["global_step"] == 4 and result["checkpoint_validation"]["ok"]
    assert result["checkpoint_recovery"]["selected_step"] == 3
    assert any(row["step"] == 999 for row in result["checkpoint_recovery"]["skipped"])
    for key in ("step", "global_step", "files", "data_state"):
        assert result["resumed_from"][key] == origin[key]
    roots = [root / "step-00000004" for root in (reference, interrupted)]
    checks = {}
    states = []
    for root in (reference, interrupted):
        checkpoint = CheckpointManager(str(root))
        assert checkpoint.validate(4).ok
        manifest = checkpoint.read_manifest(4)
        assert manifest.consumed_samples == 512
        assert manifest.metadata["native_training_policy"]["ddp"][
            "find_unused_parameters"
        ]
        states.append(
            torch.load(
                root / "step-00000004/backend_state.pt",
                map_location="cpu",
                weights_only=True,
            )["backend_state"]
        )
    for key in ("model_state_dict", "optimizer_state_dict"):
        checks[key] = compare_values(states[0][key], states[1][key], exact=True)
    for index, state in enumerate(states):
        checks[f"optimizer_budget_{index}"] = validate_optimizer_budget(state, steps=4)
    del states
    for rank in range(4):
        runtime = [
            torch.load(
                root / f"rank-{rank:05d}/runtime_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            for root in roots
        ]
        for state in runtime:
            assert state["rank"] == rank and state["world_size"] == 4
            assert state["data_state"]["consumed_micro_batches"] == 4
            assert state["data_state"]["state"]["position"] == 128
        for key in ("data_state", "rng_state"):
            checks[f"rank{rank}_{key}"] = compare_values(
                runtime[0][key], runtime[1][key], exact=True
            )
    assert validate_input_binding(base, data, model) == binding
    passed = all(row["passed"] for row in checks.values())
    summary = {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "interruption": killed,
        "input_binding": binding,
        "scope": "four-rank FP32 CLIP, shared step2 origin, process-group SIGKILL afterstep3 + incomplete newer manifest injection; no power-loss/quality/performance claim",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
