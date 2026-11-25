# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Native stage CLI dispatch, pure planning, and command lifecycle contracts."""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def settings():
    return {
        "parascale": {
            "backend": {"training_backend": "native_parallel"},
            "parallel": {"data_parallel_size": 2, "pipeline_parallel_size": 2},
        },
        "training": {
            "workload": "causal_lm",
            "max_steps": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
        },
        "model": {"pretrained_model_name_or_path": "/missing/model"},
        "data": {"type": "token_corpus", "path": "/missing/corpus", "batch_size": 3},
    }


def forbidden(*args, **kwargs):
    raise AssertionError("legacy path or pre-group artifact write reached")


def test_native_dry_run_is_exact_stage_plan(monkeypatch):
    from parascale.commands import run
    from parascale.runtime.stage_plan import build_stage_training_plan

    data = settings()
    original = copy.deepcopy(data)
    monkeypatch.setattr(run, "build_plan_payload", forbidden)
    payload = run.build_train_dry_run_payload(data)
    expected = build_stage_training_plan(data).to_dict()
    for key, value in expected.items():
        assert payload[key] == value
    assert payload["mode"] == "train"
    assert payload["dry_run"] is True
    assert payload["runtime_status"] == "plan_only"
    assert payload["entrypoint_status"] == "ready"
    assert "torchrun" in payload["next_step"]
    assert "--nproc-per-node=4" in payload["next_step"]
    assert "unsupported" not in payload and "strategy_plan" not in payload
    assert data == original


def test_native_plan_skips_legacy_builders_and_has_honest_summary(monkeypatch):
    from parascale.commands import plan
    from parascale.runtime.stage_plan import build_stage_training_plan

    data = settings()
    monkeypatch.setattr(plan, "build_runtime_context", forbidden)
    monkeypatch.setattr(plan, "build_strategy_plan", forbidden)
    payload = plan.build_plan_payload(data)
    assert payload["native_stage_plan"] == build_stage_training_plan(data).to_dict()
    assert payload["selected_backend"] == "native_parallel"
    assert payload["explain"]["selected_backend"] == "native_parallel"
    assert "strategy_plan" not in payload and "runtime_tuning" not in payload
    summary = plan.format_plan_summary(payload)
    for text in (
        "native_parallel",
        "world_size=4",
        "DP=2",
        "PP=2",
        "TP=1",
        "global_batch_size=24",
        "fp32",
    ):
        assert text in summary
    assert "unknown" not in summary


@pytest.mark.parametrize(
    "resume_step,resume_latest", [(None, False), (7, False), (None, True)]
)
def test_runtime_dispatch_precedes_legacy_validation(
    monkeypatch, resume_step, resume_latest
):
    from parascale.runtime import train_runner

    data = settings()
    calls = []
    module = ModuleType("parascale.runtime.stage_train_runner")

    def stage_runner(config_data, resume_step=None, *, resume_latest=False):
        calls.append((config_data, resume_step, resume_latest))
        return {"native": True}

    module.run_stage_train_from_config = stage_runner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(train_runner, "validate_launcher_context", forbidden)
    monkeypatch.setattr(train_runner, "build_training_components", forbidden)
    monkeypatch.setattr(train_runner, "describe_workload", forbidden)
    assert train_runner.run_train_from_config(
        data, resume_step, resume_latest=resume_latest
    ) == {"native": True}
    assert calls == [(data, resume_step, resume_latest)]


@pytest.mark.parametrize("failure", [None, "runner", "emit", "peer"])
@pytest.mark.parametrize("output", [None, "results/train.json"])
def test_native_command_coordinates_output_and_always_tears_down(
    monkeypatch, failure, output
):
    from parascale.commands import run

    data = settings()
    events = []
    monkeypatch.setattr(run, "load_config_file", lambda path: data)
    monkeypatch.setattr(run, "_write_command_config_artifacts", forbidden)

    def runner(config_data, resume_step=None, *, resume_latest=False):
        assert resume_step is None and resume_latest is True
        assert config_data["runtime"]["run_dir"] == (
            str(Path(output).with_suffix("")) if output else str(Path("runs/train"))
        )
        events.append("runner")
        if failure == "runner":
            raise RuntimeError("runner failed")
        return {"mode": "train", "config_artifacts": {"resolved": "runner-owned"}}

    def emit(payload, output_path):
        assert output_path == output
        assert events[-1] == "training output"
        assert payload["config_artifacts"]["resolved"] == "runner-owned"
        events.append("emit")
        if failure == "emit":
            raise RuntimeError("emit failed")

    def coordinate(operation, *, phase):
        events.append(phase)
        operation()
        if failure == "peer":
            raise RuntimeError("peer failed")

    monkeypatch.setattr(run, "run_train_from_config", runner)
    monkeypatch.setattr(run, "emit_json", emit)
    monkeypatch.setattr(run, "coordinated_call", coordinate, raising=False)
    monkeypatch.setattr(
        run, "destroy_distributed_runtime", lambda: events.append("destroy")
    )
    args = argparse.Namespace(
        config="config.json",
        output=output,
        dry_run=False,
        resume_step=None,
        resume_latest=True,
    )
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run.cmd_train(args)
    else:
        assert run.cmd_train(args) == 0
    assert events == (
        ["runner", "destroy"]
        if failure == "runner"
        else ["runner", "training output", "emit", "destroy"]
    )


def test_native_command_preserves_explicit_run_directory(monkeypatch):
    from parascale.commands import run

    data = settings()
    data["runtime"] = {"run_dir": "custom/run"}
    monkeypatch.setattr(run, "load_config_file", lambda path: data)
    monkeypatch.setattr(run, "_write_command_config_artifacts", forbidden)

    def runner(config_data, resume_step):
        assert config_data["runtime"]["run_dir"] == "custom/run"
        return {}

    monkeypatch.setattr(run, "run_train_from_config", runner)
    monkeypatch.setattr(run, "emit_json", lambda *args: None)
    monkeypatch.setattr(
        run, "coordinated_call", lambda operation, **kwargs: operation(), raising=False
    )
    monkeypatch.setattr(run, "destroy_distributed_runtime", lambda: None)
    assert (
        run.cmd_train(
            argparse.Namespace(config="x", output=None, dry_run=False, resume_step=5)
        )
        == 0
    )


@pytest.mark.parametrize(
    "command", [["train", "--dry-run"], ["plan", "--json"], ["plan"]]
)
def test_actual_cli_with_torch_import_blocked(tmp_path, command):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings()), encoding="utf-8")
    script = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Torch import attempted")
sys.meta_path.insert(0, BlockTorch())
from parascale.cli import main
assert main(sys.argv[1:]) == 0
assert "torch" not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, "-c", script, *command, "--config", str(config)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    if command == ["plan"]:
        assert "global_batch_size=24" in result.stdout
        assert "unknown" not in result.stdout
    else:
        payload = json.loads(result.stdout)
        stage = payload if command[0] == "train" else payload["native_stage_plan"]
        assert stage["world_size"] == 4 and stage["global_batch_size"] == 24
        if command[0] == "train":
            assert payload["entrypoint_status"] == "ready"
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "section,options",
    [
        ("runtime", {"device": "auto", "accelerator": "cpu"}),
        ("runtime", {"accelerator": "npu"}),
        ("launch", {"nnodes": 2}),
        ("distributed", {"nproc_per_node": 2}),
        ("training", {"checkpoint_dir": "ignored"}),
        ("data", {"batching_strategy": "token_budget"}),
    ],
)
def test_plan_and_dry_run_reject_ignored_requests(section, options):
    from parascale.commands.plan import build_plan_payload
    from parascale.commands.run import build_train_dry_run_payload

    data = settings()
    data.setdefault(section, {}).update(options)
    for builder in (build_plan_payload, build_train_dry_run_payload):
        with pytest.raises(ValueError):
            builder(data)
