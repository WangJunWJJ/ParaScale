# -*- coding: utf-8 -*-
# @Time : 2026/7/2 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CLI failure and exit-code contracts without Torch."""

import json

import pytest

from parascale.cli import main


def test_missing_config_returns_structured_config_error(capsys):
    exit_code = main(["plan", "--config", "missing-config.yaml"])
    payload = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["error_type"] == "config_error"
    assert payload["command"] == "plan"


def test_missing_dependency_returns_exit_three(monkeypatch, capsys, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"runtime": {"run_dir": str(tmp_path / "run")}}),
        encoding="utf-8",
    )

    def fail(*_args, **_kwargs):
        raise ImportError("DeepSpeed is not installed")

    monkeypatch.setattr("parascale.commands.run.run_train_from_config", fail)

    exit_code = main(["train", "--config", str(config)])
    payload = json.loads(capsys.readouterr().err)

    assert exit_code == 3
    assert payload["error_type"] == "dependency_error"


def test_checkpoint_validation_failure_returns_exit_five(monkeypatch, capsys):
    monkeypatch.setattr(
        "parascale.commands.checkpoint.build_checkpoint_validation_payload",
        lambda _path: {"validation": {"ok": False, "errors": ["checksum"]}},
    )

    assert main(["checkpoint", "validate", "--checkpoint", "broken"]) == 5
    assert json.loads(capsys.readouterr().out)["validation"]["ok"] is False


def test_failed_benchmark_subrun_returns_exit_six(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "parascale.commands.benchmark.run_benchmark_matrix_from_args",
        lambda _args: {
            "mode": "benchmark_matrix",
            "dry_run": False,
            "run_results": [{"status": "error", "returncode": 1}],
        },
    )

    exit_code = main(
        [
            "benchmark-matrix",
            "--scenario",
            "vlm-lora-hf-clip",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 6
    assert json.loads(capsys.readouterr().out)["run_results"][0]["status"] == "error"


def test_unexpected_error_returns_exit_seventy(monkeypatch, capsys):
    def fail(_args):
        raise KeyError("broken invariant")

    monkeypatch.setattr("parascale.commands.plan.cmd_plan", fail)

    assert main(["plan", "--config", "unused.json"]) == 70
    payload = json.loads(capsys.readouterr().err)
    assert payload["error_type"] == "internal_error"
    assert payload["exit_code"] == 70


def test_debug_mode_reraises_unexpected_error(monkeypatch):
    def fail(_args):
        raise KeyError("broken invariant")

    monkeypatch.setattr("parascale.commands.plan.cmd_plan", fail)
    monkeypatch.setenv("PARASCALE_DEBUG", "1")

    with pytest.raises(KeyError, match="broken invariant"):
        main(["plan", "--config", "unused.json"])
