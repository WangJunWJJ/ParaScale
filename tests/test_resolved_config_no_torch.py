# -*- coding: utf-8 -*-
# @Time : 2026/6/26 下午12:06
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""ResolvedConfig tests that do not require torch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


def _tmp_case(name: str) -> Path:
    path = Path(".pytest-parascale") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_resolved_config_tracks_value_source_and_override_chain():
    from parascale.configuration import resolve_config

    resolved = resolve_config(
        {
            "parascale": {
                "training_backend": "deepspeed",
                "zero_stage": 1,
                "batch_size": 2,
            },
            "deepspeed_config": {
                "zero_optimization": {"stage": 2},
            },
        },
        cli_overrides={"backend.zero_stage": 3},
    )

    field = resolved.field("backend.zero_stage")

    assert field.value == 3
    assert field.source == "cli"
    assert field.overridden_by == ["cli"]
    assert field.history[0]["source"] == "built-in defaults"
    assert field.history[-1]["path"] == "backend.zero_stage"
    assert resolved.to_dict()["backend"]["zero_stage"] == 3

    with pytest.raises(Exception):
        field.value = 2


def test_resolved_config_preserves_explicit_deepspeed_zero_stage_zero():
    from parascale.configuration import build_deepspeed_final_config, resolve_config

    resolved = resolve_config(
        {
            "parascale": {
                "training_backend": "deepspeed",
                "zero_stage": 0,
                "batch_size": 1,
            }
        }
    )

    final_config = build_deepspeed_final_config(resolved)

    assert resolved.field("backend.zero_stage").value == 0
    assert final_config["zero_optimization"]["stage"] == 0


def test_resolved_config_reports_deepspeed_batch_and_precision_conflicts():
    from parascale.configuration import resolve_config

    resolved = resolve_config(
        {
            "parascale": {
                "training_backend": "deepspeed",
                "precision": "bf16",
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
            },
            "hardware_profile": {"world_size": 2},
            "deepspeed_config": {
                "train_batch_size": 999,
                "fp16": {"enabled": True},
                "optimizer": {"type": "AdamW"},
            },
        }
    )

    warning_codes = {issue.code for issue in resolved.warnings}

    assert "deepspeed_train_batch_mismatch" in warning_codes
    assert "deepspeed_precision_conflict" in warning_codes
    assert "deepspeed_optimizer_conflict" in warning_codes


def test_plan_payload_exposes_resolved_config_without_torch():
    from parascale.commands.plan import build_plan_payload

    payload = build_plan_payload(
        {
            "parascale": {
                "training_backend": "deepspeed",
                "zero_stage": 2,
                "batch_size": 2,
            },
            "training": {"workload": "synthetic_regression"},
            "hardware_profile": {"world_size": 2},
        }
    )

    resolved = payload["resolved_config"]

    assert resolved["backend"]["training_backend"] == "deepspeed"
    assert resolved["backend"]["zero_stage"] == 2
    assert resolved["fields"]["backend.zero_stage"]["source"] == "user config"


def test_train_dry_run_payload_exposes_resolved_config_without_torch():
    from parascale.commands.run import build_train_dry_run_payload

    payload = build_train_dry_run_payload(
        {
            "parascale": {
                "training_backend": "native",
                "batch_size": 4,
            },
            "training": {"workload": "synthetic_regression"},
        }
    )

    assert payload["resolved_config"]["runtime"]["dry_run"] is True
    assert payload["resolved_config"]["training"]["batch_size"] == 4


def test_config_artifact_writer_uses_fixed_names_and_records_deepspeed_config():
    from parascale.configuration import write_config_artifacts

    tmp_path = _tmp_case("config_artifacts_deepspeed")

    artifacts = write_config_artifacts(
        {
            "parascale": {
                "training_backend": "deepspeed",
                "zero_stage": 2,
                "batch_size": 2,
            },
            "training": {"gradient_accumulation_steps": 4},
            "hardware_profile": {"world_size": 2},
        },
        tmp_path,
    )

    resolved_path = tmp_path / "config.resolved.json"
    deepspeed_path = tmp_path / "backend.deepspeed.final.json"
    assert artifacts["resolved_config"] == str(resolved_path)
    assert artifacts["deepspeed_final_config"] == str(deepspeed_path)
    assert json.loads(resolved_path.read_text(encoding="utf-8"))["backend"][
        "zero_stage"
    ] == 2
    assert json.loads(deepspeed_path.read_text(encoding="utf-8"))[
        "gradient_accumulation_steps"
    ] == 4


def test_config_artifact_writer_does_not_emit_deepspeed_file_for_native():
    from parascale.configuration import write_config_artifacts

    tmp_path = _tmp_case("config_artifacts_native")

    artifacts = write_config_artifacts(
        {"parascale": {"training_backend": "native"}},
        tmp_path,
    )

    assert artifacts["deepspeed_final_config"] is None
    assert not (tmp_path / "backend.deepspeed.final.json").exists()


def test_config_artifact_writer_preserves_oom_retry_provenance():
    from parascale.configuration import write_config_artifacts

    tmp_path = _tmp_case("config_artifacts_oom")

    write_config_artifacts(
        {"parascale": {"training_backend": "fsdp", "batch_size": 4}},
        tmp_path,
        emergency_overrides={
            "backend.training_backend": "deepspeed",
            "backend.zero_stage": 3,
            "training.batch_size": 2,
        },
    )

    resolved = json.loads(
        (tmp_path / "config.resolved.json").read_text(encoding="utf-8")
    )
    assert resolved["fields"]["backend.zero_stage"]["source"] == (
        "emergency override"
    )
    assert resolved["training"]["batch_size"] == 2


def test_runtime_artifacts_capture_auto_selected_backend():
    from parascale.config import ParaScaleConfig
    from parascale.runtime.runner_common import _write_runtime_config_artifacts

    run_dir = _tmp_case("runtime_auto_config")
    config_data = {
        "runtime": {"run_dir": str(run_dir)},
        "parascale": {"training_backend": "auto"},
    }
    runtime_config = ParaScaleConfig(training_backend="deepspeed", zero_stage=3)

    artifacts = _write_runtime_config_artifacts(
        config_data,
        runtime_config,
        strategy_selected=True,
    )

    resolved = json.loads(
        Path(artifacts["resolved_config"]).read_text(encoding="utf-8")
    )
    assert resolved["backend"]["training_backend"] == "deepspeed"
    assert resolved["fields"]["backend.training_backend"]["source"] == (
        "strategy/tuner"
    )
    assert Path(artifacts["deepspeed_final_config"]).exists()
