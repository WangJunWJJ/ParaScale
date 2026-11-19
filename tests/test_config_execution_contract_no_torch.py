# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Execution and reporting must share one torch-free configuration contract."""

import subprocess
import sys

import pytest

from parascale.config import ParaScaleConfig
from parascale.configuration import resolve_config


def test_layered_resolution_uses_owned_fields():
    resolved = resolve_config(
        {
            "parascale": {
                "backend": {"training_backend": "deepspeed", "zero_stage": 2},
                "training": {"batch_size": 7, "gradient_accumulation_steps": 3},
            }
        }
    )
    assert resolved.backend["zero_stage"] == 2
    assert resolved.training["batch_size"] == 7


@pytest.mark.parametrize(
    "section, values",
    [
        ("training", {"zero_stage": 2}),
        ("backend", {"batch_size": 2}),
        ("data", "invalid"),
    ],
)
def test_layered_owner_mismatch_rejected(section, values):
    with pytest.raises((ValueError, TypeError), match=section):
        ParaScaleConfig.from_dict({section: values})


def test_normalization_runtime_workload_report_and_backend_agree():
    from parascale.configuration import normalize_execution_config
    from parascale.runtime.backends.deepspeed import DeepSpeedTrainingBackend

    original = {
        "parascale": {"training_backend": "deepspeed"},
        "training": {"batch_size": 7, "gradient_accumulation_steps": 3},
        "deepspeed_config": {"zero_optimization": {"stage": 2}},
    }
    data, config = normalize_execution_config(original)
    assert config.batch_size == data["training"]["batch_size"] == 7
    assert config.gradient_accumulation_steps == 3
    assert config.zero_stage == 2
    assert "batch_size" not in original["parascale"]
    resolved = resolve_config(data)
    assert resolved.field("backend.zero_stage").source == "deepspeed_config.json"
    restored = ParaScaleConfig.from_dict(config.to_dict())
    final = DeepSpeedTrainingBackend(config=restored).build_deepspeed_config()
    assert final["zero_optimization"]["stage"] == 2
    assert final["train_micro_batch_size_per_gpu"] == 7
    assert final["gradient_accumulation_steps"] == 3


def test_existing_precedence_and_strategy_propagate_to_workload():
    from parascale.configuration import normalize_execution_config

    data, config = normalize_execution_config(
        {
            "parascale": {"batch_size": 2, "learning_rate": 0.1},
            "training": {"batch_size": 3, "learning_rate": 0.2, "lr": 0.3},
            "optimizer": {"lr": 0.4},
        },
        strategy_updates={"training.batch_size": 5, "data.tensor_cache": True},
    )
    assert config.batch_size == data["training"]["batch_size"] == 5
    assert config.learning_rate == data["optimizer"]["lr"] == 0.4
    assert data["parascale"]["tensor_cache"] is True
    assert resolve_config(data).field("training.batch_size").source == "strategy/tuner"


def test_normalization_import_does_not_load_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from parascale.configuration import normalize_execution_config; "
                "normalize_execution_config({}); assert 'torch' not in sys.modules"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0])
def test_clip_norm_must_be_finite_positive(value):
    with pytest.raises(ValueError, match="grad_clip_norm"):
        ParaScaleConfig(grad_clip_norm=value)


def test_resume_window_is_in_optimizer_steps():
    from types import SimpleNamespace

    from parascale.runtime.train_runner import _resume_component_config

    data = {"training": {"gradient_accumulation_steps": 3, "max_steps": 4}}
    manifest = SimpleNamespace(
        data_state={
            "resume_mode": "replay_skip",
            "consumed_micro_batches": 6,
        }
    )
    assert (
        _resume_component_config(data, manifest, max_steps=4)["training"]["max_steps"]
        == 6
    )
    manifest.data_state["consumed_micro_batches"] = 5
    with pytest.raises(ValueError, match="accumulation"):
        _resume_component_config(data, manifest, max_steps=4)


def test_cli_data_changes_reach_workload():
    from parascale.configuration import normalize_execution_config

    data, config = normalize_execution_config(
        {}, cli_overrides={"data.tensor_cache": True}
    )
    assert config.tensor_cache is True
    assert data["parascale"]["tensor_cache"] is True


def test_native_deepspeed_defaults_survive_runtime_serialization():
    from parascale.runtime.backends.deepspeed import DeepSpeedTrainingBackend

    config = ParaScaleConfig(
        training_backend="deepspeed",
        deepspeed_config={
            "zero_optimization": {"stage": 2},
            "train_micro_batch_size_per_gpu": 5,
            "gradient_accumulation_steps": 4,
        },
    )
    final = DeepSpeedTrainingBackend(config=config).build_deepspeed_config()
    assert final["zero_optimization"]["stage"] == 2
    assert final["train_micro_batch_size_per_gpu"] == 5
    assert final["gradient_accumulation_steps"] == 4


def test_normalization_preserves_default_and_native_provenance():
    from parascale.configuration import normalize_execution_config

    data, _ = normalize_execution_config(
        {
            "deepspeed_config": {
                "train_micro_batch_size_per_gpu": 5,
            }
        }
    )
    resolved = resolve_config(data)
    assert resolved.field("training.batch_size").source == "deepspeed_config.json"
    assert resolved.field("training.learning_rate").source == "built-in defaults"


def test_data_batch_alias_and_final_deepspeed_artifact_agree():
    from parascale.configuration import (
        build_deepspeed_final_config,
        normalize_execution_config,
    )
    from parascale.runtime.backends.deepspeed import DeepSpeedTrainingBackend

    data, config = normalize_execution_config(
        {
            "parascale": {"training_backend": "deepspeed", "grad_clip_norm": 1.0},
            "training": {"batch_size": 3},
            "data": {"batch_size": 6},
            "deepspeed_config": {"zero_optimization": {"stage": 2}},
        }
    )
    assert config.batch_size == 6
    assert build_deepspeed_final_config(resolve_config(data)) == (
        DeepSpeedTrainingBackend(config=config).build_deepspeed_config()
    )


def test_normalization_json_roundtrip_is_idempotent():
    import json

    from parascale.configuration import normalize_execution_config

    first, config = normalize_execution_config(
        {
            "parascale": {"backend": {"training_backend": "deepspeed"}},
            "deepspeed_config": {"zero_optimization": {"stage": 2}},
        },
        strategy_updates={"training.batch_size": 5, "data.tensor_cache": True},
    )
    second, restored = normalize_execution_config(json.loads(json.dumps(first)))
    assert restored.to_dict() == config.to_dict()
    assert resolve_config(first).to_dict() == resolve_config(second).to_dict()


def test_flat_override_of_layered_value_remains_supported():
    config = ParaScaleConfig.from_dict({"training": {"batch_size": 2}, "batch_size": 5})
    assert config.batch_size == 5


def test_conflicting_deepspeed_owners_rejected():
    from parascale.configuration import normalize_execution_config

    with pytest.raises(ValueError, match="conflicting"):
        normalize_execution_config(
            {
                "parascale": {"deepspeed_config": {"zero_optimization": {"stage": 1}}},
                "deepspeed_config": {"zero_optimization": {"stage": 2}},
            }
        )


def test_layered_train_dry_run_matches_execution_config():
    from parascale.commands.run import build_train_dry_run_payload
    from parascale.configuration import normalize_execution_config

    raw = {
        "parascale": {
            "backend": {"training_backend": "deepspeed", "zero_stage": 2},
            "training": {"batch_size": 3},
        },
        "training": {"gradient_accumulation_steps": 4},
    }
    _, config = normalize_execution_config(raw)
    payload = build_train_dry_run_payload(raw)
    assert payload["backend"] == config.training_backend
    assert payload["resolved_config"]["training"]["batch_size"] == config.batch_size
    assert payload["resolved_config"]["backend"]["zero_stage"] == config.zero_stage


def test_override_owner_mismatch_is_rejected():
    with pytest.raises(ValueError, match="backend.batch_size"):
        resolve_config({}, strategy_updates={"backend.batch_size": 5})


def test_conflicting_training_backend_alias_is_rejected():
    from parascale.configuration import normalize_execution_config

    with pytest.raises(ValueError, match="training.backend"):
        normalize_execution_config({
            "parascale": {"training_backend": "native"},
            "training": {"backend": "deepspeed"},
        })
