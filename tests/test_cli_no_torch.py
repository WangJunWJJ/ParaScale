# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
import tempfile
from pathlib import Path

import pytest

from parascale.checkpoint import CheckpointManager, CheckpointManifest
from parascale.cli import (
    build_benchmark_dry_run_payload,
    build_checkpoint_validation_payload,
    build_plan_payload,
    build_serve_dry_run_payload,
    build_smoke_report,
    build_train_dry_run_payload,
    load_config_file,
    main,
    run_train_from_config,
)
from parascale.commands.common import emit_json
from parascale.commands.plan import build_plan_payload as command_build_plan_payload
from parascale.commands.run import (
    build_serve_dry_run_payload as command_build_serve_dry_run_payload,
)
from parascale.workloads.yolo import _resolve_model_path


def test_emit_json_skips_nonzero_distributed_rank(monkeypatch):
    output = _workspace_tmp("cli_rank_output") / "payload.json"
    output.unlink(missing_ok=True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")

    emit_json({"rank": 1}, str(output))

    assert not output.exists()


def _sample_config():
    return {
        "parascale": {
            "training_backend": "auto",
            "batching_strategy": "token_budget",
            "max_tokens_per_batch": 2048,
        },
        "model_profile": {
            "total_params": 2_000_000_000,
            "total_memory": 8_000_000_000,
            "num_layers": 32,
            "model_type": "transformer",
        },
        "hardware_profile": {
            "num_gpus": 8,
            "gpus_per_node": 8,
            "gpu_memory": 80 * 1024**3,
            "available_memory": 70 * 1024**3,
        },
        "runtime_profile": {
            "padding_ratio": 0.35,
            "peak_memory_per_gpu": 60 * 1024**3,
            "batch_tokens": 2048,
        },
    }


def _workspace_tmp(tmp_name):
    path = Path(tempfile.gettempdir()) / "parascale-test-runs" / tmp_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_cli_plan_payload_builds_strategy_and_dataloader_plan():
    payload = build_plan_payload(_sample_config())

    assert payload["strategy_plan"]["backend"] == "fsdp"
    assert payload["dataloader_plan"]["batch_sampler"] == "token_budget"
    assert "runtime_tuning" in payload


def test_cli_reexports_productized_command_payload_builders():
    assert build_plan_payload is command_build_plan_payload
    assert build_serve_dry_run_payload is command_build_serve_dry_run_payload


def test_cli_plan_command_defaults_to_summary_and_keeps_json_flag(capsys):
    tmp_path = _workspace_tmp("cli_plan_summary")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_sample_config()), encoding="utf-8")

    assert main(["plan", "--config", str(config_path)]) == 0
    summary = capsys.readouterr().out
    assert "ParaScale plan" in summary
    assert "- backend: fsdp" in summary
    assert "Use --json" in summary

    assert main(["plan", "--config", str(config_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "plan"
    assert payload["strategy_plan"]["backend"] == "fsdp"


def test_load_config_file_accepts_utf8_bom_json():
    tmp_path = _workspace_tmp("cli_utf8_bom")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_sample_config()), encoding="utf-8-sig")

    payload = load_config_file(str(config_path))

    assert payload["parascale"]["training_backend"] == "auto"


def test_load_config_file_expands_nested_environment_references(monkeypatch):
    tmp_path = _workspace_tmp("cli_environment_config")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"path": "${PARASCALE_MODEL_ROOT}/clip"},
                "data": {
                    "shards": ["${PARASCALE_DATA_ROOT}/000.tar"],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PARASCALE_MODEL_ROOT", "/models")
    monkeypatch.setenv("PARASCALE_DATA_ROOT", "/dataset")

    payload = load_config_file(str(config_path))

    assert payload["model"]["path"] == "/models/clip"
    assert payload["data"]["shards"] == ["/dataset/000.tar"]


def test_load_config_file_rejects_unresolved_environment_reference():
    tmp_path = _workspace_tmp("cli_missing_environment_config")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model": {"path": "${PARASCALE_MISSING_MODEL}/clip"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PARASCALE_MISSING_MODEL"):
        load_config_file(str(config_path))


def test_cli_train_dry_run_payload_marks_explicit_dry_run():
    payload = build_train_dry_run_payload(_sample_config())

    assert payload["mode"] == "train"
    assert payload["dry_run"] is True
    assert payload["entrypoint_status"] == "dry_run_only"
    assert payload["runtime_status"] == "plan_only"
    assert payload["strategy_plan"]["backend"] == "fsdp"
    assert payload["dataloader_plan"]["batch_sampler"] == "token_budget"


def test_cli_serve_dry_run_payload_accepts_checkpoint():
    payload = build_serve_dry_run_payload(
        {"serving": {"host": "127.0.0.1"}}, checkpoint="ckpt/manifest.json"
    )

    assert payload["mode"] == "serve"
    assert payload["dry_run"] is True
    assert payload["runtime_status"] == "plan_only"
    assert payload["checkpoint"] == "ckpt/manifest.json"
    assert payload["serving"]["host"] == "127.0.0.1"


def test_cli_checkpoint_validation_payload_accepts_manifest_path():
    tmp_path = _workspace_tmp("cli_checkpoint_validate")
    manager = CheckpointManager(str(tmp_path))
    payload_path = manager.payload_path(3, "backend_state.pt")
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"payload")
    manifest_path = manager.write_manifest(
        CheckpointManifest(
            step=3,
            backend="native",
            files=[
                {"path": "backend_state.pt", "role": "backend_state", "format": "torch"}
            ],
        )
    )

    payload = build_checkpoint_validation_payload(str(manifest_path))

    assert payload["mode"] == "checkpoint_validate"
    assert payload["checkpoint"] == str(manifest_path)
    assert payload["validation"]["ok"] is True
    assert payload["validation"]["checked_files"] == 1


def test_cli_checkpoint_validate_command_writes_json():
    tmp_path = _workspace_tmp("cli_checkpoint_validate_command")
    manager = CheckpointManager(str(tmp_path))
    payload_path = manager.payload_path(1, "backend_state.pt")
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"payload")
    manifest_path = manager.write_manifest(
        CheckpointManifest(
            step=1,
            backend="native",
            files=[
                {"path": "backend_state.pt", "role": "backend_state", "format": "torch"}
            ],
        )
    )
    output_path = tmp_path / "validate.json"

    assert (
        main(
            [
                "checkpoint",
                "validate",
                "--checkpoint",
                str(manifest_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "checkpoint_validate"
    assert payload["validation"]["ok"] is True


def test_cli_benchmark_dry_run_payload_exposes_expected_metrics():
    payload = build_benchmark_dry_run_payload(_sample_config())

    assert payload["mode"] == "benchmark"
    assert payload["dry_run"] is True
    assert payload["runtime_status"] == "plan_only"
    assert "tokens_per_second" in payload["metrics"]
    assert "images_per_second" in payload["metrics"]


def test_cli_train_dry_run_writes_json():
    tmp_path = _workspace_tmp("cli_train_dry_run")
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "train-plan.json"
    config_path.write_text(json.dumps(_sample_config()), encoding="utf-8")

    assert (
        main(
            [
                "train",
                "--config",
                str(config_path),
                "--dry-run",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "train"
    assert payload["dry_run"] is True
    artifact_dir = tmp_path / "train-plan"
    assert payload["config_artifacts"]["run_dir"] == str(artifact_dir)
    assert (artifact_dir / "config.resolved.json").exists()


def test_cli_real_train_rejects_non_native_backend_without_launcher(capsys):
    tmp_path = _workspace_tmp("cli_real_train_failure")
    config_path = tmp_path / "config.json"
    config = _sample_config()
    config["parascale"]["training_backend"] = "deepspeed"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["train", "--config", str(config_path)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error_type"] == "config_error"
    assert "requires a distributed launcher" in error["message"]


def test_cli_auto_backend_uses_native_for_local_smoke(monkeypatch):
    config = _sample_config()
    config["training"] = {"workload": "torch_tiny_mlp", "max_steps": 1}

    def fail(_config):
        raise ImportError("Torch runtime factory workloads require PyTorch.")

    monkeypatch.setattr("parascale.runtime.orchestrator.build_training_components", fail)
    with pytest.raises(ImportError, match="Torch runtime factory"):
        run_train_from_config(config)


def test_resolve_model_path_uses_existing_path():
    model = _workspace_tmp("resolve_model_existing") / "model.pt"
    model.write_bytes(b"weights")

    assert _resolve_model_path(str(model)) == str(model)


def test_resolve_model_path_uses_offline_model_dirs(monkeypatch):
    model_dir = _workspace_tmp("resolve_model_dirs") / "models"
    model_dir.mkdir(exist_ok=True)
    model = model_dir / "yolov8s-worldv2.pt"
    model.write_bytes(b"weights")
    monkeypatch.setenv("PARASCALE_MODEL_DIRS", str(model_dir))

    assert _resolve_model_path("/models/yolov8s-worldv2.pt") == str(model)


def test_cli_real_train_requires_torch_for_synthetic_workload(monkeypatch):
    config = _sample_config()
    config["parascale"]["training_backend"] = "native"
    config["training"] = {"workload": "synthetic_regression", "max_steps": 1}

    def fail(_config):
        raise ImportError("Torch runtime factory workloads require PyTorch.")

    monkeypatch.setattr(
        "parascale.runtime.orchestrator.build_training_components",
        fail,
    )
    with pytest.raises(ImportError, match="require PyTorch"):
        run_train_from_config(config)


def test_cli_doctor_payload_is_lightweight_and_explicit():
    output_path = _workspace_tmp("cli_doctor") / "doctor.json"

    assert main(["doctor", "--output", str(output_path)]) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "doctor"
    assert "python" in payload
    assert "dependencies" in payload
    assert "torch_runtime" in payload
    assert "distributed_runtime" in payload
    assert "ascend_runtime" in payload
    assert "device_backends" in payload
    assert (
        "recommended_backends" in payload["distributed_runtime"]
        or payload["distributed_runtime"]["available"] is False
    )


def test_infer_command_runs_synthetic_clip_and_yolo_without_torch():
    import json

    from parascale.commands.run import run_inference_from_config

    clip_payload = run_inference_from_config(
        {
            "runtime": {"accelerator": "cpu"},
            "inference": {
                "workload": "clip_synthetic",
                "batch_size": 2,
                "num_batches": 1,
            },
        }
    )
    yolo_payload = run_inference_from_config(
        {
            "runtime": {"accelerator": "cpu"},
            "inference": {
                "workload": "yolo_world_synthetic",
                "batch_size": 2,
                "num_batches": 1,
            },
        }
    )

    assert clip_payload["mode"] == "infer"
    assert clip_payload["task"] == "multimodal_embedding"
    assert clip_payload["metrics"]["image_text_pairs"] == 2
    assert yolo_payload["task"] == "vision_detection"
    assert yolo_payload["metrics"]["images"] == 2
    assert json.dumps(yolo_payload)


def test_cli_smoke_report_skip_real_builds_doctor_and_plan_only():
    tmp_path = _workspace_tmp("cli_smoke_report")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_sample_config()), encoding="utf-8")

    report = build_smoke_report(str(config_path), skip_real=True)

    assert report["config"] == str(config_path)
    assert report["steps"]["doctor"]["ok"] is True
    assert report["steps"]["plan"]["ok"] is True
    assert "train" not in report["steps"]
    assert "resume" not in report["steps"]
    assert "serve" not in report["steps"]


def test_cli_smoke_skip_real_writes_json_report():
    tmp_path = _workspace_tmp("cli_smoke_command")
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "smoke-report.json"
    config_path.write_text(json.dumps(_sample_config()), encoding="utf-8")

    assert (
        main(
            [
                "smoke",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
                "--skip-real",
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["config"] == str(config_path)
    assert sorted(payload["steps"].keys()) == ["doctor", "plan"]
    assert payload["steps"]["plan"]["result"]["mode"] == "plan"
