# -*- coding: utf-8 -*-
# @Time : 2026/6/15 下午5:27
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest

import parascale.commands.launcher as launcher_command
import parascale.commands.scenario as scenario_command
import parascale.commands.stability_report as stability_report
import parascale.commands.stability_resume as stability_resume
from parascale.cli import main
from parascale.commands.benchmark import benchmark_payload_failed


def _tmp_case(name: str) -> Path:
    path = Path(".pytest-parascale") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_benchmark_matrix_dry_run_generates_unified_commands():
    case_dir = _tmp_case("vlm_matrix")
    output = case_dir / "matrix.json"
    rc = main(
        [
            "benchmark-matrix",
            "--scenario",
            "vlm-lora-hf-clip",
            "--backends",
            "native_ddp",
            "fsdp",
            "--output-dir",
            str(case_dir / "runs"),
            "--pipeline-cache",
            "--dataset-local-cache-dir",
            str(case_dir / "dataset-cache"),
            "--cuda-prefetch",
            "--prompt-template-cache",
            "--preprocess-in-workers",
            "--pipeline-cache-max-entries",
            "64",
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload_text = json.dumps(payload, ensure_ascii=False)
    assert payload["mode"] == "benchmark_matrix"
    assert payload["scenario"] == "vlm-lora-hf-clip"
    assert payload["evidence"]["runtime_status"] == "plan_only"
    assert payload["evidence"]["benchmark_matrix"]["planned_commands"] == 2
    assert "torchrun" in payload_text
    assert "native_ddp" in payload_text
    assert "fsdp" in payload_text
    config_text = (case_dir / "runs" / "hf_clip_lora_native_ddp.config.json").read_text(
        encoding="utf-8"
    )
    assert '"pipeline_cache": true' in config_text
    assert '"dataset_local_cache_dir":' in config_text
    assert '"cuda_prefetch": true' in config_text
    assert '"prompt_template_cache": true' in config_text
    assert '"preprocess_in_workers": true' in config_text
    assert '"pipeline_cache_max_entries": 64' in config_text


def test_clip_datacomp_golden_stability_scenario_uses_product_config():
    scenario = scenario_command.benchmark_matrix_scenario_config(
        "clip-datacomp-golden", Namespace(base_config=None, run_id=None)
    )

    assert scenario["base_config"] == "configs/golden/clip_datacomp_vit_b.json"
    assert scenario["runs"] == [{"run_id": "clip_datacomp_vit_b"}]


def test_benchmark_matrix_disables_final_checkpoint_io():
    base_config = json.loads(
        Path("configs/golden/clip_datacomp_vit_b.json").read_text(
            encoding="utf-8"
        )
    )

    config = scenario_command.build_matrix_config(
        scenario="clip-datacomp-golden",
        base_config=base_config,
        run_spec={"run_id": "clip"},
        backend="native_ddp",
        output_dir=Path("runs"),
        max_steps=2,
        warmup_steps=0,
        batch_size=1,
        num_samples=8,
    )

    assert config["training"]["skip_final_checkpoint"] is True


def test_clip_golden_stability_defaults_to_validated_worker_count(monkeypatch):
    case_dir = _tmp_case("clip_golden_stability")
    monkeypatch.setenv("PARASCALE_MODEL_ROOT", "/models")
    monkeypatch.setenv("PARASCALE_DATA_ROOT", "/dataset")
    monkeypatch.setenv("PARASCALE_RUN_ROOT", str(case_dir / "product-runs"))

    rc = main(
        [
            "benchmark-stability",
            "--scenario",
            "clip-datacomp-golden",
            "--output-dir",
            str(case_dir / "runs"),
            "--dry-run",
        ]
    )

    assert rc == 0
    config_path = case_dir / "runs" / "clip_datacomp_vit_b_w4_native_ddp.config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["data"]["num_workers"] == 4
    assert config["data"]["persistent_workers"] is True


def test_kill_restart_dry_run_keeps_full_initial_window(monkeypatch):
    case_dir = _tmp_case("kill_restart_dry_run")
    monkeypatch.setenv("PARASCALE_MODEL_ROOT", "/models")
    monkeypatch.setenv("PARASCALE_DATA_ROOT", "/dataset")
    monkeypatch.setenv("PARASCALE_RUN_ROOT", str(case_dir / "product-runs"))
    output = case_dir / "payload.json"

    rc = main(
        [
            "benchmark-stability",
            "--scenario",
            "clip-datacomp-golden",
            "--max-steps",
            "10",
            "--kill-step",
            "5",
            "--resume-steps",
            "5",
            "--resume-stress",
            "--kill-restart",
            "--output-dir",
            str(case_dir / "runs"),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["interruption_mode"] == "sigkill_after_checkpoint"
    config = json.loads(
        (case_dir / "runs" / "clip_datacomp_vit_b_w4_native_ddp.config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["training"]["max_steps"] == 10
    assert config["training"]["checkpoint_interval"] == 5


def test_yolo_matrix_dry_run_writes_variant_configs():
    case_dir = _tmp_case("yolo_matrix")
    output = case_dir / "matrix.json"
    rc = main(
        [
            "benchmark-matrix",
            "--scenario",
            "yolo-world-large",
            "--variants",
            "m",
            "--backends",
            "native_ddp",
            "--output-dir",
            str(case_dir / "runs"),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert rc == 0
    config_path = case_dir / "runs" / "yolov8m-worldv2_native_ddp.config.json"
    config_text = config_path.read_text(encoding="utf-8")
    assert "/models/yolov8m-worldv2.pt" in config_text
    assert '"training_backend": "native_ddp"' in config_text


def test_matrix_dry_run_expands_batch_size_sweep():
    case_dir = _tmp_case("vlm_real_sweep")
    output = case_dir / "matrix.json"
    rc = main(
        [
            "benchmark-matrix",
            "--scenario",
            "vlm-lora-real",
            "--backends",
            "native_ddp",
            "deepspeed_zero2",
            "--batch-size-sweep",
            "1",
            "2",
            "--output-dir",
            str(case_dir / "runs"),
            "--output",
            str(output),
            "--dry-run",
            "--oom-retry",
        ]
    )

    assert rc == 0
    payload = output.read_text(encoding="utf-8")
    assert '"batch_size_sweep": [' in payload
    assert '"oom_retry": true' in payload
    assert "real_vlm_lora_b1_native_ddp" in payload
    assert "real_vlm_lora_b2_deepspeed_zero2" in payload


def test_vlm_real_native_ddp_enables_find_unused_parameters():
    case_dir = _tmp_case("vlm_native_ddp_unused")
    output = case_dir / "matrix.json"
    rc = main(
        [
            "benchmark-matrix",
            "--scenario",
            "vlm-lora-real",
            "--backends",
            "native_ddp",
            "--output-dir",
            str(case_dir / "runs"),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert rc == 0
    config_text = (
        case_dir / "runs" / "real_vlm_lora_native_ddp.config.json"
    ).read_text(encoding="utf-8")
    assert '"ddp_find_unused_parameters": true' in config_text
    assert '"enable_activation_checkpointing": false' in config_text
    assert '"activation_checkpointing": false' in config_text


def test_matrix_dry_run_zero3_writes_deepspeed_stage_three_config():
    case_dir = _tmp_case("vlm_real_zero3")
    output = case_dir / "matrix.json"
    rc = main(
        [
            "benchmark-matrix",
            "--scenario",
            "vlm-lora-real",
            "--backends",
            "deepspeed_zero3",
            "--batch-size",
            "1",
            "--output-dir",
            str(case_dir / "runs"),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert rc == 0
    config_path = case_dir / "runs" / "real_vlm_lora_deepspeed_zero3.config.json"
    config_text = config_path.read_text(encoding="utf-8")
    assert '"training_backend": "deepspeed"' in config_text
    assert '"zero_stage": 3' in config_text

    run_dir = case_dir / "runs" / "real_vlm_lora_deepspeed_zero3"
    resolved = run_dir / "config.resolved.json"
    final_deepspeed = run_dir / "backend.deepspeed.final.json"
    assert resolved.exists()
    assert final_deepspeed.exists()


def test_stability_dry_run_expands_workers_and_resume_commands():
    case_dir = _tmp_case("stability_vlm")
    output = case_dir / "stability.json"
    rc = main(
        [
            "benchmark-stability",
            "--scenario",
            "vlm-lora-real",
            "--backends",
            "deepspeed_zero2",
            "--dataloader-workers-sweep",
            "0",
            "2",
            "--persistent-workers-sweep",
            "false",
            "true",
            "--prefetch-factor-sweep",
            "2",
            "4",
            "--pin-memory-sweep",
            "false",
            "true",
            "--pipeline-cache",
            "--pipeline-cache-dir",
            str(case_dir / "cache"),
            "--pipeline-cache-max-entries",
            "32",
            "--pipeline-cache-max-bytes",
            "4096",
            "--pipeline-cache-ttl-seconds",
            "120",
            "--prompt-template-cache",
            "--prompt-template-cache-dir",
            str(case_dir / "prompts"),
            "--preprocess-in-workers",
            "--max-steps",
            "10",
            "--checkpoint-interval",
            "5",
            "--resume-stress",
            "--output-dir",
            str(case_dir / "runs"),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = output.read_text(encoding="utf-8")
    assert '"mode": "benchmark_stability"' in payload
    assert '"phase": "train"' in payload
    assert "real_vlm_lora_w0_pw0_pf2_pin0_deepspeed_zero2" in payload
    assert "real_vlm_lora_w2_pw1_pf4_pin1_deepspeed_zero2" in payload
    config_text = (
        case_dir / "runs" / "real_vlm_lora_w2_pw1_pf4_pin1_deepspeed_zero2.config.json"
    ).read_text(encoding="utf-8")
    assert '"dataloader_num_workers": 2' in config_text
    assert '"dataloader_persistent_workers": true' in config_text
    assert '"dataloader_prefetch_factor": 4' in config_text
    assert '"dataloader_pin_memory": true' in config_text
    assert '"pipeline_cache": true' in config_text
    assert '"pipeline_cache_max_entries": 32' in config_text
    assert '"pipeline_cache_max_bytes": 4096' in config_text
    assert '"pipeline_cache_ttl_seconds": 120.0' in config_text
    assert '"prompt_template_cache": true' in config_text
    assert '"preprocess_in_workers": true' in config_text
    assert '"checkpoint_interval": 5' in config_text


def test_stability_helpers_live_in_focused_modules():
    assert callable(stability_resume.append_stability_resume_command)
    row = {
        "run_id": "run",
        "status": "ok",
        "backend": "native_ddp",
        "global_step": 10,
        "throughput": 12.0,
        "peak_memory_gb": 1.5,
        "stable_dataloader_wait_ms": 0.25,
        "step_time_jitter_ratio": 0.1,
        "resumed": True,
        "stable_loss": 1.25,
    }

    formatted = stability_report.format_stability_result_row(row)
    assert formatted.startswith("| run | ok |")
    assert "| 1.250000 |" in formatted


def test_stability_resume_continuity_compares_train_and_resume_rows():
    rows = [
        {
            "run_id": "clip_native_ddp",
            "status": "ok",
            "global_step": 250,
            "stable_loss": 1.2,
            "stable_throughput": 100.0,
            "checkpoint_ok": True,
            "resumed": False,
        },
        {
            "run_id": "clip_native_ddp_resume",
            "status": "ok",
            "global_step": 500,
            "stable_loss": 1.1,
            "stable_throughput": 98.0,
            "checkpoint_ok": True,
            "resumed": True,
        },
    ]

    continuity = stability_report.build_resume_continuity(rows)

    assert len(continuity) == 1
    assert continuity[0]["run_id"] == "clip_native_ddp"
    assert continuity[0]["resume_run_id"] == "clip_native_ddp_resume"
    assert continuity[0]["status"] == "ok"
    assert continuity[0]["initial_step"] == 250
    assert continuity[0]["final_step"] == 500
    assert continuity[0]["loss_ratio"] == pytest.approx(1.1 / 1.2)
    assert continuity[0]["throughput_ratio"] == pytest.approx(0.98)
    assert continuity[0]["checkpoint_ok"] is True


def test_stability_resume_continuity_rejects_loss_or_throughput_jump():
    rows = [
        {
            "run_id": "vlm_native_ddp",
            "status": "ok",
            "global_step": 250,
            "stable_loss": 1.0,
            "stable_throughput": 100.0,
            "checkpoint_ok": True,
            "resumed": False,
        },
        {
            "run_id": "vlm_native_ddp_resume",
            "status": "ok",
            "global_step": 500,
            "stable_loss": 2.0,
            "stable_throughput": 60.0,
            "checkpoint_ok": True,
            "resumed": True,
        },
    ]

    continuity = stability_report.build_resume_continuity(rows)

    assert continuity[0]["status"] == "error"
    assert "loss_jump" in continuity[0]["reasons"]
    assert "throughput_drop" in continuity[0]["reasons"]


def test_failed_train_phase_produces_skipped_resume_dependency():
    failed = {
        "run_id": "clip_native_ddp",
        "status": "error",
        "returncode": 1,
        "error": "distributed timeout",
    }

    skipped = stability_resume.skipped_resume_result(
        run_id="clip_native_ddp",
        backend="native_ddp",
        training_result=failed,
    )

    assert stability_resume.can_run_resume_phase(failed) is False
    assert skipped == {
        "run_id": "clip_native_ddp_resume",
        "phase": "resume",
        "backend": "native_ddp",
        "status": "skipped",
        "reason": "upstream_failed",
        "depends_on": "clip_native_ddp",
        "upstream_returncode": 1,
        "upstream_error": "distributed timeout",
    }

    interrupted = {
        "status": "interrupted",
        "intentional_kill": True,
        "checkpoint_ok": True,
    }
    assert stability_resume.can_run_resume_phase(interrupted) is True


def test_launcher_classifies_distributed_collective_timeout():
    log = (
        "[rank1]: Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=2594, OpType=ALLREDUCE, NumelIn=1)\n"
        "rank : 1 (local_rank: 1)\n"
        "traceback : Signal 6 (SIGABRT) received"
    )

    failure = launcher_command.classify_launcher_failure(log, returncode=1)

    assert failure == {
        "failure_type": "distributed_timeout",
        "failed_rank": 1,
        "collective": "ALLREDUCE",
        "collective_sequence": 2594,
        "signal": "SIGABRT",
    }


def test_launcher_sigkill_after_valid_checkpoint():
    case_dir = _tmp_case("launcher_sigkill")
    checkpoint_root = case_dir / "checkpoints"
    manifest_path = checkpoint_root / "step-00000001" / "manifest.json"
    child_marker = case_dir / "child-survived"
    child_script = (
        "import pathlib,time;time.sleep(1);"
        f"pathlib.Path({str(child_marker)!r}).write_text('alive')"
    )
    script = (
        "import pathlib,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_script!r}],start_new_session=True); "
        f"p=pathlib.Path({str(manifest_path)!r}); "
        "p.parent.mkdir(parents=True,exist_ok=True); "
        "p.write_text('{\"step\":1,\"backend\":\"native\"}'); "
        "time.sleep(30)"
    )

    result = launcher_command.run_matrix_command_until_checkpoint(
        [sys.executable, "-c", script],
        env=dict(os.environ),
        backend="native_ddp",
        run_id="kill_test",
        error_path=case_dir / "kill.error.json",
        log_path=case_dir / "kill.log",
        checkpoint_root=checkpoint_root,
        checkpoint_step=1,
        timeout_seconds=5.0,
        poll_interval_seconds=0.02,
    )

    assert result["status"] == "interrupted"
    assert result["intentional_kill"] is True
    assert result["checkpoint_ok"] is True
    assert result["checkpoint_step"] == 1
    time.sleep(1.2)
    assert not child_marker.exists()


def test_intentional_kill_with_valid_checkpoint_is_not_benchmark_failure():
    payload = {
        "results": [
            {
                "status": "interrupted",
                "returncode": -9,
                "intentional_kill": True,
                "checkpoint_ok": True,
            },
            {"status": "ok", "returncode": 0},
        ]
    }

    assert benchmark_payload_failed(payload) is False


def test_restart_validation_reports_successful_sigkill_resume():
    validation = stability_report.build_restart_validation(
        [
            {
                "run_id": "clip",
                "status": "interrupted",
                "intentional_kill": True,
                "checkpoint_ok": True,
                "checkpoint_step": 10,
            },
            {"run_id": "clip_resume", "status": "ok", "returncode": 0},
        ],
        [
            {
                "run_id": "clip_resume",
                "status": "ok",
                "global_step": 20,
                "resumed": True,
                "checkpoint_ok": True,
            }
        ],
    )

    assert validation == [
        {
            "run_id": "clip",
            "resume_run_id": "clip_resume",
            "status": "ok",
            "checkpoint_step": 10,
            "final_step": 20,
            "intentional_kill": True,
            "checkpoint_ok": True,
            "resumed": True,
            "reasons": [],
        }
    ]


def test_oom_retry_stops_after_non_oom_retry_failure(monkeypatch):
    case_dir = _tmp_case("oom_retry_stop")
    args = Namespace(
        max_steps=1,
        warmup_steps=0,
        batch_size=4,
        num_samples=8,
        nproc_per_node=2,
        master_port=29710,
    )
    attempts = []

    def fake_run_matrix_command(
        command,
        *,
        env,
        backend,
        run_id,
        error_path,
        log_path,
    ):
        attempts.append((backend, run_id))
        return {
            "run_id": run_id,
            "backend": backend,
            "status": "error",
            "error": "checkpoint shape mismatch",
            "returncode": 1,
            "log": str(log_path),
        }

    monkeypatch.setattr(launcher_command, "run_matrix_command", fake_run_matrix_command)
    output_dir = case_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_config = scenario_command.benchmark_matrix_scenario_config(
        "vlm-lora-real", Namespace(base_config=None, run_id=None)
    )
    results = launcher_command.run_oom_retry_sequence(
        scenario="vlm-lora-real",
        scenario_config=scenario_config,
        base_run_spec={"run_id": "real_vlm_lora_b4"},
        failed_backend="fsdp",
        failed_batch_size=4,
        output_dir=output_dir,
        env={},
        args=args,
        commands=[],
    )

    assert [attempt[0] for attempt in attempts] == ["fsdp"]
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert results[0]["retry_terminated"] is True
    assert results[0]["retry_termination_reason"] == "non_oom_failure"
    retry_error = json.loads(
        (
            output_dir / "real_vlm_lora_b4_oom_retry1_fsdp.error.json"
        ).read_text(encoding="utf-8")
    )
    assert retry_error["attempt"] == 1
    assert retry_error["retry_trigger"] == "oom"
    assert retry_error["retry_termination_reason"] == "non_oom_failure"
    retry_dir = output_dir / "real_vlm_lora_b4_oom_retry1_fsdp"
    resolved = json.loads(
        (retry_dir / "config.resolved.json").read_text(encoding="utf-8")
    )
    assert resolved["fields"]["training.batch_size"]["source"] == (
        "emergency override"
    )
