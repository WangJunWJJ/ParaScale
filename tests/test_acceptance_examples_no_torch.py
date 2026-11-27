# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Acceptance orchestration must not confuse execution with qualification."""

import json
import os
import sys
from pathlib import Path

import pytest

from examples.acceptance.cases import CASES, GROUPS, build_plan, validate_environment
from examples.acceptance.checks import check_inference, check_training, verify_stream
from examples.acceptance.run import execute_suite, run_process, validate_output
from examples.acceptance.storage import parameter_bound, required_free, storage_plan


def environment():
    return {"schema_version": 1, "gpu_ids": [0, 1, 3, 4], "assets": {}}


def test_catalog_has_eight_groups_and_explicit_three_dimensional_limit():
    assert set(GROUPS) == {
        "installation", "parallel_training", "llama7b_finetuning", "quantization",
        "distillation", "pruning", "generative_inference", "stability",
    }
    assert CASES["parallel_training.dp2_tp2_pp2"].gpus == 8
    assert all(case.requirements for case in CASES.values())


@pytest.mark.parametrize("update", [
    {"gpu_ids": [0, 0]}, {"gpu_ids": [0, 1, 2, 3, 4]}, {"gpu_ids": [True]},
    {"schema_version": 2}, {"timeout_seconds": 0}, {"unknown": True},
    {"budgets": {"train_steps": True}}, {"budgets": {"precision": "int8"}},
    {"budgets": {"sequence_length": 1}},
])
def test_invalid_environment_fails_before_execution(update):
    with pytest.raises(ValueError):
        validate_environment({**environment(), **update})


def test_parallel_plan_preserves_global_batch_and_product_entry(tmp_path):
    env = validate_environment({**environment(), "assets": {
        "llama_model": str(tmp_path / "model"), "token_corpus": str(tmp_path / "tokens"),
    }})
    plan = build_plan(CASES["parallel_training.dp2_pp2"], env, tmp_path / "run", "train", "infer")
    config = plan["configs"]["train.json"]
    assert config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"] * 2 == 8
    assert config["parascale"]["pipeline_parallel_size"] == 2
    assert "--no-python" in plan["commands"][0]
    assert "train" in plan["commands"][0] and "run" in plan["commands"][0]
    assert "--resume-latest" in plan["commands"][1]


def test_output_must_be_new_outside_repository_and_assets(tmp_path):
    repo, asset = tmp_path / "repo", tmp_path / "model"
    repo.mkdir()
    asset.mkdir()
    for path in (repo / "results", asset / "results", tmp_path, asset):
        with pytest.raises(ValueError):
            validate_output(path, repo, [asset])
    assert validate_output(tmp_path / "results", repo, [asset]) == tmp_path / "results"


def test_missing_assets_and_unimplemented_quality_never_launch(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("blocked case launched a process")

    result = execute_suite(environment(), tmp_path / "results", [
        "parallel_training.dp2", "pruning.quality", "parallel_training.dp2_tp2_pp2",
    ], mode="acceptance", process=forbidden)
    assert result["status"] == "BLOCKED"
    assert all(row["status"] == "BLOCKED" for row in result["cases"])
    assert result["full_acceptance_passed"] is False
    assert (tmp_path / "results" / "protocol.json").is_file()


def inference_payload():
    return {
        "dry_run": False, "mock": False, "synthetic": False,
        "runtime_status": "real_local", "task": "causal_lm_generation",
        "metrics": {"requests": 2, "generated_tokens": 4, "generated_tokens_per_second": 2.0},
        "output_stream": {"complete": True, "records": 2},
        "execution": {"model_assets_sha256": {"model": "abc"}, "input_sha256": "abc"},
    }


@pytest.mark.parametrize("field,value", [("synthetic", True), ("mock", True), ("dry_run", True)])
def test_inference_rejects_simulated_evidence(field, value):
    payload = inference_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        check_inference(payload, "transformer")


def test_inference_rejects_nonfinite_and_incomplete_evidence():
    payload = inference_payload()
    payload["metrics"]["generated_tokens_per_second"] = float("nan")
    with pytest.raises(ValueError):
        check_inference(payload, "transformer")
    payload = inference_payload()
    payload["output_stream"]["complete"] = False
    with pytest.raises(ValueError):
        check_inference(payload, "transformer")
    assert check_inference(inference_payload(), "transformer")["requests"] == 2


def training_payload(step, initial):
    return {
        "dry_run": False, "mock": False, "synthetic": False,
        "backend": "native_parallel", "runtime_status": "real_distributed",
        "initial_step": initial, "global_step": step, "completed_updates": 2,
        "checkpoint": {"step": step, "path": "checkpoint"},
        "resume_selection": {"step": initial},
        "scored_tokens_per_second": 5.0, "metrics_history": [{"loss": 2.0}],
        "timing_scope": "includes checkpoints",
    }


def test_training_requires_actual_continuation_and_checkpoint():
    train, resume = training_payload(2, 0), training_payload(4, 2)
    assert check_training(train, resume, 2)["resumed_step"] == 4
    resume["initial_step"] = 0
    with pytest.raises(ValueError):
        check_training(train, resume, 2)


def test_unknown_case_and_existing_output_rejected(tmp_path):
    with pytest.raises(ValueError):
        execute_suite(environment(), tmp_path / "results", ["typo"])
    assert not (tmp_path / "results").exists()
    with pytest.raises(ValueError):
        execute_suite(environment(), tmp_path, ["installation.entries"])


def test_subprocess_records_failure_and_timeout(tmp_path):
    pytest.importorskip("psutil")
    failed = run_process([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, {}, 10, tmp_path / "bad")
    assert failed["returncode"] == 3
    timed = run_process([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, {}, 0.1, tmp_path / "slow")
    assert timed["timed_out"] is True
    assert (tmp_path / "slow.stderr.log").exists()


@pytest.mark.skipif(os.name == "nt", reason="separate POSIX sessions match torchrun workers")
@pytest.mark.parametrize("interrupted", [False, True])
def test_timeout_stops_separate_session_worker(tmp_path, monkeypatch, interrupted):
    psutil = pytest.importorskip("psutil")
    import subprocess
    pid_file = tmp_path / "worker.pid"
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    original_wait = subprocess.Popen.wait
    raised = False
    def interrupt_once(process, *args, **kwargs):
        nonlocal raised
        if interrupted and pid_file.exists() and not raised:
            raised = True
            raise KeyboardInterrupt
        return original_wait(process, *args, **kwargs)
    monkeypatch.setattr(subprocess.Popen, "wait", interrupt_once)
    try:
        if interrupted:
            with pytest.raises(KeyboardInterrupt):
                run_process([sys.executable, "-c", code], tmp_path, {}, 3, tmp_path / "tree")
        else:
            result = run_process([sys.executable, "-c", code], tmp_path, {}, 1, tmp_path / "tree")
            assert result["timed_out"]
        assert pid_file.exists()
        pid = int(pid_file.read_text())
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if pid_file.exists():
            try:
                psutil.Process(int(pid_file.read_text())).kill()
            except psutil.NoSuchProcess:
                pass


def test_installation_functional_pass_is_not_production_acceptance(tmp_path):
    def completed(command, cwd, env, timeout, log):
        name = "parascale-train" if int(log.name) < 2 else "parascale-infer"
        Path(f"{log}.stdout.log").write_text(f"{name} 0.2.0\n")
        return {"returncode": 0, "timed_out": False}

    report = execute_suite(environment(), tmp_path / "results", ["installation.entries"],
                           mode="functional", process=completed, tools=(sys.executable, sys.executable))
    assert report["status"] == "PASS"
    assert report["full_acceptance_passed"] is False
    assert report["unselected_cases"]
    formal = execute_suite(environment(), tmp_path / "formal", ["installation.entries"],
                           mode="acceptance", process=completed, tools=(sys.executable, sys.executable))
    assert formal["cases"][0]["functional_status"] == "PASS"
    assert formal["status"] == "BLOCKED"
    assert json.loads((tmp_path / "formal" / "summary.json").read_text())["status"] == "BLOCKED"


def test_wrong_executable_cannot_pass_entrypoint_checks(tmp_path):
    def wrong(command, cwd, env, timeout, log):
        Path(f"{log}.stdout.log").write_text("Python 3.10\n")
        return {"returncode": 0, "timed_out": False}

    report = execute_suite(environment(), tmp_path / "results", ["installation.entries"],
                           mode="functional", process=wrong, tools=(sys.executable, sys.executable))
    assert report["status"] == "FAIL"


def test_protocol_source_identity_excludes_bytecode(tmp_path):
    report = execute_suite(environment(), tmp_path / "results", ["pruning.quality"])
    protocol = json.loads(Path(report["protocol"]).read_text())
    assert set(protocol["source_sha256"]) == {"run.py", "checks.py", "cases.py", "storage.py", "processes.py", "preflight.py"}


def test_storage_accounts_for_replicas_retention_and_remaining_work(tmp_path):
    header = json.dumps({"weight": {"dtype": "F16", "shape": [10, 20]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    assert parameter_bound(tmp_path) == 200
    env = validate_environment({**environment(), "assets": {"llama_model": str(tmp_path)}})
    plan = storage_plan(CASES["parallel_training.dp2"], env)
    assert plan["checkpoint_payload_bytes"] == 200 * 12 * 2
    assert required_free(plan, 0) - required_free(plan, 1) == 200 * 12 * 2


def test_storage_includes_unsharded_tensor_parallel_replicas(tmp_path):
    tensors = {
        "model.embed_tokens.weight": {"dtype": "F16", "shape": [10, 20]},
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "F16", "shape": [20, 20]},
        "model.layers.0.self_attn.o_proj.bias": {"dtype": "F16", "shape": [20]},
        "model.norm.weight": {"dtype": "F16", "shape": [20]},
    }
    header = json.dumps(tensors).encode()
    (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    env = validate_environment({**environment(), "assets": {"llama_model": str(tmp_path)}})
    plan = storage_plan(CASES["parallel_training.tp2"], env)
    assert plan["checkpoint_payload_bytes"] == (400 + 2 * (200 + 20 + 20)) * 12


def test_insufficient_disk_is_blocked_without_launch(tmp_path, monkeypatch):
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("examples.acceptance.run.shutil.disk_usage", lambda path: usage(10, 9, 1))
    def forbidden(*args, **kwargs):
        pytest.fail("disk-blocked command launched")
    report = execute_suite(environment(), tmp_path / "run", ["installation.entries"],
                           process=forbidden, tools=(sys.executable, sys.executable))
    assert report["status"] == "BLOCKED"
    assert "disk" in str(report["cases"][0]["reasons"])


@pytest.mark.parametrize("record", [
    {"file": "missing.png", "sha256": "incorrect"},
    {"file": "../outside.png", "sha256": "incorrect"},
])
def test_diffusion_rejects_missing_or_external_images(tmp_path, record):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    payload = {"output_stream": {"path": str(manifest), "records": 1}}
    with pytest.raises(ValueError):
        verify_stream(payload, "diffusion", tmp_path)


def test_diffusion_checks_image_hash_dimensions_and_duplicates(tmp_path):
    import hashlib
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.png"
    image.new("RGB", (8, 8)).save(path)
    record = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "width": 8, "height": 8}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    payload = {"output_stream": {"path": str(manifest), "records": 1}}
    verify_stream(payload, "diffusion", tmp_path)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        verify_stream(payload, "diffusion", tmp_path)
    image.new("RGB", (8, 8)).save(path)
    manifest.write_text((json.dumps(record) + "\n") * 2)
    payload["output_stream"]["records"] = 2
    with pytest.raises(ValueError):
        verify_stream(payload, "diffusion", tmp_path)


def test_resource_blocked_layout_does_not_abort_suite_on_batch_divisibility(tmp_path):
    env = {**environment(), "gpu_ids": [0, 1], "budgets": {"global_batch": 2}}
    result = execute_suite(env, tmp_path / "run", ["parallel_training.dp4", "pruning.quality"])
    assert result["status"] == "BLOCKED"
    assert len(result["cases"]) == 2


def test_cleanup_failure_prevents_remaining_launches(tmp_path, monkeypatch):
    from dataclasses import replace

    from examples.acceptance.processes import ProcessCleanupError

    monkeypatch.setitem(CASES, "installation.second", replace(CASES["installation.entries"], id="installation.second"))
    calls = []

    def failed_cleanup(*args):
        calls.append(args)
        raise ProcessCleanupError("worker still alive")

    result = execute_suite(environment(), tmp_path / "run",
                           ["installation.entries", "installation.second"],
                           process=failed_cleanup, tools=(sys.executable, sys.executable))
    assert len(calls) == 1
    assert result["status"] == "FAIL"
    assert result["cases"][1]["status"] == "BLOCKED"
    assert "cleanup failed" in str(result["cases"][1]["reasons"])
