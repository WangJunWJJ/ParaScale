# -*- coding: utf-8 -*-
# @Time : 2026/11/25 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real training checkpoint to standalone HF student release contracts."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from parascale.cli import main  # noqa: E402
from parascale.runtime.train_runner import run_train_from_config  # noqa: E402
from tests.test_causal_lm_inference import infer_config  # noqa: E402
from tests.test_causal_lm_workload import configuration, prepare  # noqa: E402
from tests.test_causal_lm_workload import model_path as model_path  # noqa: E402


@pytest.fixture
def trained(tmp_path, model_path, capsys):
    corpus = prepare(tmp_path, model_path, capsys)

    def train(*, distilled=False, tied=False):
        if tied:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
            model.config.tie_word_embeddings = True
            model.tie_weights()
            model.save_pretrained(model_path)
        config = configuration(model_path, corpus, tmp_path / "checkpoints")
        if distilled:
            from parascale.workloads import build_training_components

            teacher, _, _, _ = build_training_components(config)
            torch.save(teacher.state_dict(), tmp_path / "teacher.pt")
            config["distillation"] = {
                "enabled": True,
                "task": "causal_lm",
                "teacher": {
                    "config": configuration(model_path, corpus, tmp_path / "unused"),
                    "checkpoint": str(tmp_path / "teacher.pt"),
                },
            }
        result = run_train_from_config(config)
        return Path(result["checkpoint"])

    return train


def export(checkpoint, model_path, output):
    from parascale.checkpoint.causal_lm_export import export_causal_lm_checkpoint

    return export_causal_lm_checkpoint(checkpoint, model_path, output)


def rewrite_state(checkpoint, mutate):
    manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["role"] == "backend_state")
    path = checkpoint.parent / entry["path"]
    state = torch.load(path, weights_only=True, map_location="cpu")
    mutate(state["backend_state"])
    torch.save(state, path)
    entry["size_bytes"] = path.stat().st_size
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    "distilled,tied", [(False, False), (True, False), (True, True)]
)
def test_cli_roundtrip_without_teacher_or_original_weights(
    tmp_path, model_path, trained, capsys, distilled, tied
):
    checkpoint = trained(distilled=distilled, tied=tied)
    saved = torch.load(
        checkpoint.parent / "backend_state.pt", weights_only=True, map_location="cpu"
    )
    expected = saved["backend_state"]["model_state_dict"]
    for path in model_path.glob("*.safetensors"):
        path.unlink()
    output = tmp_path / "release"
    assert (
        main(
            [
                "checkpoint",
                "export-causal-lm",
                "--checkpoint",
                str(checkpoint),
                "--model",
                str(model_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads((output / "release.json").read_text(encoding="utf-8"))
    assert report["step"] == 2
    assert report["format"] == "parascale.causal_lm.release.v1"
    assert report["tokenizer_identity"]
    assert all(
        "teacher" not in path.name and path.suffix != ".pt" for path in output.iterdir()
    )
    for entry in report["files"]:
        path = output / entry["path"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["size_bytes"] == path.stat().st_size
    restored = transformers.AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True
    )
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[f"model.{name}"], rtol=0, atol=0)
    if tied:
        assert (
            restored.get_input_embeddings().weight
            is restored.get_output_embeddings().weight
        )
    shutil.rmtree(model_path)
    shutil.rmtree(checkpoint.parent.parent)
    (tmp_path / "teacher.pt").unlink(missing_ok=True)
    config = infer_config(output, tmp_path)
    config_path = tmp_path / "infer.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    summary = tmp_path / "inference-result.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "parascale.cli",
            "infer",
            "--config",
            str(config_path),
            "--output",
            str(summary),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["metrics"]["requests"] == 3
    assert result["execution"]["tokenizer_identity"] == report["tokenizer_identity"]


@pytest.mark.parametrize(
    "mismatch", ["config", "tokenizer", "corrupt", "backend", "missing_identity"]
)
def test_export_rejects_mismatch_without_output(
    tmp_path, model_path, trained, mismatch
):
    checkpoint = trained()
    if mismatch in {"config", "tokenizer"}:
        path = model_path / (
            "config.json" if mismatch == "config" else "tokenizer.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        if mismatch == "config":
            data["rms_norm_eps"] *= 10
        else:
            vocab = data["model"]["vocab"]
            vocab["a"], vocab["b"] = vocab["b"], vocab["a"]
        path.write_text(json.dumps(data), encoding="utf-8")
    elif mismatch == "corrupt":
        with (checkpoint.parent / "backend_state.pt").open("ab") as stream:
            stream.write(b"corrupt")
    else:
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        if mismatch == "backend":
            data["backend"] = "fsdp"
        else:
            data["metadata"]["objective"]["workload"].pop("tokenizer_identity", None)
        checkpoint.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "release"
    with pytest.raises(ValueError):
        export(checkpoint, model_path, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "invalid", ["nonfinite", "dtype", "extra", "missing", "adapter", "tied"]
)
def test_export_rejects_invalid_student_state(tmp_path, model_path, trained, invalid):
    checkpoint = trained(tied=invalid == "tied")

    def mutate(backend):
        state = backend["model_state_dict"]
        key = next(iter(state))
        if invalid == "nonfinite":
            state[key].view(-1)[0] = float("nan")
        elif invalid == "dtype":
            state[key] = state[key].half()
        elif invalid == "extra":
            state["teacher.weight"] = torch.zeros(1)
        elif invalid == "missing":
            del state[key]
        elif invalid == "adapter":
            backend["adapter_only_checkpoint"] = True
        else:
            state["model.lm_head.weight"] = state["model.lm_head.weight"].clone() + 1

    rewrite_state(checkpoint, mutate)
    with pytest.raises((ValueError, RuntimeError)):
        export(checkpoint, model_path, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_export_preserves_existing_destination(tmp_path, model_path, trained):
    checkpoint = trained()
    output = tmp_path / "release"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export(checkpoint, model_path, output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_failed_serialization_does_not_publish(
    tmp_path, model_path, trained, monkeypatch
):
    checkpoint = trained()

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(transformers.LlamaForCausalLM, "save_pretrained", fail)
    with pytest.raises(OSError, match="full disk"):
        export(checkpoint, model_path, tmp_path / "release")
    assert not (tmp_path / "release").exists()
    assert not list(tmp_path.glob(".release-*"))


def test_export_respects_existing_lock(tmp_path, model_path, trained):
    checkpoint = trained()
    lock = tmp_path / ".release-export.lock"
    lock.write_text("other owner", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export(checkpoint, model_path, tmp_path / "release")
    assert lock.read_text(encoding="utf-8") == "other owner"
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("input_tree", ["model", "checkpoint", "checkpoint_root"])
def test_export_rejects_output_inside_inputs(tmp_path, model_path, trained, input_tree):
    checkpoint = trained()
    parent = {
        "model": model_path,
        "checkpoint": checkpoint.parent,
        "checkpoint_root": checkpoint.parent.parent,
    }[input_tree]
    with pytest.raises(ValueError, match="overlap"):
        export(checkpoint, model_path, parent / "release")
    assert not (parent / "release").exists()


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_export_accepts_fp32_training_from_low_precision_source_config(
    tmp_path, model_path, trained, dtype
):
    path = model_path / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["torch_dtype"] = dtype
    config["dtype"] = dtype
    path.write_text(json.dumps(config), encoding="utf-8")
    checkpoint = trained()
    report = export(checkpoint, model_path, tmp_path / "release")
    assert report["step"] == 2
    restored = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "release")
    assert {p.dtype for p in restored.parameters()} == {torch.float32}


def test_export_rejects_model_not_supported_by_product_inference(
    tmp_path, model_path, trained
):
    model = transformers.MistralForCausalLM(
        transformers.MistralConfig(
            vocab_size=7,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.save_pretrained(model_path)
    checkpoint = trained()
    with pytest.raises(ValueError, match="Llama"):
        export(checkpoint, model_path, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def _ddp_export_worker(rank, root, model_path, corpus):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    root = Path(root)
    torch.distributed.init_process_group(
        "nccl",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        config = configuration(model_path, corpus, root / "ddp-checkpoint")
        config["parascale"]["training_backend"] = "native_ddp"
        config["hardware_profile"] = {"world_size": 2, "gpus_per_node": 2}
        result = run_train_from_config(config)
        assert result["train_device"] == f"cuda:{rank}"
        assert result["strategy_plan"]["dp_size"] == 2
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_gpu_ddp_checkpoint_exports_exact_student(tmp_path, model_path, capsys):
    corpus = prepare(tmp_path, model_path, capsys)
    context = torch.multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_ddp_export_worker,
            args=(rank, str(tmp_path), str(model_path), str(corpus)),
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + 120
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    root = tmp_path / "ddp-checkpoint"
    report = export(root, model_path, tmp_path / "release")
    state = torch.load(
        root / "step-00000002" / "backend_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "release")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value,
            state["backend_state"]["model_state_dict"][f"module.model.{key}"],
            rtol=0,
            atol=0,
        )
    assert report["step"] == 2
