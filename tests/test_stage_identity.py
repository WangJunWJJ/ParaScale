# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CPU-only effective model and local asset checkpoint identity tests."""

import gc
import json
import weakref
from types import SimpleNamespace

import pytest


class Record(SimpleNamespace):
    pass


@pytest.fixture
def inputs(tmp_path):
    from parascale.optimizers.spec import OptimizerSpec

    (tmp_path / "config.json").write_text('{"use_cache": true}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    stage = Record(
        effective_model_config={"use_cache": False, "vocab_size": 7}, vocab_size=7
    )
    cursor = Record(
        corpus=Record(
            manifest={"tokenizer": {"sha256": "tokenizer"}},
            identity={"sha256": "corpus"},
        )
    )
    plan = Record(
        workload=Record(model_path=str(tmp_path)), config=Record(precision="fp32")
    )
    plan.optimizer_spec = OptimizerSpec.from_config(
        {"optimizer": {"type": "adamw", "lr": 0.001}}
    )
    return stage, cursor, plan


def build(*args):
    from parascale.runtime.stage_identity import build_stage_checkpoint_context

    return build_stage_checkpoint_context(*args)


def test_effective_configuration_is_detached_before_source_release(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import parascale.parallel.llama as llama
    from parascale.parallel import DistributedParallelConfig
    from parascale.workloads import causal_lm

    references, configs = [], []

    def load(*args, **kwargs):
        source = torch.nn.Module()
        source.config = transformers.LlamaConfig(vocab_size=7, use_cache=False)
        source.config._attn_implementation = "eager"
        source.config._name_or_path = "volatile"
        source.config._commit_hash = "volatile"
        configs.append(source.config)
        references.append(weakref.ref(source))
        return source, Record()

    monkeypatch.setattr(causal_lm, "_load_source", load)
    monkeypatch.setattr(
        llama, "LlamaStage", lambda *args, **kwargs: Record(vocab_size=7)
    )
    stage, _, _ = causal_lm.build_causal_lm_stage_components(
        {
            "model": {"pretrained_model_name_or_path": "unused"},
            "data": {"type": "token_corpus", "path": "unused"},
        },
        DistributedParallelConfig(),
        rank=0,
        world_size=1,
    )
    assert references[0]() is None
    snapshot = stage.effective_model_config
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    assert snapshot["use_cache"] is False
    assert snapshot["_attn_implementation"] == "eager"
    assert "_name_or_path" not in snapshot
    assert "_commit_hash" not in snapshot
    configs[0].use_cache = True
    assert snapshot["use_cache"] is False


def test_context_uses_effective_config_and_execution_identity(inputs):
    import torch
    import transformers

    from parascale.parallel.checkpoint_context import canonical_checkpoint_context

    stage, cursor, plan = inputs
    result = build(*inputs)
    assert result["model"]["config"] == stage.effective_model_config
    assert result["model"]["config"]["use_cache"] is False
    assert result["tokenizer"] == cursor.corpus.manifest["tokenizer"]
    assert result["data"] == cursor.corpus.identity
    assert result["precision"] == plan.config.precision
    assert result["loss"] == {
        "type": "causal_next_token_v1",
        "shift": 1,
        "ignore_index": -100,
        "vocab_size": 7,
    }
    assert result["execution"]["torch"] == torch.__version__
    assert result["execution"]["transformers"] == transformers.__version__
    hashes = result["execution"]["implementation_sha256"]
    assert "parallel/llama.py" in hashes
    assert "workloads/causal_lm.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())
    assert json.loads(canonical_checkpoint_context(result)) == result


@pytest.mark.parametrize(
    "change", ["effective", "config", "weights", "tokenizer", "data", "precision"]
)
def test_changes_invalidate_context(inputs, tmp_path, change):
    before = build(*inputs)
    stage, cursor, plan = inputs
    if change == "effective":
        stage.effective_model_config["use_cache"] = True
    elif change == "config":
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    elif change == "weights":
        (tmp_path / "model.safetensors").write_bytes(b"changed")
    elif change == "tokenizer":
        cursor.corpus.manifest["tokenizer"]["sha256"] = "changed"
    elif change == "data":
        cursor.corpus.identity["sha256"] = "changed"
    else:
        plan.config.precision = "bf16"
    assert build(*inputs) != before


def test_context_does_not_retain_stage_or_mutable_inputs(inputs):
    original, cursor, plan = inputs
    # Pytest retains fixture values; use a separately owned stage for this check.
    stage = Record(
        effective_model_config=dict(original.effective_model_config), vocab_size=7
    )
    reference = weakref.ref(stage)
    result = build(stage, cursor, plan)
    stage.effective_model_config["use_cache"] = True
    cursor.corpus.identity["sha256"] = "changed"
    cursor.corpus.manifest["tokenizer"]["sha256"] = "changed"
    assert result["model"]["config"]["use_cache"] is False
    assert result["data"]["sha256"] == "corpus"
    assert result["tokenizer"]["sha256"] == "tokenizer"
    del stage
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "weight,index",
    [
        ("pytorch_model.bin", None),
        ("model.safetensors", None),
        ("part-01.safetensors", "model.safetensors.index.json"),
        ("part-01.bin", "pytorch_model.bin.index.json"),
    ],
)
def test_resolves_existing_weights_and_hashes_index(
    inputs, tmp_path, weight, index, monkeypatch
):
    import torch

    (tmp_path / "model.safetensors").unlink()
    (tmp_path / weight).write_bytes(b"payload")
    if index:
        (tmp_path / index).write_text(
            json.dumps({"weight_map": {"a": weight, "b": weight}})
        )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("must not load tensors")
    )
    result = build(*inputs)
    assets = result["model"]["assets_sha256"]
    assert set(assets) == {"config.json", weight} | ({index} if index else set())
    if index:
        (tmp_path / index).write_text(json.dumps({"weight_map": {"a": weight}}))
        assert build(*inputs) != result


@pytest.mark.parametrize(
    "kind", ["absent", "missing_shard", "empty_map", "escape", "empty_config"]
)
def test_invalid_model_identity_is_rejected(inputs, tmp_path, kind):
    (tmp_path / "model.safetensors").unlink()
    if kind in {"missing_shard", "empty_map", "escape"}:
        weights = (
            {}
            if kind == "empty_map"
            else {"a": "../outside.bin" if kind == "escape" else "missing.bin"}
        )
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps({"weight_map": weights})
        )
    if kind == "empty_config":
        (tmp_path / "model.safetensors").write_bytes(b"weights")
        inputs[0].effective_model_config = {}
    with pytest.raises(ValueError, match="weight|model|config|shard"):
        build(*inputs)


def test_weight_resolution_prefers_safetensors_over_bin(inputs, tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"unused")
    result = build(*inputs)
    assert set(result["model"]["assets_sha256"]) == {
        "config.json",
        "model.safetensors",
    }
    (tmp_path / "pytorch_model.bin").write_bytes(b"changed unused asset")
    assert build(*inputs) == result


def test_repeated_shard_references_are_streamed_once(inputs, tmp_path, monkeypatch):
    from parascale.workloads import causal_lm_assets

    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "part.bin").write_bytes(b"weights")
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"a": "part.bin", "b": "part.bin"}}),
        encoding="utf-8",
    )
    original = causal_lm_assets.file_sha256
    calls = []

    def observe(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(causal_lm_assets, "file_sha256", observe)
    build(*inputs)
    assert calls.count(tmp_path / "part.bin") == 1


def test_implementation_change_invalidates_context(inputs, monkeypatch):
    from parascale.runtime import stage_identity

    before = build(*inputs)
    original = stage_identity._digest

    def changed(path):
        return "0" * 64 if path.name == "llama.py" else original(path)

    monkeypatch.setattr(stage_identity, "_digest", changed)
    assert build(*inputs) != before


@pytest.mark.parametrize(
    "relative",
    [
        "workloads/causal_lm_model.py",
        "workloads/causal_lm_assets.py",
        "data/text/__init__.py",
        "data/text/artifact.py",
        "data/text/batches.py",
        "optimizers/factory.py",
        "optimizers/spec.py",
        "optimizers/schedule.py",
        "runtime/training/data_state.py",
        "runtime/training/random_state.py",
    ],
)
def test_semantic_implementation_changes_invalidate_context(
    inputs, monkeypatch, relative
):
    from pathlib import Path

    from parascale.runtime import stage_identity

    before = build(*inputs)
    original = stage_identity._digest
    target = Path(stage_identity.__file__).resolve().parents[1] / relative
    monkeypatch.setattr(
        stage_identity,
        "_digest",
        lambda path: "0" * 64 if path == target else original(path),
    )
    assert build(*inputs) != before


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", 0.002),
        ("betas", (0.8, 0.99)),
        ("eps", 1e-6),
        ("weight_decay", 0.02),
    ],
)
def test_requested_optimizer_hyperparameters_invalidate_context(inputs, field, value):
    from dataclasses import asdict, replace

    before = build(*inputs)
    assert before["optimizer"] == json.loads(
        json.dumps(asdict(inputs[2].optimizer_spec))
    )
    assert before["optimizer"]["betas"] == [0.9, 0.999]
    inputs[2].optimizer_spec = replace(inputs[2].optimizer_spec, **{field: value})
    assert build(*inputs) != before


def test_real_stage_context_tracks_effective_config_and_releases_source(
    tmp_path, monkeypatch, capsys
):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.3":
        pytest.skip("stage dependency pin: transformers==4.51.3")
    from parascale.optimizers.spec import OptimizerSpec
    from parascale.parallel import DistributedParallelConfig
    from parascale.workloads.causal_lm import build_causal_lm_stage_components
    from tests.test_causal_lm_workload import configuration, prepare
    from tests.test_causal_lm_workload import model_path as model_fixture

    model_path = model_fixture.__wrapped__(tmp_path)
    config = transformers.LlamaConfig.from_pretrained(model_path)
    config.attention_dropout = 0
    transformers.LlamaForCausalLM(config).save_pretrained(model_path)
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    data_path = prepare(tmp_path, model_path, capsys)
    settings = configuration(model_path, data_path, tmp_path / "checkpoints")
    sources = []
    original = transformers.AutoModelForCausalLM.from_pretrained

    def load(*args, **kwargs):
        model = original(*args, **kwargs)
        sources.append(weakref.ref(model))
        return model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    stage, cursor, _ = build_causal_lm_stage_components(
        settings,
        DistributedParallelConfig(),
        rank=0,
        world_size=1,
    )
    assert sources[0]() is None
    plan = Record(
        workload=Record(model_path=str(model_path)),
        config=Record(precision="fp32"),
        optimizer_spec=OptimizerSpec.from_config(settings),
    )
    context = build(stage, cursor, plan)
    assert raw["use_cache"] is True
    assert context["model"]["config"]["use_cache"] is False
    assert context["model"]["config"]["_attn_implementation"] == "sdpa"
    reference = weakref.ref(stage)
    del stage
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "filename,dp,pp",
    [
        ("single_gpu.yaml", 1, 1),
        ("dp2_pp2.yaml", 2, 2),
    ],
)
def test_native_parallel_example_configuration(filename, dp, pp):
    from pathlib import Path

    from parascale.commands.common import load_config_file
    from parascale.runtime.stage_plan import build_stage_training_plan

    root = Path(__file__).resolve().parents[1]
    path = root / "examples/gpu/example_012_native_parallel_training" / filename
    settings = load_config_file(path)
    assert "optimizer" in settings and "optimizer" not in settings["training"]
    plan = build_stage_training_plan(settings)
    assert plan.optimizer_spec.lr == settings["optimizer"]["lr"] == 0.00001
    assert plan.optimizer_spec.weight_decay == settings["optimizer"]["weight_decay"]
    assert (plan.topology.dp_size, plan.topology.tp_size, plan.topology.pp_size) == (
        dp,
        1,
        pp,
    )
    assert plan.topology.world_size == dp * pp
    assert plan.topology.global_batch_size(plan.workload.batch_size) == 16 * dp
    assert plan.workload.sequence_length == 256
    assert plan.config.precision == "bf16"
    assert plan.checkpoint_interval == 10
    assert plan.keep_last >= 2
    assert plan.metrics_history_limit == 20
    assert plan.max_steps == 20
    assert plan.workload.model_path == "/models/tinyllama"
    assert plan.workload.data_path == "/data/wikitext-tokens"


def test_matched_parallel_examples_keep_the_same_training_budget():
    from copy import deepcopy
    from pathlib import Path

    from parascale.commands.common import load_config_file
    from parascale.runtime.stage_plan import build_stage_training_plan

    root = Path(__file__).resolve().parents[1]
    folder = root / "examples/gpu/example_012_native_parallel_training"
    canonical = []
    for name, pp, schedule in (
        ("matched_single.yaml", 1, "gpipe"),
        ("matched_pp2.yaml", 2, "1f1b"),
    ):
        settings = load_config_file(folder / name)
        plan = build_stage_training_plan(settings)
        assert (plan.topology.dp_size, plan.topology.tp_size, plan.topology.pp_size) == (
            1,
            1,
            pp,
        )
        assert plan.topology.global_batch_size(plan.workload.batch_size) == 64
        assert plan.topology.microbatches == 16
        assert plan.topology.pipeline_schedule == schedule
        assert plan.workload.batch_size == 4
        assert plan.workload.sequence_length == 256
        assert plan.config.precision == "bf16"
        assert plan.max_steps == plan.checkpoint_interval == 16
        assert plan.metrics_history_limit == 16
        assert settings["data"]["shuffle"] is False
        assert settings["data"]["repeat"] is False
        assert settings["training"]["seed"] == settings["data"]["seed"] == 42
        assert settings["optimizer"] == {
            "type": "adamw",
            "lr": 0.00001,
            "betas": [0.9, 0.999],
            "eps": 0.00000001,
            "weight_decay": 0.01,
        }
        assert settings["model"] == {
            "pretrained_model_name_or_path": "/models/tinyllama",
            "attention_backend": "sdpa",
        }
        assert settings["data"]["path"] == "/data/wikitext-tokens"
        shared = deepcopy(settings)
        for key in (
            "pipeline_parallel_size",
            "pipeline_schedule",
            "checkpoint_save_path",
        ):
            del shared["parascale"][key]
        del shared["runtime"]["run_dir"]
        canonical.append(shared)
    assert canonical[0] == canonical[1]


def test_shared_assets_config_only_needs_no_float_weights(tmp_path):
    import hashlib

    from parascale.workloads.causal_lm_assets import model_asset_hashes

    content = b'{"model_type": "llama"}'
    (tmp_path / "config.json").write_bytes(content)
    # Even an invalid weight index is irrelevant to artifact-only reload.
    (tmp_path / "model.safetensors.index.json").write_text("invalid", encoding="utf-8")
    assert model_asset_hashes(tmp_path, include_weights=False) == {
        "config.json": hashlib.sha256(content).hexdigest()
    }


def test_shared_assets_default_requires_weights(tmp_path):
    from parascale.workloads.causal_lm_assets import model_asset_hashes

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="No supported local model weights"):
        model_asset_hashes(tmp_path)


def test_shared_assets_stage_alias_and_implementation_dependencies():
    from parascale.runtime import stage_identity
    from parascale.workloads.causal_lm_assets import model_asset_hashes

    assert stage_identity._model_assets is model_asset_hashes
    hashes = stage_identity._implementation_hashes()
    assert "workloads/causal_lm_model.py" in hashes
    assert "workloads/causal_lm_assets.py" in hashes


@pytest.mark.parametrize("as_string", [False, True])
def test_shared_file_sha256_streams_one_mib_chunks(tmp_path, monkeypatch, as_string):
    import hashlib
    from pathlib import Path

    from parascale.workloads.causal_lm_assets import file_sha256

    content = b"a" * (2 * 1024 * 1024 + 17)
    path = tmp_path / "opaque.bin"
    path.write_bytes(content)
    original = Path.open
    reads = []

    class Reader:
        def __enter__(self):
            self.stream = original(path, "rb")
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: Reader())
    assert (
        file_sha256(str(path) if as_string else path)
        == hashlib.sha256(content).hexdigest()
    )
    assert reads == [1024 * 1024] * 4


def test_shared_assets_import_without_torch_or_transformers(tmp_path):
    import subprocess
    import sys

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"opaque unit weights")
    code = """
import builtins, sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "transformers"}:
        raise AssertionError("Asset helper must not import tensor frameworks")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from parascale.workloads.causal_lm_assets import model_asset_hashes, file_sha256
root = Path(sys.argv[1])
assert set(model_asset_hashes(root)) == {"config.json", "model.safetensors"}
assert set(model_asset_hashes(root, include_weights=False)) == {"config.json"}
assert len(file_sha256(root / "config.json")) == 64
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_shared_assets_preserve_trusted_external_symlinks(tmp_path):
    import hashlib

    from parascale.workloads.causal_lm_assets import model_asset_hashes

    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    blob = tmp_path / "trusted-blob"
    blob.write_bytes(b"opaque unit weights")
    try:
        (root / "model.safetensors").symlink_to(blob)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    assert (
        model_asset_hashes(root)["model.safetensors"]
        == hashlib.sha256(blob.read_bytes()).hexdigest()
    )


def test_shared_assets_reject_duplicate_index_keys(tmp_path):
    from parascale.workloads.causal_lm_assets import model_asset_hashes

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "part.bin").write_bytes(b"opaque unit weights")
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map":{"a":"part.bin","a":"part.bin"}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid model weight index"):
        model_asset_hashes(tmp_path)
