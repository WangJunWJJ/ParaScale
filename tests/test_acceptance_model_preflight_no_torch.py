# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Local metadata must block incompatible layouts before expensive work."""

import builtins
import json

import pytest

from examples.acceptance import run
from examples.acceptance.cases import CASES, validate_environment


def config():
    return {"model_type": "llama", "hidden_size": 576, "intermediate_size": 1536,
            "num_attention_heads": 9, "num_key_value_heads": 3,
            "num_hidden_layers": 3, "vocab_size": 49153,
            "tie_word_embeddings": False}


def environment(tmp_path, metadata):
    model, corpus = tmp_path / "model", tmp_path / "corpus"
    model.mkdir()
    corpus.mkdir()
    if metadata is not None:
        (model / "config.json").write_text(metadata, encoding="utf-8")
    return validate_environment({"schema_version": 1, "gpu_ids": [0, 1, 2, 3],
                                 "assets": {"llama_model": str(model), "token_corpus": str(corpus)}})


@pytest.mark.parametrize("field,value,case,expected", [
    ("num_attention_heads", 9, "tp2", "num_attention_heads=9"),
    ("num_key_value_heads", 3, "tp2", "num_key_value_heads=3"),
    ("hidden_size", 575, "tp2", "hidden_size=575"),
    ("intermediate_size", 1535, "tp2", "intermediate_size=1535"),
    ("tie_word_embeddings", True, "pp2", "tie_word_embeddings"),
    ("num_hidden_layers", 1, "pp2", "num_hidden_layers"),
    ("attention_dropout", 0.1, "single", "attention_dropout"),
    ("pretraining_tp", 2, "single", "pretraining_tp"),
    ("model_type", "mistral", "single", "model_type"),
    ("hidden_size", True, "single", "hidden_size"),
    ("num_attention_heads", "9", "single", "num_attention_heads"),
    ("num_hidden_layers", 0, "single", "num_hidden_layers"),
    ("vocab_size", -1, "single", "vocab_size"),
    ("tie_word_embeddings", "false", "single", "tie_word_embeddings"),
    ("attention_dropout", False, "single", "attention_dropout"),
    ("pretraining_tp", True, "single", "pretraining_tp"),
])
def test_contract_blockers(tmp_path, field, value, case, expected):
    env = environment(tmp_path, json.dumps({**config(), field: value}))
    _, reasons = run.prepare_case(CASES[f"parallel_training.{case}"], env, tmp_path / "out", "train", "infer")
    assert expected in " ".join(reasons)
    assert "config.json" in " ".join(reasons)


@pytest.mark.parametrize("metadata", [None, "{", "[]", "null", "{}", '\ufeff{}',
    '{"model_type":"llama","model_type":"llama"}', '{"value":NaN}'])
def test_bad_metadata_blocks_before_hash_storage_or_launch(tmp_path, monkeypatch, metadata):
    env = environment(tmp_path, metadata)

    def forbidden(*args, **kwargs):
        pytest.fail("blocked metadata reached hashing, storage estimation or launch")

    monkeypatch.setattr(run, "asset_identity", forbidden)
    monkeypatch.setattr(run, "storage_plan", forbidden)
    report = run.execute_suite(env, tmp_path / "out", ["parallel_training.single"],
                               mode="functional", process=forbidden, tools=("train", "infer"))
    assert report["status"] == "BLOCKED"
    assert report["cases"][0]["functional_status"] == "BLOCKED"
    assert "config.json" in str(report["cases"][0]["reasons"])


@pytest.mark.parametrize("field", ["model_type", "hidden_size", "intermediate_size",
                                  "num_attention_heads", "num_hidden_layers", "vocab_size"])
def test_missing_required_metadata(tmp_path, field):
    metadata = config()
    del metadata[field]
    env = environment(tmp_path, json.dumps(metadata))
    _, reasons = run.prepare_case(CASES["parallel_training.single"], env, tmp_path / "out", "train", "infer")
    assert field in " ".join(reasons)


@pytest.mark.parametrize("case,updates", [
    ("single", {"tie_word_embeddings": True}),
    ("dp2", {"tie_word_embeddings": True}),
    ("pp2", {}),
    ("tp2_pp2", {"num_attention_heads": 8, "num_key_value_heads": 2}),
    ("tp2", {"num_attention_heads": 8, "num_key_value_heads": None}),
    ("tp2", {"num_attention_heads": 8, "num_key_value_heads": "omit"}),
    ("single", {"torch_dtype": "bfloat16", "_attn_implementation": "flash_attention_2"}),
])
def test_supported_metadata_without_invented_restrictions(tmp_path, case, updates):
    metadata = {**config(), **updates}
    if metadata.get("num_key_value_heads") == "omit":
        del metadata["num_key_value_heads"]
    env = environment(tmp_path, json.dumps(metadata))
    _, reasons = run.prepare_case(CASES[f"parallel_training.{case}"], env, tmp_path / "out", "train", "infer")
    assert reasons == []


def test_plan_and_execute_share_blockers_without_heavy_imports(tmp_path, monkeypatch, capsys):
    env = environment(tmp_path, json.dumps(config()))
    env_path = tmp_path / "environment.json"
    env_path.write_text(json.dumps(env), encoding="utf-8")
    monkeypatch.setattr(run.shutil, "which", lambda name: name)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "out"
    assert run.main(["--plan", "--config", str(env_path), "--output", str(output),
                     "--case", "parallel_training.tp2"]) == 0
    plan = json.loads(capsys.readouterr().out)["parallel_training.tp2"]
    assert "num_attention_heads=9" in str(plan["blockers"])
    assert "num_key_value_heads=3" in str(plan["blockers"])
    assert not output.exists()
    def forbidden(*args, **kwargs):
        pytest.fail("blocked layout launched a process")

    report = run.execute_suite(env, output, ["parallel_training.tp2"], process=forbidden,
                               tools=("parascale-train", "parascale-infer"))
    assert report["cases"][0]["reasons"] == plan["blockers"]
    assert report["status"] == "BLOCKED"


def test_nonparallel_cases_do_not_apply_llama_stage_contract(tmp_path):
    env = environment(tmp_path, "malformed")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{}\n', encoding="utf-8")
    env["assets"]["llama_prompts"] = str(prompts)
    for key in ("installation.entries", "generative_inference.transformer"):
        _, reasons = run.prepare_case(CASES[key], env, tmp_path / "out", "train", "infer")
        assert reasons == []
