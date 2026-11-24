# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Canonical local causal-LM templates and strict reference artifact reloads."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tests.test_causal_lm_workload import (  # noqa: E402, F401
    configuration,
    prepare,
)
from tests.test_causal_lm_workload import (  # noqa: E402
    model_path as model_path,
)


def test_stable_wrapper_forward_and_config(model_path):
    from parascale.workloads.causal_lm_model import CausalLmModel, load_causal_lm_model

    model = load_causal_lm_model(str(model_path))
    assert type(model) is CausalLmModel
    assert CausalLmModel.__qualname__ == "CausalLmModel"
    assert CausalLmModel.__module__ == "parascale.workloads.causal_lm_model"
    assert model.config is model.model.config
    assert model.config.use_cache is False
    assert model.config._attn_implementation == "sdpa"
    assert all(
        p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()
    )
    ids = torch.tensor([[3, 4, 5, 6]])
    mask = torch.ones_like(ids)
    model.train()
    torch.manual_seed(81)
    expected = model.model(
        input_ids=ids, labels=ids, attention_mask=mask, use_cache=False
    )
    torch.manual_seed(81)
    actual = model(
        ids, labels=ids, attention_mask=mask, tokens=3, block_ids=[0], use_cache=True
    )
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    assert actual.past_key_values is None
    actual.loss.backward()
    assert model.model.get_input_embeddings().weight.grad is not None


def test_tokenizer_identity_is_bound_and_read_only(model_path):
    from parascale.data.text import tokenizer_fingerprint
    from parascale.quantization.core import signature
    from parascale.workloads.causal_lm_model import load_causal_lm_model

    model = load_causal_lm_model(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=True, trust_remote_code=False
    )
    expected = tokenizer_fingerprint(tokenizer)
    assert model.tokenizer_identity == expected
    assert signature(model)["modules"][0]["semantics"]["tokenizer_identity"] == expected
    assert "tokenizer_identity" not in model.config.to_dict()
    with pytest.raises(AttributeError):
        model.tokenizer_identity = "replacement"


def _config_only_copy(model_path, tmp_path):
    path = tmp_path / "config-only"
    shutil.copytree(model_path, path)
    for weights in path.glob("*.safetensors"):
        weights.unlink()
    for weights in path.glob("*.bin"):
        weights.unlink()
    return path


def test_config_only_never_loads_pretrained_weights(model_path, tmp_path, monkeypatch):
    from parascale.quantization.core import signature
    from parascale.workloads.causal_lm_model import load_causal_lm_model

    expected = load_causal_lm_model(model_path, attention_backend="eager")
    path = _config_only_copy(model_path, tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("config-only construction attempted pretrained weight loading")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.PreTrainedModel, "from_pretrained", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    model = load_causal_lm_model(path, attention_backend="eager", from_config=True)
    assert signature(model) == signature(expected)
    assert model.config.use_cache is False
    assert all(
        p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()
    )


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_strict_converted_round_trip(model_path, tmp_path, mode):
    from parascale.quantization.base import QuantizationConfig
    from parascale.quantization.core import signature
    from parascale.quantization.ptq import (
        PostTrainingQuantization,
        load_quantized_model,
    )
    from parascale.quantization.qat import QuantizationAwareTraining
    from parascale.workloads.causal_lm_model import load_causal_lm_model

    source = load_causal_lm_model(model_path)
    ids = torch.tensor([[3, 4, 5, 6]])
    batch = {"input_ids": ids, "labels": ids, "attention_mask": torch.ones_like(ids)}
    config = QuantizationConfig(mode=mode, bits=8, calib_batches=1)
    quantizer = PostTrainingQuantization if mode == "ptq" else QuantizationAwareTraining
    owner = quantizer(source, config)
    prepared = owner.prepare()
    if mode == "ptq":
        owner.calibrate([batch])
    else:
        optimizer = torch.optim.SGD(prepared.parameters(), lr=0.01)
        prepared(**batch).loss.backward()
        optimizer.step()
    converted = owner.convert()
    artifact = tmp_path / f"{mode}.pt"
    owner.export(artifact)
    path = _config_only_copy(model_path, tmp_path)
    template = load_causal_lm_model(path, from_config=True)
    assert signature(template) == signature(source)
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    assert payload["manifest"] == signature(template)
    restored, loaded_config, _ = load_quantized_model(artifact, template)
    assert loaded_config.to_dict() == config.to_dict()
    assert signature(restored) == signature(converted)
    torch.testing.assert_close(
        restored.state_dict(), converted.state_dict(), rtol=0, atol=0
    )
    with torch.no_grad():
        torch.testing.assert_close(
            restored(**batch).logits, converted(**batch).logits, rtol=0, atol=0
        )
        generation = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
        )
        assert torch.equal(
            restored.model.generate(**generation),
            converted.model.generate(**generation),
        )
    with pytest.raises(ValueError, match="template manifest mismatch"):
        load_quantized_model(artifact, template.model)
    tokenizer_path = path / "tokenizer.json"
    original_tokenizer = tokenizer_path.read_text(encoding="utf-8")
    permuted = json.loads(original_tokenizer)
    vocabulary = permuted["model"]["vocab"]
    vocabulary["a"], vocabulary["b"] = vocabulary["b"], vocabulary["a"]
    tokenizer_path.write_text(json.dumps(permuted), encoding="utf-8")
    wrong_tokenizer = load_causal_lm_model(path, from_config=True)
    assert wrong_tokenizer.config.to_dict() == template.config.to_dict()
    with pytest.raises(ValueError, match="template manifest mismatch"):
        load_quantized_model(artifact, wrong_tokenizer)
    tokenizer_path.write_text(original_tokenizer, encoding="utf-8")
    config_path = path / "config.json"
    wrong = json.loads(config_path.read_text(encoding="utf-8"))
    wrong["rms_norm_eps"] *= 2
    config_path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="template manifest mismatch"):
        load_quantized_model(artifact, load_causal_lm_model(path, from_config=True))


def test_training_builder_uses_stable_wrapper(
    model_path, tmp_path, capsys, monkeypatch
):
    from parascale.workloads.causal_lm import build_causal_lm_components
    from parascale.workloads.causal_lm_model import CausalLmModel

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    data = prepare(tmp_path, model_path, capsys)
    config = configuration(model_path, data, tmp_path / "checkpoint")
    config["model"]["activation_checkpointing"] = True
    model, _, cursor, loss = build_causal_lm_components(config)
    assert type(model) is CausalLmModel
    assert model.tokenizer_identity == cursor.corpus.manifest["tokenizer"]
    assert model.training and model.model.is_gradient_checkpointing
    assert cursor.seed == 9
    batch = next(iter(cursor))
    output = model(**batch)
    assert loss(output, batch) is output.loss
    assert loss.normalization_weight(batch) == batch["tokens"]


@pytest.mark.parametrize("from_config", [False, True])
def test_loader_rejects_nonlocal_directory(tmp_path, from_config):
    from parascale.workloads.causal_lm_model import load_causal_lm_model

    with pytest.raises(ValueError, match="local model directory"):
        load_causal_lm_model(tmp_path / "missing", from_config=from_config)


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_benchmark_loads_product_artifact_without_float_weights(
    model_path, tmp_path, mode
):
    from parascale.quantization import (
        PostTrainingQuantization,
        QuantizationAwareTraining,
        QuantizationConfig,
    )
    from parascale.workloads.causal_lm_model import load_causal_lm_model
    from tests.benchmarks.scripts.quantized_artifact_fixture import (
        load_reference_artifact,
        materialize_reference_weights,
    )

    config = transformers.AutoConfig.from_pretrained(model_path)
    config.hidden_size, config.intermediate_size = 32, 64
    config.num_attention_heads, config.num_key_value_heads = 4, 2
    transformers.LlamaForCausalLM(config).save_pretrained(model_path)
    source = load_causal_lm_model(model_path)
    policy = QuantizationConfig(mode=mode, bits=8, calib_batches=1)
    owner = (PostTrainingQuantization if mode == "ptq" else QuantizationAwareTraining)(
        source, policy
    )
    prepared = owner.prepare()
    ids = torch.tensor([[3, 4, 5, 6]])
    if mode == "ptq":
        owner.calibrate([{"input_ids": ids}])
    else:
        optimizer = torch.optim.SGD(prepared.parameters(), lr=0.01)
        prepared(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
    converted = owner.convert().eval()
    artifact = tmp_path / f"{mode}.pt"
    owner.export(artifact)
    local_assets = _config_only_copy(model_path, tmp_path)
    loaded, metadata = load_reference_artifact(artifact, local_assets)
    assert type(loaded) is type(converted.model)
    assert metadata == policy.to_dict()
    loaded.eval()
    with torch.inference_mode():
        expected = converted.model(input_ids=ids, use_cache=False).logits
        torch.testing.assert_close(
            loaded(input_ids=ids, use_cache=False).logits, expected, rtol=0, atol=0
        )
        dense = materialize_reference_weights(loaded)
        torch.testing.assert_close(
            dense(input_ids=ids, use_cache=False).logits, expected, rtol=0, atol=0
        )
    tokenizer_path = local_assets / "tokenizer.json"
    changed = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = changed["model"]["vocab"]
    vocab["a"], vocab["b"] = vocab["b"], vocab["a"]
    tokenizer_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="template manifest mismatch"):
        load_reference_artifact(artifact, local_assets)


def test_cli_and_workload_import_without_torch():
    code = """
import sys
class BlockHeavyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('eager dependency import: ' + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
import parascale.cli
import parascale.workloads.causal_lm
assert 'parascale.workloads.causal_lm_model' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
