# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Offline Llama correctness tests; no model download is performed."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def validator():
    path = Path(__file__).parent / "benchmarks/scripts/validate_llama_quantization.py"
    spec = importlib.util.spec_from_file_location("llama_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("bits", [4, 8])
def test_tiny_llama_roundtrip(validator, tmp_path, mode, bits):
    args = validator.parser().parse_args(
        [
            "--tiny",
            "--mode",
            mode,
            "--bits",
            str(bits),
            "--device",
            "cpu",
            "--seqlen",
            "8",
            "--batches",
            "2",
            "--qat-steps",
            "2",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--output",
            str(tmp_path),
        ]
    )
    report = validator.run(args)
    result = report[mode]
    assert result["loss"] > 0
    assert result["perplexity"] > 0
    assert result["roundtrip_max_abs_error"] <= 1e-5
    assert result["checkpoint_bytes"] > 0
    assert result["logits_mae"] >= 0
    assert "procedure_memory" in result
    assert "inference_memory" in result["timing"]
    assert result["timing"]["inference_memory"]["cuda_peak_allocated_bytes"] is None
    assert len(result["checkpoint_sha256"]) == 64
    assert len(report["baseline"]["checkpoint_sha256"]) == 64
    assert report["model_source"]["kind"] == "random_llama"
    if mode == "qat":
        assert result["training"]["changed_parameters"] > 0
        assert result["training"]["nonzero_gradient_steps"] == 2


def test_identical_corpora_rejected(validator, tmp_path):
    calibration = tmp_path / "calibration.txt"
    evaluation = tmp_path / "evaluation.txt"
    calibration.write_text("same text", encoding="utf-8")
    evaluation.write_text("same text", encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        validator.corpus_evidence([calibration], [evaluation])


def test_pretrained_requires_real_text(validator, tmp_path):
    args = validator.parser().parse_args(
        ["--model", "unused", "--output", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="calibration.*evaluation"):
        validator.run(args)


def test_text_batches_are_bounded(validator, tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("abcdefgh" * 100, encoding="utf-8")

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) % 32 for c in text]}

    batches = list(validator.text_batches([path], Tokenizer(), 8, 2, 3))
    assert len(batches) == 3
    assert all(batch["input_ids"].shape == (2, 8) for batch in batches)


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_preparation_preserves_float_model_and_head(mode):
    from transformers import LlamaConfig, LlamaForCausalLM

    from parascale.quantization import (
        PostTrainingQuantization,
        QuantizationAwareTraining,
        QuantizationConfig,
    )

    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    original = {name: value.clone() for name, value in model.state_dict().items()}
    config = QuantizationConfig(
        enabled=True,
        mode=mode,
        bits=4,
        fuse_modules=False,
        per_channel=True,
        quantize_activations=False,
        exclude_modules=["lm_head"],
    )
    engine = (PostTrainingQuantization if mode == "ptq" else QuantizationAwareTraining)(
        model, config
    )
    prepared = engine.prepare()
    assert prepared is not model
    assert type(prepared.lm_head) is torch.nn.Linear
    assert any(hasattr(layer, "weight_fake_quant") for layer in prepared.modules())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    with torch.no_grad():
        next(prepared.parameters()).add_(1)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_evaluation_weights_by_token_count(validator):
    import math
    from types import SimpleNamespace

    class UniformModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 4))

    def batches():
        return iter(
            [
                {"input_ids": torch.tensor([[0, 1]])},
                {"input_ids": torch.tensor([[0, 1, 2, 3]])},
            ]
        )

    result = validator.evaluate(UniformModel(), batches, torch.device("cpu"))
    assert result["evaluated_tokens"] == 4
    assert result["loss"] == pytest.approx(math.log(4))
    assert result["perplexity"] == pytest.approx(4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_measure_resets_cuda_peak_after_warmup(validator):
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(
                logits=torch.zeros(*input_ids.shape, 4, device=input_ids.device)
            )

    device = torch.device("cuda:0")
    allocation = torch.empty(8 * 1024 * 1024, device=device)
    previous_peak = torch.cuda.max_memory_allocated(device)
    del allocation

    def batches():
        return iter([{"input_ids": torch.tensor([[0, 1]])}])

    result = validator.measure(Model(), batches, device, 1, 2)
    memory = result["inference_memory"]
    assert memory["cuda_peak_allocated_bytes"] < previous_peak
    assert memory["cuda_peak_allocated_bytes"] >= memory["cuda_start_allocated_bytes"]
    assert memory["cuda_incremental_peak_bytes"] >= 0
