# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Safety, coverage and real CUDA dispatch for fused INT4 inference."""

import copy

import pytest
import torch
from torch import nn

from parascale.quantization import InferenceQuantizationConfig, quantize_inference_model
from parascale.quantization.inference import _select_linear_layers


def test_cpu_rejection_preserves_model():
    model = nn.Sequential(nn.Linear(1024, 16)).eval()
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="CUDA"):
        quantize_inference_model(model, inplace=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_training_model_rejected():
    with pytest.raises(ValueError, match="eval"):
        quantize_inference_model(nn.Linear(1024, 16))


def test_selection_excludes_heads_ties_small_and_unaligned_layers():
    model = nn.ModuleDict(
        {
            "projection": nn.Linear(1024, 32),
            "small": nn.Linear(128, 16),
            "unaligned": nn.Linear(1024, 15),
            "embedding": nn.Embedding(32, 1024),
            "tied": nn.Linear(1024, 32, bias=False),
            "lm_head": nn.Linear(1024, 32),
            "decoder": nn.Sequential(nn.Linear(1024, 16)),
        }
    )
    model["tied"].weight = model["embedding"].weight
    selected, skipped = _select_linear_layers(
        model, InferenceQuantizationConfig(exclude_modules=("lm_head", "decoder"))
    )
    assert set(selected) == {"projection"}
    assert skipped["tied"] == "shared_parameter"
    assert skipped["lm_head"] == skipped["decoder.0"] == "excluded"
    assert skipped["small"] == skipped["unaligned"] == "unaligned_shape"


def test_shared_module_alias_cannot_bypass_exclusion():
    layer = nn.Linear(1024, 32)
    model = nn.ModuleDict({"projection": layer, "lm_head": layer})
    selected, skipped = _select_linear_layers(model, InferenceQuantizationConfig())
    assert not selected
    assert skipped["projection"] == "shared_parameter"


def test_no_supported_layers_is_explicit():
    with pytest.raises(ValueError, match="No eligible"):
        quantize_inference_model(nn.Sequential(nn.Linear(12, 3)).eval())


def test_hooked_layers_keep_their_activation_preprocessing():
    model = nn.Sequential(nn.Linear(1024, 32), nn.Linear(1024, 32)).eval()
    model[0].register_forward_pre_hook(lambda module, inputs: (inputs[0] * 2,))
    selected, skipped = _select_linear_layers(model, InferenceQuantizationConfig())
    assert set(selected) == {"1"}
    assert skipped["0"] == "execution_hooks"


def test_int8_excludes_functional_weight_parents():
    model = nn.ModuleDict(
        {"linear": nn.Linear(128, 32), "attention": nn.TransformerEncoderLayer(128, 4, 256)}
    ).eval()
    selected, skipped = _select_linear_layers(
        model, InferenceQuantizationConfig(backend="gemlite_int8")
    )
    assert set(selected) == {"linear"}
    assert skipped["attention.linear1"] == "functional_weight_access"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_int8_fused_numerics_state_and_dtype_guard():
    pytest.importorskip("gemlite")
    source = nn.Sequential(nn.Linear(128, 128)).cuda().bfloat16().eval()
    original = source[0].weight.detach().clone()
    config = InferenceQuantizationConfig(backend="gemlite_int8")
    with torch.inference_mode():
        model, report = quantize_inference_model(source, config)
        x = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
        actual = model(x)
        assert (actual.float() - source(x).float()).norm() / source(
            x
        ).float().norm() < 0.02
        torch.testing.assert_close(source[0].weight, original)
        assert report["packed_weight_bytes"] < report["source_weight_bytes"] * 0.55
        restored, _ = quantize_inference_model(source, config)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        torch.testing.assert_close(restored(x), actual)
        assert model(
            torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
        ).shape == (0, 128)
        with pytest.raises(RuntimeError, match="functional"):
            _ = model[0].weight
        model.bfloat16()
        with pytest.raises(ValueError, match="scales"):
            model(x)


def test_benchmark_greedy_policy_overrides_model_sampling_defaults(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from tests.benchmarks.scripts.benchmark_quantized_inference import (
        fixed_length_generation,
    )

    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_layer=1,
            n_head=2,
            n_embd=16,
            vocab_size=32,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.6
    model.generation_config.transformers_version = "4.51.3"
    original_prepare = model._prepare_generation_config
    observed = []

    def capture(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        observed.append(result[0])
        return result

    monkeypatch.setattr(model, "_prepare_generation_config", capture)
    prompt = torch.tensor([[1, 3, 4]])
    with torch.inference_mode():
        first = fixed_length_generation(model, prompt, torch.ones_like(prompt), 4, 0)
        second = fixed_length_generation(model, prompt, torch.ones_like(prompt), 4, 0)
    assert first.shape == (1, 7)
    torch.testing.assert_close(first, second)
    assert all(
        not config.do_sample and config.eos_token_id is None for config in observed
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
def test_cuda_fused_dispatch_numerics_and_original_preserved():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires CUDA SM80+")
    pytest.importorskip("torchao")
    from parascale.quantization.quantized_layers import QuantizedLinear

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(1024, 1024)).cuda().bfloat16().eval()
    x = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16)
    original = model[0].weight.detach().clone()
    with torch.inference_mode():
        expected = model(x)
        quantized, report = quantize_inference_model(model)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as trace:
            actual = quantized(x)
        assert any("_weight_int4pack_mm" in event.key for event in trace.key_averages())
        assert torch.isfinite(actual).all()
        assert (
            actual.float() - expected.float()
        ).norm() / expected.float().norm() < 0.15
        torch.testing.assert_close(model[0].weight, original)
        assert report["converted_layers"] == ["0"]
        assert report["packed_weight_bytes"] < report["source_weight_bytes"] / 2
        assert report["performance_validated"] is False
        assert not any(
            isinstance(layer, QuantizedLinear) for layer in quantized.modules()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_generation_preserves_dynamic_cache_outputs():
    transformers = pytest.importorskip("transformers")
    from tests.benchmarks.scripts.benchmark_quantized_inference import (
        compile_generation_forward,
        fixed_length_generation,
    )

    torch.manual_seed(42)
    model = (
        transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                n_layer=1,
                n_head=2,
                n_embd=32,
                vocab_size=32,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        .eval()
        .cuda()
        .bfloat16()
    )
    prompt = torch.tensor([[1, 3, 4]], device="cuda")
    with torch.inference_mode():
        expected = fixed_length_generation(model, prompt, torch.ones_like(prompt), 4, 0)
        compile_generation_forward(model)
        for _ in range(2):
            actual = fixed_length_generation(
                model, prompt, torch.ones_like(prompt), 4, 0
            )
            torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")
def test_packed_linear_matches_torchao_without_tensor_subclass_dispatch():
    pytest.importorskip("torchao")
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    from parascale.quantization.inference import PackedInt4Linear

    torch.manual_seed(42)
    source = nn.Linear(1024, 32, bias=True).cuda().bfloat16().eval()
    quantize_(
        source,
        Int4WeightOnlyConfig(
            group_size=32,
            version=2,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq",
            set_inductor_config=False,
        ),
    )
    packed = PackedInt4Linear(source, 32)
    with torch.inference_mode():
        for shape in [(1024,), (2, 1024), (2, 3, 1024), (0, 1024)]:
            x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            if x.numel():
                torch.testing.assert_close(packed(x), source(x), rtol=0, atol=0)
            else:
                assert packed(x).shape == (0, 32)
        x = torch.randn(2, 1024, 3, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        torch.testing.assert_close(packed(x), source(x), rtol=0, atol=0)
        torch.testing.assert_close(
            nn.functional.linear(x, packed.weight, packed.bias),
            source(x),
            rtol=0,
            atol=0,
        )
        restored = PackedInt4Linear(source, 32)
        restored.load_state_dict(copy.deepcopy(packed.state_dict()))
        torch.testing.assert_close(restored(x), packed(x), rtol=0, atol=0)
        with pytest.raises(ValueError, match="bfloat16"):
            packed(x.float())
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as trace:
            packed(x.contiguous())
        keys = {event.key for event in trace.key_averages()}
        assert "aten::_weight_int4pack_mm" in keys
        assert "aten::pad" not in keys
