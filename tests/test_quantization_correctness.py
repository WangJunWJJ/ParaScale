# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Numerical contracts for reference quantization."""

import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from parascale.quantization import (
    FakeQuantize,
    PostTrainingQuantization,
    QuantizationAwareTraining,
    QuantizationConfig,
    QuantizedLinear,
    convert_qat_model,
    load_quantized_model,
)


def test_native_ste_and_serializable_state():
    config = QuantizationConfig(per_channel=False, scheme="asymmetric")
    quant = FakeQuantize(config)
    x = torch.tensor([0.2, 0.7, 1.3], requires_grad=True)
    quant(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert quant.zero_point.item() != 0
    quant.enable_observer(False)
    restored = FakeQuantize(config)
    restored.load_state_dict(quant.state_dict())
    assert not restored.observer_enabled
    torch.testing.assert_close(restored(x), quant(x))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("activations", [False, True])
def test_qat_updates_parity_and_export(bits, activations, tmp_path):
    torch.manual_seed(12)
    original = nn.Sequential(nn.Linear(5, 3))
    config = QuantizationConfig(bits=bits, quantize_activations=activations)
    qat = QuantizationAwareTraining(original, config)
    model = qat.prepare()
    before = model[0].weight.detach().clone()
    inputs = torch.randn(4, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(inputs).square().mean().backward()
    assert model[0].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model[0].weight)
    torch.testing.assert_close(original[0].weight, before)
    qat.freeze_observer()
    expected = model(inputs)
    converted = qat.convert()
    assert isinstance(converted[0], QuantizedLinear)
    torch.testing.assert_close(converted(inputs), expected)
    assert converted[0].packed_weight.dtype == (
        torch.uint8 if bits == 4 else torch.int8
    )
    assert converted[0].packed_weight.numel() == ((15 + 1) // 2 if bits == 4 else 15)
    path = tmp_path / "qat.pt"
    qat.export(path)
    loaded, _, _ = load_quantized_model(path, original)
    torch.testing.assert_close(loaded(inputs), expected)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model_state_dict"].pop("0.weight_scale")
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_quantized_model(path, original)


def test_ptq_empty_nonfinite_and_manual_oracle():
    config = QuantizationConfig(mode="ptq", scheme="asymmetric", bits=4)
    source = nn.Sequential(nn.Linear(5, 3, bias=False))
    source[0].weight.data.copy_(torch.linspace(0.1, 2, 15).reshape(3, 5))
    ptq = PostTrainingQuantization(source, config)
    ptq.prepare()
    with pytest.raises(ValueError):
        ptq.calibrate([])
    with pytest.raises(ValueError):
        ptq.calibrate([torch.full((2, 5), float("nan"))])
    x = torch.randn(2, 5)
    ptq.calibrate([(x, torch.zeros(2))])
    model = ptq.convert()
    scale = model[0].weight_scale.reshape(-1, 1)
    zp = model[0].weight_zero_point.reshape(-1, 1)
    expected_weight = ((source[0].weight * scale.reciprocal()).round() + zp).clamp(
        -8, 7
    )
    expected_weight = (expected_weight - zp) * scale
    assert (zp != 0).any()
    torch.testing.assert_close(model(x), x @ expected_weight.T)
    assert config.to_dict() == copy.deepcopy(config).to_dict()


@pytest.mark.parametrize("value", [0.0, 2.0, -2.0])
def test_constant_affine_ranges(value):
    quant = FakeQuantize(QuantizationConfig(scheme="asymmetric", per_channel=False))
    x = torch.full((2, 3), value)
    torch.testing.assert_close(quant(x), x)
    assert torch.isfinite(quant.scale).all()
    assert ((quant.zero_point >= -128) & (quant.zero_point <= 127)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("per_channel", [False, True])
def test_conv_qat_resume_and_dtype(dtype, per_channel):
    device = torch.device(
        "cuda" if dtype == torch.float16 and torch.cuda.is_available() else "cpu"
    )
    if dtype == torch.float16 and device.type == "cpu":
        try:
            nn.functional.pad(
                torch.ones(1, 1, 3, 3, dtype=dtype), (1, 1, 1, 1), mode="reflect"
            )
        except RuntimeError as error:
            if "not implemented" in str(error):
                pytest.skip(
                    "This PyTorch version does not implement CPU float16 reflection padding"
                )
            raise
    source = nn.Conv2d(4, 6, 3, padding=1, groups=2, padding_mode="reflect").to(dtype)
    source = source.to(device)
    config = QuantizationConfig(
        bits=4, per_channel=per_channel, quantize_activations=True
    )
    qat = QuantizationAwareTraining(source, config)
    model = qat.prepare()
    x = torch.randn(2, 4, 6, 6, device=device).to(dtype)
    model(x).float().square().mean().backward()
    assert torch.isfinite(model.weight.grad).all()
    assert model.weight.grad.abs().sum() > 0
    qat.freeze_observer()
    resumed = QuantizationAwareTraining(source, config).prepare()
    state = copy.deepcopy(model.state_dict())
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    resumed.load_state_dict(state, strict=True)
    torch.testing.assert_close(resumed(x), model(x))
    torch.testing.assert_close(qat.convert()(x), model(x))
    resumed.half()
    assert resumed.weight_fake_quant.scale.dtype == torch.float32


def test_selection_and_private_configuration():
    source = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    config = QuantizationConfig(exclude_modules=["1"])
    qat = QuantizationAwareTraining(source, config)
    config.exclude_modules.clear()
    prepared = qat.prepare()
    assert type(prepared[1]) is nn.Linear
    assert type(source[0]) is nn.Linear
    source[1].weight = source[0].weight
    with pytest.raises(ValueError, match="tied"):
        QuantizationAwareTraining(source, QuantizationConfig()).prepare()
    with pytest.raises(ValueError, match="only Linear"):
        QuantizationConfig(quantizable_layers=["ConvTranspose2d"])


@pytest.mark.parametrize("bits", [4, 8])
def test_llama_ptq_qat_checkpoint(bits, tmp_path):
    transformers = pytest.importorskip("transformers")
    llama_config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    source = transformers.LlamaForCausalLM(llama_config).eval()
    batch = {
        "input_ids": torch.randint(0, 32, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    config = QuantizationConfig(bits=bits, mode="ptq")
    ptq = PostTrainingQuantization(source, config)
    ptq.prepare()
    ptq.calibrate([batch])
    model = ptq.convert()
    assert type(model.lm_head) is nn.Linear
    path = tmp_path / "llama.pt"
    ptq.export(path)
    loaded, _, _ = load_quantized_model(path, source)
    torch.testing.assert_close(loaded(**batch).logits, model(**batch).logits)
    qat = QuantizationAwareTraining(source, QuantizationConfig(bits=bits))
    prepared = qat.prepare()
    parameter = prepared.model.layers[0].self_attn.q_proj.weight
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD(prepared.parameters(), lr=0.1)
    prepared(**batch, labels=batch["input_ids"]).loss.backward()
    assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, parameter)
    qat.freeze_observer()
    torch.testing.assert_close(qat.convert()(**batch).logits, prepared(**batch).logits)


@pytest.mark.parametrize(
    "corruption", ["scale", "dtype", "manifest", "schema", "extra"]
)
def test_strict_checkpoint_rejection(corruption, tmp_path):
    source = nn.Sequential(nn.Linear(3, 2))
    ptq = PostTrainingQuantization(source, QuantizationConfig(mode="ptq"))
    ptq.prepare()
    ptq.calibrate([torch.randn(2, 3)])
    ptq.convert()
    path = tmp_path / "bad.pt"
    ptq.export(path)
    checkpoint = torch.load(path, weights_only=True)
    if corruption == "scale":
        checkpoint["model_state_dict"]["0.weight_scale"].zero_()
    elif corruption == "dtype":
        checkpoint["model_state_dict"]["0.packed_weight"] = checkpoint[
            "model_state_dict"
        ]["0.packed_weight"].float()
    elif corruption == "manifest":
        checkpoint["selected_modules"] = []
    elif corruption == "schema":
        checkpoint["schema_version"] = 999
    else:
        checkpoint["model_state_dict"]["unexpected"] = torch.tensor(1)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        load_quantized_model(path, source)
    with pytest.raises(ValueError, match="only pytorch"):
        ptq.export(path, format="onnx")


def test_nested_export_and_nonquantized_scale(tmp_path):
    source = nn.Sequential(nn.Linear(3, 2))
    source.register_parameter("scale", nn.Parameter(torch.tensor(-2.0)))
    ptq = PostTrainingQuantization(source, QuantizationConfig(mode="ptq"))
    ptq.prepare()
    with patch(
        "parascale.quantization.ptq.convert_model",
        side_effect=AssertionError("calibration must not convert"),
    ):
        ptq.calibrate([torch.randn(2, 3)])
    ptq.convert()
    path = tmp_path / "nested" / "model.pt"
    ptq.export(path)
    restored, _, _ = load_quantized_model(path, source)
    assert restored.scale.item() == -2.0


def test_qat_export_remains_fresh_after_conversion(tmp_path):
    torch.manual_seed(42)
    source = nn.Sequential(nn.Linear(3, 2))
    qat = QuantizationAwareTraining(source, QuantizationConfig())
    model = qat.prepare()
    x = torch.randn(2, 3)
    model(x)
    qat.freeze_observer()
    first = qat.convert()(x)
    with torch.no_grad():
        model[0].weight.mul_(0.5)
    expected = model(x)
    assert not torch.allclose(first, expected)
    path = tmp_path / "fresh.pt"
    qat.export(path)
    loaded, _, _ = load_quantized_model(path, source)
    torch.testing.assert_close(loaded(x), expected)


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("bits", [4, 8])
def test_native_rounding_boundary_conversion(per_channel, bits):
    torch.manual_seed(42)
    source = nn.Linear(20, 10, bias=False)
    qat = QuantizationAwareTraining(
        source, QuantizationConfig(bits=bits, per_channel=per_channel)
    )
    model = qat.prepare()
    x = torch.eye(20)
    model(x)
    qat.freeze_observer()
    with torch.no_grad():
        model.weight.mul_(0.5)
    torch.testing.assert_close(qat.convert()(x), model(x), rtol=0, atol=0)


def test_qat_rejects_nonfinite_updated_weights():
    qat = QuantizationAwareTraining(nn.Linear(3, 2), QuantizationConfig())
    model = qat.prepare()
    model(torch.randn(2, 3))
    qat.freeze_observer()
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        qat.convert()


class WeightCastingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.layer(x.to(self.layer.weight.dtype))


@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_converted_weight_interface(mode, bits, dtype, tmp_path):
    source = WeightCastingModel().to(dtype)
    config = QuantizationConfig(mode=mode, bits=bits)
    engine = (PostTrainingQuantization if mode == "ptq" else QuantizationAwareTraining)(
        source, config
    )
    prepared = engine.prepare()
    x = torch.randn(2, 3)
    if mode == "ptq":
        engine.calibrate([x])
    else:
        prepared(x)
        engine.freeze_observer()
    expected = prepared(x)
    converted = engine.convert()
    assert converted.layer.weight.shape == source.layer.weight.shape
    assert converted.layer.weight.dtype == dtype
    torch.testing.assert_close(converted(x), expected)
    assert converted.layer.packed_weight.dtype == (
        torch.uint8 if bits == 4 else torch.int8
    )
    with pytest.raises(AttributeError):
        converted.layer.weight = torch.zeros(2, 3)
    path = tmp_path / "casting.pt"
    engine.export(path)
    loaded, _, _ = load_quantized_model(path, source)
    torch.testing.assert_close(loaded(x), expected)
    assert loaded.layer.weight.dtype == dtype
    converted.float()
    assert converted.layer.weight.dtype == torch.float32


@pytest.mark.parametrize(
    "change",
    [
        {"bits": 4},
        {"scheme": "asymmetric"},
        {"quantize_activations": True},
        {"observer_type": "moving_average"},
        {"moving_average_ratio": 0.2},
        {"per_channel": False},
    ],
)
def test_resume_rejects_numerical_configuration_mismatch(change):
    source = nn.Linear(3, 2)
    model = QuantizationAwareTraining(source, QuantizationConfig()).prepare()
    model(torch.randn(2, 3))
    target = QuantizationAwareTraining(source, QuantizationConfig(**change)).prepare()
    with pytest.raises(RuntimeError, match="configuration"):
        target.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("kind", ["norm", "pool"])
def test_template_rejects_same_shape_semantic_mismatch(kind, tmp_path):
    source = nn.Sequential(
        nn.Conv2d(2, 2, 1), nn.LayerNorm([2, 4, 4]), nn.AvgPool2d(2, stride=1)
    )
    target = copy.deepcopy(source)
    if kind == "norm":
        target[1].eps = 0.1
    else:
        target[2].stride = 2
    ptq = PostTrainingQuantization(source, QuantizationConfig(mode="ptq"))
    ptq.prepare()
    ptq.calibrate([torch.randn(1, 2, 4, 4)])
    ptq.convert()
    path = tmp_path / "semantics.pt"
    ptq.export(path)
    with pytest.raises(ValueError, match="manifest"):
        load_quantized_model(path, target)


@pytest.mark.parametrize("field", ["rope_theta", "num_attention_heads"])
def test_llama_signature_rejects_semantic_config(field, tmp_path):
    transformers = pytest.importorskip("transformers")
    from parascale.quantization.core import signature

    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    source = transformers.LlamaForCausalLM(config)
    target = copy.deepcopy(source)
    setattr(target.config, field, getattr(target.config, field, 10000) * 2)
    assert signature(source) != signature(target)
    ptq = PostTrainingQuantization(source, QuantizationConfig(mode="ptq"))
    ptq.prepare()
    ptq.calibrate([{"input_ids": torch.randint(0, 16, (1, 4))}])
    ptq.convert()
    path = tmp_path / "llama-semantics.pt"
    ptq.export(path)
    with pytest.raises(ValueError, match="manifest"):
        load_quantized_model(path, target)


def test_signature_ignores_only_hf_deployment_metadata():
    transformers = pytest.importorskip("transformers")
    from parascale.quantization.core import signature

    source = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    target = copy.deepcopy(source)
    target.config._name_or_path = "/another/local/model"
    target.config.transformers_version = "deployment-version"
    target.config.torch_dtype = torch.float16
    target.config._attn_implementation = "eager"
    assert signature(source) == signature(target)


@pytest.mark.parametrize("strict", [False, True])
def test_resume_requires_intact_configuration_fingerprint(strict):
    quantizer = FakeQuantize(QuantizationConfig())
    quantizer(torch.randn(2, 3))
    state = copy.deepcopy(quantizer.state_dict())
    assert state["config_fingerprint"].dtype == torch.uint8
    state["config_fingerprint"][0] ^= 1
    target = FakeQuantize(QuantizationConfig())
    with pytest.raises(RuntimeError, match="configuration"):
        target.load_state_dict(state, strict=strict)
    state.pop("config_fingerprint")
    with pytest.raises(RuntimeError, match="configuration"):
        target.load_state_dict(state, strict=strict)


def test_reference_configuration_only():
    config = QuantizationConfig()
    assert config.backend == "reference" and config.fuse_modules is False
    with pytest.raises(ValueError, match="fusion"):
        QuantizationConfig(fuse_modules=True)
    with pytest.raises(ValueError, match="reference"):
        QuantizationConfig(backend="fbgemm")


@pytest.mark.parametrize("converted_input", [False, True])
def test_qat_conversion_requires_prepared_layers(converted_input):
    config = QuantizationConfig()
    source = nn.Sequential(nn.Linear(3, 2))
    model = source
    if converted_input:
        qat = QuantizationAwareTraining(source, config)
        prepared = qat.prepare()
        prepared(torch.randn(2, 3))
        qat.freeze_observer()
        model = convert_qat_model(prepared, config)
        assert isinstance(model[0], QuantizedLinear)
    with pytest.raises(ValueError, match="prepared"):
        convert_qat_model(model, config)


@pytest.mark.parametrize(
    "overrides", [{"bits": 4}, {"scheme": "asymmetric"}, {"quantize_activations": True}]
)
def test_qat_conversion_rejects_configuration_changes(overrides):
    config = QuantizationConfig()
    qat = QuantizationAwareTraining(nn.Sequential(nn.Linear(3, 2)), config)
    prepared = qat.prepare()
    prepared(torch.randn(2, 3))
    changed = QuantizationConfig.from_dict({**config.to_dict(), **overrides})
    with pytest.raises(ValueError, match="conversion configuration"):
        convert_qat_model(prepared, changed)
