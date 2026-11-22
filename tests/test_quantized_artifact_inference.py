# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午7:10
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Stored PTQ/QAT code identity and strict fused deployment admission."""

import copy
from importlib import import_module

import pytest
import torch
from torch import nn

from parascale.quantization import (
    PostTrainingQuantization,
    QuantizationAwareTraining,
    QuantizationConfig,
    QuantizedConv2d,
    load_quantized_model,
)


def _api():
    return import_module("parascale.quantization.inference_artifact")


def _reference(mode="ptq", bias=True, nested=True, **kwargs):
    torch.manual_seed(61)
    source = nn.Linear(128, 128, bias=bias)
    if nested:
        source = nn.Sequential(source)
    config = QuantizationConfig(mode=mode, **kwargs)
    controller = (
        PostTrainingQuantization if mode == "ptq" else QuantizationAwareTraining
    )(source, config)
    prepared = controller.prepare()
    inputs = torch.randn(2, 128)
    if mode == "ptq":
        controller.calibrate([inputs])
    else:
        optimizer = torch.optim.SGD(prepared.parameters(), lr=0.01)
        prepared(inputs).square().mean().backward()
        optimizer.step()
        controller.freeze_observer()
    return source, controller, controller.convert()


@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("nested", [False, True])
def test_reference_selection_accepts_canonical_artifact(mode, nested, tmp_path):
    source, controller, _ = _reference(mode, nested=nested)
    path = tmp_path / "quantized.pt"
    controller.export(path)
    loaded, _, _ = load_quantized_model(path, source)
    selected = _api()._select_reference_layers(loaded)
    assert list(selected) == (["0"] if nested else [""])


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("bits", "INT8"),
        ("per_tensor", "per-channel"),
        ("activation", "weight-only"),
        ("zero", "zero-point"),
        ("scale_nan", "scale"),
        ("scale_zero", "scale"),
        ("scale_shape", "scale"),
        ("scale_dtype", "scale"),
        ("code_shape", "codes"),
        ("code_dtype", "codes"),
        ("zero_dtype", "zero-point"),
        ("zero_shape", "zero-point"),
        ("bias_nan", "bias"),
        ("bias_shape", "bias"),
        ("bias_overflow", "BF16"),
        ("weight_overflow", "BF16"),
        ("unaligned", "aligned"),
        ("fake_activation", "weight-only"),
    ],
)
def test_reference_selection_rejects_unsupported_or_corrupt_storage(
    corruption, message
):
    _, _, model = _reference()
    layer = model[0]
    if corruption == "bits":
        layer.quant_config.bits = 4
    elif corruption == "per_tensor":
        layer.quant_config.per_channel = False
    elif corruption == "activation":
        layer.quant_config.quantize_activations = True
    elif corruption == "zero":
        layer.weight_zero_point[0] = 1
    elif corruption == "scale_nan":
        layer.weight_scale[0] = float("nan")
    elif corruption == "scale_zero":
        layer.weight_scale[0] = 0
    elif corruption == "scale_shape":
        layer.weight_scale = layer.weight_scale[:-1]
    elif corruption == "scale_dtype":
        layer.weight_scale = layer.weight_scale.half()
    elif corruption == "code_shape":
        layer.packed_weight = layer.packed_weight[:-1]
    elif corruption == "code_dtype":
        layer.packed_weight = layer.packed_weight.float()
    elif corruption == "zero_dtype":
        layer.weight_zero_point = layer.weight_zero_point.float()
    elif corruption == "zero_shape":
        layer.weight_zero_point = layer.weight_zero_point[:-1]
    elif corruption == "bias_nan":
        layer.bias[0] = float("nan")
    elif corruption == "bias_shape":
        layer.bias = layer.bias[:-1]
    elif corruption == "bias_overflow":
        layer.bias[0] = 3.4e38
    elif corruption == "weight_overflow":
        layer.packed_weight[0] = 127
        layer.weight_scale[0] = 3.4e38
    elif corruption == "unaligned":
        layer.in_features = 127
    elif corruption == "fake_activation":
        layer.activation_fake_quant = nn.ReLU()
    with pytest.raises(ValueError, match=message):
        _api()._select_reference_layers(model)


def test_reference_selection_rejects_conv_and_aliases_without_partial_coverage():
    _, _, model = _reference()
    model.add_module("conv", QuantizedConv2d(1, 1, 1))
    with pytest.raises(ValueError, match="Linear"):
        _api()._select_reference_layers(model)
    del model.conv
    model.add_module("alias", model[0])
    with pytest.raises(ValueError, match="shared"):
        _api()._select_reference_layers(model)


def test_reference_selection_rejects_functional_weight_parents():
    _, _, model = _reference(nested=False)
    parent = nn.TransformerEncoderLayer(128, 4, 128)
    parent.linear1 = model
    with pytest.raises(ValueError, match="functional"):
        _api()._select_reference_layers(parent)


def test_reference_selection_rejects_float_and_prepared_models():
    source = nn.Linear(128, 128)
    with pytest.raises(ValueError, match="reference"):
        _api()._select_reference_layers(source)
    prepared = QuantizationAwareTraining(source, QuantizationConfig()).prepare()
    with pytest.raises(ValueError, match="converted"):
        _api()._select_reference_layers(prepared)


def test_cpu_rejection_is_before_backend_loading_and_preserves_source(monkeypatch):
    _, _, model = _reference()
    api = _api()

    def unexpected_load():
        pytest.fail("backend must not load before CUDA validation")

    monkeypatch.setattr(api, "_load_gemlite", unexpected_load)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="CUDA"):
        api.prepare_quantized_inference_model(model, inplace=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("hook", ["module", "global", "compiled", "serialization"])
def test_artifact_fusion_rejects_runtime_state_before_replacement(hook):
    _, _, model = _reference()
    handle = None
    if hook == "module":
        model.register_forward_pre_hook(lambda module, inputs: inputs)
    elif hook == "global":
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda *args: None
        )
    elif hook == "compiled":
        model._compiled_call_impl = lambda inputs: model.forward(inputs)
    else:
        model.register_load_state_dict_post_hook(lambda *args: None)
    try:
        with pytest.raises(ValueError, match="execution|serialization"):
            _api().prepare_quantized_inference_model(model)
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("bias", [False, True])
def test_packing_transfers_exact_codes_and_scales_without_source_aliases(mode, bias):
    _, _, model = _reference(mode, bias=bias)
    layer = model[0].bfloat16()
    expected_codes = layer.packed_weight.clone()
    expected_scales = layer.weight_scale.clone()

    class Processor:
        def __init__(self, *, device, dtype, fp32_scale):
            assert device == "cpu" and dtype == torch.bfloat16 and fp32_scale is True

        def from_weights(self, *, weight, bias, scales):
            assert weight.dtype == torch.int8
            assert bias is None or bias.dtype == torch.bfloat16
            torch.testing.assert_close(weight.flatten(), expected_codes, rtol=0, atol=0)
            torch.testing.assert_close(
                scales.flatten(), expected_scales, rtol=0, atol=0
            )
            kernel = nn.Module()
            kernel.in_features, kernel.out_features = weight.shape[::-1]
            kernel.register_buffer("W_q", weight.t())
            kernel.register_buffer("scales", scales.reshape(1, -1))
            kernel.register_buffer("zeros", torch.empty(0, dtype=torch.int32))
            kernel.register_buffer("bias", bias)
            kernel.W_group_mode, kernel.channel_scale_mode = 2, 0
            kernel.get_meta_args = lambda: [
                0,
                8,
                kernel.in_features,
                255,
                1,
                2,
                2,
                0,
                0,
                0,
                2,
                1,
            ]
            return kernel

    packed = _api()._pack_reference_layer(layer, Processor)
    assert (
        packed.W_q.untyped_storage().data_ptr()
        != layer.packed_weight.untyped_storage().data_ptr()
    )
    assert (
        packed.scales.untyped_storage().data_ptr()
        != layer.weight_scale.untyped_storage().data_ptr()
    )
    if bias:
        torch.testing.assert_close(packed.bias, layer.bias.bfloat16(), rtol=0, atol=0)
        assert packed.bias.data_ptr() != layer.bias.data_ptr()
    packed.W_q.zero_()
    packed.scales.zero_()
    torch.testing.assert_close(layer.packed_weight, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_scale, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize("changed", ["codes", "scales", "bias", "scaling_mode"])
def test_packing_rejects_backend_changes_even_when_input_storage_is_mutated(changed):
    _, _, model = _reference()
    layer = model[0].bfloat16()
    original = copy.deepcopy(layer.state_dict())

    class Processor:
        def __init__(self, **kwargs):
            pass

        def from_weights(self, *, weight, bias, scales):
            if changed == "codes":
                weight.zero_()
            elif changed == "scales":
                scales.mul_(2)
            elif changed == "bias":
                bias.zero_()
            kernel = nn.Module()
            kernel.in_features, kernel.out_features = weight.shape[::-1]
            kernel.register_buffer("W_q", weight.t())
            kernel.register_buffer("scales", scales.reshape(1, -1))
            kernel.register_buffer("zeros", torch.empty(0, dtype=torch.int32))
            kernel.register_buffer("bias", bias)
            kernel.W_group_mode = 0 if changed == "scaling_mode" else 2
            kernel.channel_scale_mode = 0
            return kernel

    with pytest.raises(RuntimeError, match="changed.*reference"):
        _api()._pack_reference_layer(layer, Processor)
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(128, 128), (576, 1536), (2048, 5632)])
def test_cuda_fused_artifact_codes_numerics_dispatch_and_reload(
    mode, bias, shape, tmp_path
):
    pytest.importorskip("gemlite")
    from parascale.quantization import prepare_quantized_inference_model

    torch.manual_seed(71)
    in_features, out_features = shape
    source = nn.Sequential(nn.Linear(in_features, out_features, bias=bias)).cuda()
    controller = (
        PostTrainingQuantization if mode == "ptq" else QuantizationAwareTraining
    )(source, QuantizationConfig(mode=mode))
    prepared = controller.prepare()
    calibration = torch.randn(2, in_features, device="cuda")
    if mode == "ptq":
        controller.calibrate([calibration])
    else:
        optimizer = torch.optim.SGD(prepared.parameters(), lr=0.01)
        prepared(calibration).square().mean().backward()
        optimizer.step()
        controller.freeze_observer()
    controller.convert()
    path = tmp_path / "canonical.pt"
    controller.export(path)
    reference, _, _ = load_quantized_model(path, source.cpu())
    reference = reference.cuda().bfloat16().eval()
    original = copy.deepcopy(reference.state_dict())
    fused, report = prepare_quantized_inference_model(reference)
    assert report["requantized"] is False
    assert report["stored_codes_and_scales_preserved"] is True
    assert report["converted_layers"] == ["0"]
    assert report["packed_weight_bytes"] > in_features * out_features
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    with torch.inference_mode():
        for batch_shape in [(1,), (4,), (2, 16)]:
            inputs = torch.randn(
                *batch_shape, in_features, device="cuda", dtype=torch.bfloat16
            )
            expected, actual = reference(inputs).float(), fused(inputs).float()
            assert torch.isfinite(actual).all()
            assert (actual - expected).norm() / expected.norm() < 0.01
        assert fused(
            torch.empty(0, in_features, device="cuda", dtype=torch.bfloat16)
        ).shape == (0, out_features)
        # State copying is supported into an already identically packed runtime.
        restored, _ = prepare_quantized_inference_model(reference)
        restored.load_state_dict(copy.deepcopy(fused.state_dict()), strict=True)
        torch.testing.assert_close(restored(inputs), fused(inputs), rtol=0, atol=0)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as trace:
            fused(inputs)
            torch.cuda.synchronize()
        names = [event.key.lower() for event in trace.key_averages()]
        assert any(
            "gemlite" in name or "gemv_int" in name or "gemm_int" in name
            for name in names
        )
        fused.bfloat16()
        with pytest.raises(ValueError, match="scales"):
            fused(inputs)


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_dense_artifact_control_materializes_the_same_effective_weights(mode):
    from tests.benchmarks.scripts.quantized_artifact_fixture import (
        materialize_reference_weights,
    )

    _, _, model = _reference(mode)
    model.bfloat16()
    inputs = torch.randn(2, 128, dtype=torch.bfloat16)
    expected = model(inputs)
    dense = materialize_reference_weights(copy.deepcopy(model))
    assert type(dense[0]) is nn.Linear
    torch.testing.assert_close(dense(inputs), expected, rtol=0, atol=0)
