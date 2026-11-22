# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午10:02
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""The fused adapter owns live tensor state without backend Python dispatch."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def packed_kernel():
    kernel = torch.nn.Module()
    kernel.in_features = kernel.out_features = 128
    kernel.register_buffer("W_q", torch.ones(128, 128, dtype=torch.int8))
    kernel.register_buffer("scales", torch.full((1, 128), 0.25))
    kernel.register_buffer("zeros", torch.empty(0, dtype=torch.int32))
    kernel.register_buffer("bias", torch.zeros(128, dtype=torch.bfloat16))
    kernel.get_meta_args = lambda: [0, 8, 128, 255, 1, 2, 2, 0, 0, 0, 2, 1]
    return kernel


def test_adapter_reads_replaced_registered_state_without_backend_entry(monkeypatch):
    from parascale.quantization.inference_gemlite import GemliteInt8Linear

    source = packed_kernel()
    model = GemliteInt8Linear(source)
    assert set(dict(model.named_buffers())) == {"W_q", "scales", "zeros", "bias"}
    assert len(list(model.modules())) == 1
    observed = []

    def fused(inputs, bias, tensors, metadata):
        observed.append((tensors, metadata))
        assert metadata == source.get_meta_args()
        weight, scales, zeros = tensors
        assert zeros.numel() == 0
        return torch.nn.functional.linear(
            inputs, (weight.float() * scales).t().bfloat16(), bias
        )

    monkeypatch.setattr(
        torch.ops.gemlite,
        "forward_functional",
        SimpleNamespace(default=fused),
        raising=False,
    )
    inputs = torch.ones(1, 128, dtype=torch.bfloat16)
    torch.testing.assert_close(model(inputs), torch.full_like(inputs, 32))
    model.scales = torch.full_like(model.scales, 0.5)
    torch.testing.assert_close(model(inputs), torch.full_like(inputs, 64))
    assert observed[-1][0][1] is model.scales
    state = copy.deepcopy(model.state_dict())
    model.W_q.zero_()
    torch.testing.assert_close(model(inputs), torch.zeros_like(inputs))
    model.load_state_dict(state)
    torch.testing.assert_close(model(inputs), torch.full_like(inputs, 64))
    model.bfloat16()
    with pytest.raises(ValueError, match="FP32 scales"):
        model(inputs)


@pytest.mark.parametrize("index", range(11))
def test_adapter_rejects_changed_execution_metadata(index):
    from parascale.quantization.inference_gemlite import GemliteInt8Linear

    kernel = packed_kernel()
    metadata = kernel.get_meta_args()
    metadata[index] += 1
    kernel.get_meta_args = lambda: metadata
    with pytest.raises(ValueError, match="metadata"):
        GemliteInt8Linear(kernel)


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        [0] * 13,
        [False, 8, 128, 255, 1, 2, 2, 0, 0, 0, 2, 1],
        [0, 8, 128, 255, 1, 2, 2, 0, 0, 0, 2, 2],
    ],
)
def test_adapter_rejects_malformed_metadata(metadata):
    from parascale.quantization.inference_gemlite import GemliteInt8Linear

    kernel = packed_kernel()
    kernel.get_meta_args = lambda: metadata
    with pytest.raises(ValueError, match="metadata"):
        GemliteInt8Linear(kernel)


@pytest.mark.parametrize(
    "invalid",
    [
        "weight_dtype",
        "weight_shape",
        "weight_stride",
        "scale_shape",
        "scale_device",
        "zeros",
        "bias",
    ],
)
def test_adapter_rejects_invalid_replacement_before_dispatch(invalid):
    from parascale.quantization.inference_gemlite import GemliteInt8Linear

    model = GemliteInt8Linear(packed_kernel())
    if invalid == "weight_dtype":
        model.W_q = model.W_q.float()
    elif invalid == "weight_shape":
        model.W_q = model.W_q[:, :-1]
    elif invalid == "weight_stride":
        model.W_q = model.W_q.t()
    elif invalid == "scale_shape":
        model.scales = model.scales[:, :-1]
    elif invalid == "scale_device":
        model.scales = model.scales.to("meta")
    elif invalid == "zeros":
        model.zeros = torch.zeros(1)
    else:
        model.bias = model.bias.float()
    with pytest.raises(ValueError, match="packed tensor state"):
        model(torch.ones(1, 128, dtype=torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_adapter_reads_buffer_replacement_mutation_and_restore():
    pytest.importorskip("gemlite")
    from parascale.quantization import (
        InferenceQuantizationConfig,
        quantize_inference_model,
    )

    torch.compiler.reset()
    try:
        source = torch.nn.Linear(128, 128, bias=False).cuda().bfloat16().eval()
        model, _ = quantize_inference_model(
            source, InferenceQuantizationConfig(backend="gemlite_int8")
        )
        saved = copy.deepcopy(model.state_dict())
        weight_stride = model.W_q.stride()
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        with torch.inference_mode():
            inputs = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
            initial = compiled(inputs).clone()
            model.scales = model.scales * 2
            actual = compiled(inputs).clone()
            torch.testing.assert_close(actual, model(inputs), rtol=0, atol=0)
            assert actual.float().norm() > initial.float().norm() * 1.8
            model.W_q = torch.zeros_like(model.W_q)
            torch.testing.assert_close(compiled(inputs), torch.zeros_like(initial))
            model.bias = torch.ones(128, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(compiled(inputs), torch.ones_like(initial))
            model.bias = None
            model.load_state_dict(saved)
            torch.testing.assert_close(compiled(inputs), initial, rtol=0, atol=0)
            for rows in (2, 1):
                batch = inputs.expand(rows, -1).contiguous()
                torch.testing.assert_close(
                    compiled(batch), model(batch), rtol=0, atol=0
                )
            model.cpu().cuda()
            assert model.W_q.stride() == weight_stride
            for name, expected in saved.items():
                torch.testing.assert_close(
                    getattr(model, name), expected, rtol=0, atol=0
                )
            torch.testing.assert_close(compiled(inputs), initial, rtol=0, atol=0)
            cloned = copy.deepcopy(model)
            assert cloned.W_q.stride() == weight_stride
            assert cloned.W_q.data_ptr() != model.W_q.data_ptr()
            assert cloned.scales.data_ptr() != model.scales.data_ptr()
            for name, expected in saved.items():
                torch.testing.assert_close(
                    getattr(cloned, name), expected, rtol=0, atol=0
                )
            torch.testing.assert_close(cloned(inputs), initial, rtol=0, atol=0)
    finally:
        torch.compiler.reset()
