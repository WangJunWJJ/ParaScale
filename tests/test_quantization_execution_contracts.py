# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午6:42
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Reject execution state that layer replacement or checkpointing would lose."""

import copy

import pytest
import torch
from torch import nn

from parascale.quantization import (
    PostTrainingQuantization,
    QuantizationAwareTraining,
    QuantizationConfig,
    load_quantized_model,
)


def _controller(model, mode, **kwargs):
    cls = PostTrainingQuantization if mode == "ptq" else QuantizationAwareTraining
    return cls(model, QuantizationConfig(mode=mode, **kwargs))


def _attach(layer, kind):
    if kind == "pre":
        layer.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
    elif kind == "post":
        layer.register_forward_hook(lambda module, args, output: output * 2)
    elif kind == "backward_pre":
        layer.register_full_backward_pre_hook(lambda module, gradients: gradients)
    elif kind == "backward":
        layer.register_full_backward_hook(lambda module, inputs, outputs: inputs)
    elif kind == "forward_override":
        forward = layer.forward
        layer.forward = lambda inputs: forward(inputs * 2)
    elif kind == "compiled_call":
        layer._compiled_call_impl = lambda inputs: layer.forward(inputs * 2)
    elif kind == "parameter":
        layer.weight.register_hook(lambda grad: grad * 2)
    elif kind == "post_accumulate":
        layer.weight.register_post_accumulate_grad_hook(lambda parameter: None)
    else:
        raise AssertionError(kind)


def _fixture(layer_type, nested):
    if layer_type == "linear":
        layer = nn.Linear(2, 2, bias=False)
        inputs = torch.ones(2, 2)
        model = nn.Sequential(nn.Linear(2, 2), layer) if nested else layer
    else:
        layer = nn.Conv2d(2, 2, 1, bias=False)
        inputs = torch.ones(2, 2, 3, 3)
        model = nn.Sequential(nn.Conv2d(2, 2, 1), layer) if nested else layer
    return model, layer, inputs


def _observe(controller, inputs, mode):
    prepared = controller.prepare()
    if mode == "ptq":
        controller.calibrate([inputs])
    else:
        prepared(inputs).square().mean().backward()
        controller.freeze_observer()
    return prepared


@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("layer_type", ["linear", "conv"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "pre",
        "post",
        "backward_pre",
        "backward",
        "forward_override",
        "compiled_call",
        "parameter",
        "post_accumulate",
    ],
)
def test_prepare_rejects_execution_state_without_mutating_source(
    mode, layer_type, nested, kind
):
    source, layer, _ = _fixture(layer_type, nested)
    _attach(layer, kind)
    before = copy.deepcopy(source.state_dict())
    controller = _controller(source, mode)
    with pytest.raises(ValueError, match="execution.*(1|<root>|weight)"):
        controller.prepare()
    assert type(layer) in (nn.Linear, nn.Conv2d)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert controller.quantized_model is None


@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("layer_type", ["linear", "conv"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["pre", "post", "forward_override", "parameter"])
def test_conversion_rejects_execution_state_added_after_observation(
    mode, layer_type, nested, kind
):
    source, _, inputs = _fixture(layer_type, nested)
    controller = _controller(source, mode)
    prepared = _observe(controller, inputs, mode)
    layer = prepared[1] if nested else prepared
    _attach(layer, kind)
    with pytest.raises(ValueError, match="execution"):
        controller.convert()
    assert controller.quantized_model is None
    assert hasattr(layer, "weight_fake_quant")


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_conversion_rejects_hook_on_discarded_quantizer(mode):
    source, _, inputs = _fixture("linear", False)
    controller = _controller(source, mode)
    prepared = _observe(controller, inputs, mode)
    _attach(prepared.weight_fake_quant, "post")
    with pytest.raises(ValueError, match="execution.*weight_fake_quant"):
        controller.convert()


@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("location", ["parent", "excluded", "selected"])
def test_export_rejects_unserializable_execution_state_atomically(
    mode, location, tmp_path
):
    source, _, inputs = _fixture("linear", True)
    controller = _controller(source, mode, exclude_modules=["1"])
    prepared = _observe(controller, inputs, mode)
    converted = controller.convert()
    model = converted if mode == "ptq" else prepared
    target = (
        model if location == "parent" else model[1 if location == "excluded" else 0]
    )
    _attach(target, "post")
    path = tmp_path / "existing.pt"
    path.write_bytes(b"previous valid artifact")
    with pytest.raises(ValueError, match="execution"):
        controller.export(path)
    assert path.read_bytes() == b"previous valid artifact"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("location", ["parent", "excluded", "selected"])
@pytest.mark.parametrize("kind", ["pre", "post", "forward_override"])
def test_load_rejects_template_execution_state(location, kind, tmp_path):
    source, _, inputs = _fixture("linear", True)
    controller = _controller(source, "ptq", exclude_modules=["1"])
    _observe(controller, inputs, "ptq")
    controller.convert()
    path = tmp_path / "model.pt"
    controller.export(path)
    target = (
        source if location == "parent" else source[1 if location == "excluded" else 0]
    )
    _attach(target, kind)
    with pytest.raises(ValueError, match="execution"):
        load_quantized_model(path, source)


@pytest.mark.parametrize("location", ["parent", "excluded"])
@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_in_memory_conversion_preserves_non_replaced_module_hooks(location, mode):
    source, _, inputs = _fixture("linear", True)
    target = source if location == "parent" else source[1]
    _attach(target, "post")
    controller = _controller(source, mode, exclude_modules=["1"])
    prepared = _observe(controller, inputs, mode)
    expected = prepared(inputs)
    converted = controller.convert()
    torch.testing.assert_close(converted(inputs), expected)
    assert (
        len((converted if location == "parent" else converted[1])._forward_hooks) == 1
    )


def test_prepare_rejects_gradient_hooks_even_on_excluded_parameters():
    source, layer, _ = _fixture("linear", True)
    _attach(layer, "parameter")
    controller = _controller(source, "qat", exclude_modules=["1"])
    with pytest.raises(ValueError, match="execution.*1.weight"):
        controller.prepare()


@pytest.mark.parametrize("location", ["parent", "excluded"])
@pytest.mark.parametrize("stage", ["prepare", "convert", "export"])
def test_copy_rejects_compiled_calls_outside_selected_layers(location, stage, tmp_path):
    source, _, inputs = _fixture("linear", True)
    controller = _controller(source, "qat", exclude_modules=["1"])
    prepared = _observe(controller, inputs, "qat")
    model = source if stage == "prepare" else prepared
    _attach(model if location == "parent" else model[1], "compiled_call")
    action = {
        "prepare": controller.prepare,
        "convert": controller.convert,
        "export": lambda: controller.export(tmp_path / "compiled.pt"),
    }[stage]
    with pytest.raises(ValueError, match="execution override"):
        action()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["prepare", "convert", "export", "load"])
@pytest.mark.parametrize(
    "registration",
    [
        "register_module_forward_pre_hook",
        "register_module_forward_hook",
        "register_module_full_backward_pre_hook",
        "register_module_full_backward_hook",
    ],
)
def test_global_execution_hooks_are_rejected_and_removable(
    stage, registration, tmp_path
):
    source, _, inputs = _fixture("linear", True)
    controller = _controller(source, "ptq")
    _observe(controller, inputs, "ptq")
    controller.convert()
    path = tmp_path / "global.pt"
    controller.export(path)
    action = {
        "prepare": controller.prepare,
        "convert": controller.convert,
        "export": lambda: controller.export(path),
        "load": lambda: load_quantized_model(path, source),
    }[stage]
    register = getattr(torch.nn.modules.module, registration)
    handle = register(lambda *args: None)
    try:
        with pytest.raises(ValueError, match="execution.*global"):
            action()
    finally:
        handle.remove()
    action()


@pytest.mark.parametrize("stage", ["construct", "prepare", "convert", "export", "load"])
@pytest.mark.parametrize(
    "registration",
    [
        "register_state_dict_pre_hook",
        "register_state_dict_post_hook",
        "register_load_state_dict_pre_hook",
        "register_load_state_dict_post_hook",
    ],
)
def test_serialization_callbacks_are_rejected_before_they_execute(
    stage, registration, tmp_path
):
    source, _, inputs = _fixture("linear", True)
    controller = _controller(source, "ptq")
    prepared = _observe(controller, inputs, "ptq")
    converted = controller.convert()
    path = tmp_path / "serialized.pt"
    controller.export(path)
    original = path.read_bytes()
    calls = []

    def callback(*args):
        calls.append(True)

    model = {"convert": prepared, "export": converted}.get(stage, source)
    getattr(model, registration)(callback)
    action = {
        "construct": lambda: _controller(source, "ptq"),
        "prepare": controller.prepare,
        "convert": controller.convert,
        "export": lambda: controller.export(path),
        "load": lambda: load_quantized_model(path, source),
    }[stage]
    with pytest.raises(ValueError, match="serialization.*<root>"):
        action()
    assert not calls
    assert path.read_bytes() == original
