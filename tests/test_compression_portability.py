# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Cross-family reference compression contracts, not pretrained quality tests."""

import copy
import os
from collections import UserDict

import pytest

torch = pytest.importorskip("torch")

from parascale.distillation import (  # noqa: E402
    DistillationConfig,
    OnlineDistillationLoss,
)
from parascale.quantization import (  # noqa: E402
    PostTrainingQuantization,
    QuantizationAwareTraining,
    QuantizationConfig,
    load_quantized_model,
)


def test_ptq_accepts_tokenizer_style_mapping():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 2)

        def forward(self, input_ids):
            return self.proj(input_ids)

    quantizer = PostTrainingQuantization(Model(), QuantizationConfig(mode="ptq"))
    quantizer.prepare()
    batch = UserDict(input_ids=torch.ones(2, 3))
    quantizer.calibrate([batch])
    assert torch.isfinite(quantizer.convert()(**batch)).all()
    with pytest.raises(ValueError, match="finite"):
        quantizer.calibrate([UserDict(input_ids=torch.full((2, 3), float("nan")))])


def build_family(name, teacher=False):
    hf = pytest.importorskip("transformers")
    device = os.environ.get("PARASCALE_PORTABILITY_DEVICE", "cpu")
    width = 32 if teacher else 16
    layers = 2 if teacher else 1
    if name == "bert":
        model = hf.BertForSequenceClassification(
            hf.BertConfig(
                vocab_size=32,
                hidden_size=width,
                intermediate_size=width * 2,
                num_hidden_layers=layers,
                num_attention_heads=2,
                num_labels=3,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
            )
        )
        inputs = {
            "input_ids": torch.randint(0, 32, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
    elif name == "vit":
        model = hf.ViTForImageClassification(
            hf.ViTConfig(
                image_size=16,
                patch_size=8,
                hidden_size=width,
                intermediate_size=width * 2,
                num_hidden_layers=layers,
                num_attention_heads=2,
                num_labels=3,
            )
        )
        inputs = {"pixel_values": torch.randn(2, 3, 16, 16)}
    elif name == "resnet":
        model = hf.ResNetForImageClassification(
            hf.ResNetConfig(
                embedding_size=8,
                hidden_sizes=[width, width * 2, width * 4, width * 8],
                depths=[1, 1, 1, 1],
                layer_type="basic",
                num_labels=3,
            )
        )
        inputs = {"pixel_values": torch.randn(2, 3, 32, 32)}
    elif name == "qwen2":
        model = hf.Qwen2ForCausalLM(
            hf.Qwen2Config(
                vocab_size=32,
                hidden_size=width,
                intermediate_size=width * 2,
                num_hidden_layers=layers,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                tie_word_embeddings=True,
            )
        )
        inputs = {
            "input_ids": torch.randint(0, 32, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
    else:
        raise AssertionError(name)
    return model.to(device).eval(), {
        key: value.to(device) for key, value in inputs.items()
    }


def test_parameter_alias_check_scales_linearly(monkeypatch):
    from parascale.quantization.core import selected_layers

    model = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(12)])
    original = torch.Tensor.untyped_storage
    calls = []

    def storage(tensor):
        calls.append(None)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "untyped_storage", storage)
    assert len(selected_layers(model, QuantizationConfig())) == 12
    assert len(calls) <= 2 * len(list(model.parameters()))


@pytest.mark.parametrize("family", ["bert", "vit", "resnet", "qwen2"])
@pytest.mark.parametrize("bits", [4, 8])
def test_cross_family_ptq_qat_and_reload(family, bits, tmp_path):
    torch.manual_seed(123)
    source, inputs = build_family(family)
    source_state = copy.deepcopy(source.state_dict())
    ptq = PostTrainingQuantization(source, QuantizationConfig(mode="ptq", bits=bits))
    ptq.prepare()
    ptq.calibrate([inputs])
    converted = ptq.convert()
    expected = converted(**inputs).logits
    assert torch.isfinite(expected).all()
    checkpoint = tmp_path / "ptq.pt"
    ptq.export(checkpoint)
    restored, _, _ = load_quantized_model(checkpoint, source)
    torch.testing.assert_close(restored(**inputs).logits, expected)
    qat = QuantizationAwareTraining(source, QuantizationConfig(bits=bits))
    prepared = qat.prepare()
    weights = [
        module.weight
        for module in prepared.modules()
        if hasattr(module, "weight_fake_quant")
    ]
    initial = [weight.detach().clone() for weight in weights]
    optimizer = torch.optim.SGD(prepared.parameters(), lr=0.1)
    prepared(**inputs).logits.square().mean().backward()
    assert any(
        weight.grad is not None and torch.count_nonzero(weight.grad)
        for weight in weights
    )
    optimizer.step()
    assert any(
        not torch.equal(before, after) for before, after in zip(initial, weights)
    )
    qat.freeze_observer()
    expected = prepared(**inputs).logits
    torch.testing.assert_close(qat.convert()(**inputs).logits, expected)
    qat.export(checkpoint)
    restored, _, _ = load_quantized_model(checkpoint, source)
    torch.testing.assert_close(restored(**inputs).logits, expected)
    for key, value in source_state.items():
        torch.testing.assert_close(source.state_dict()[key], value)


@pytest.mark.parametrize("family", ["bert", "vit", "resnet", "qwen2"])
def test_cross_family_teacher_student_update(family):
    torch.manual_seed(321)
    student, inputs = build_family(family)
    teacher, _ = build_family(family, teacher=True)
    causal = family == "qwen2"
    labels = (
        inputs["input_ids"].clone()
        if causal
        else torch.tensor([0, 1], device=next(student.parameters()).device)
    )
    config = DistillationConfig(task="causal_lm" if causal else "classification")
    objective = OnlineDistillationLoss(student, teacher, config)
    before = [parameter.detach().clone() for parameter in student.parameters()]
    teacher_before = copy.deepcopy(teacher.state_dict())
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    loss = objective(student(**inputs), {**inputs, "labels": labels})
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, student.parameters()))
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for key, value in teacher_before.items():
        torch.testing.assert_close(teacher.state_dict()[key], value)


def test_gpt2_projection_layout_is_not_silently_supported():
    hf = pytest.importorskip("transformers")
    model = hf.GPT2LMHeadModel(
        hf.GPT2Config(vocab_size=32, n_embd=16, n_head=2, n_layer=1)
    )
    with pytest.raises(ValueError, match="no supported layers"):
        PostTrainingQuantization(model, QuantizationConfig(mode="ptq")).prepare()
