# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from parascale.distillation import (  # noqa: E402
    DistillationConfig,
    OnlineDistillationLoss,
    distillation_loss,
)


def test_analytic_kl_temperature_and_reduction():
    s = torch.tensor([[1.0, -2.0, 0.0], [0.0, 1.0, 2.0]], requires_grad=True)
    t = torch.tensor([[0.0, 1.0, 3.0], [2.0, -1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 2])
    config = DistillationConfig(temperature=3.0, alpha=0.7)
    result = distillation_loss(s, t, labels=labels, config=config)
    p = (t.detach() / 3).softmax(-1)
    expected = (p * (p.log() - (s / 3).log_softmax(-1))).sum(-1).mean() * 9
    torch.testing.assert_close(result.soft_loss, expected)
    torch.testing.assert_close(
        result.total,
        0.3 * torch.nn.functional.cross_entropy(s, labels) + 0.7 * expected,
    )
    assert result.valid_count == 2
    result.total.backward()
    assert t.grad is None and s.grad is not None


def test_causal_shift_padding_ignore_and_gradients():
    s = torch.randn(2, 5, 4, requires_grad=True)
    t = torch.randn_like(s)
    labels = torch.tensor([[-100, 1, 2, 3, -100], [0, 1, -100, 2, 3]])
    mask = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    config = DistillationConfig(task="causal_lm")
    result = distillation_loss(s, t, labels=labels, attention_mask=mask, config=config)
    valid = torch.tensor([[False, True, True, False], [True, False, True, False]])
    reference = distillation_loss(
        s[:, :-1][valid], t[:, :-1][valid], labels=labels[:, 1:][valid]
    )
    torch.testing.assert_close(result.total, reference.total)
    assert result.valid_count == 4
    result.total.backward()
    assert torch.count_nonzero(s.grad[:, :-1][~valid]) == 0
    assert torch.count_nonzero(s.grad[:, -1]) == 0


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_endpoints_and_identical_extreme_logits(alpha):
    s = torch.tensor([[10000.0, -10000.0]], dtype=torch.float16, requires_grad=True)
    result = distillation_loss(
        s,
        s.detach(),
        labels=torch.tensor([0]) if alpha == 0 else None,
        config=DistillationConfig(alpha=alpha),
    )
    assert result.total.dtype == torch.float32
    assert result.total.item() == 0
    result.total.backward()
    assert torch.isfinite(s.grad).all()


@pytest.mark.parametrize(
    "case",
    [
        "shape",
        "nan",
        "integer",
        "labels_float",
        "labels_bool",
        "labels_range",
        "labels_shape",
        "no_labels",
        "empty",
        "mask",
        "all_ignored",
    ],
)
def test_invalid_inputs(case):
    s, t = torch.randn(2, 3), torch.randn(2, 3)
    labels, mask = torch.tensor([0, 1]), None
    if case == "shape":
        t = torch.randn(2, 4)
    if case == "nan":
        t[0, 0] = float("nan")
    if case == "integer":
        s = s.long()
    if case == "labels_float":
        labels = labels.float()
    if case == "labels_bool":
        labels = labels.bool()
    if case == "labels_range":
        labels[0] = 3
    if case == "labels_shape":
        labels = labels[:, None]
    if case == "no_labels":
        labels = None
    if case == "empty":
        s, t, labels = s[:0], t[:0], labels[:0]
    if case == "mask":
        mask = torch.tensor([0, 2])
    if case == "all_ignored":
        labels[:] = -100
    with pytest.raises((ValueError, TypeError)):
        distillation_loss(s, t, labels=labels, attention_mask=mask)


def test_label_free_causal_shift_and_classification_mask():
    s, t = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    valid = mask[:, :-1].bool() & mask[:, 1:].bool()
    result = distillation_loss(
        s,
        t,
        attention_mask=mask,
        config=DistillationConfig(task="causal_lm", alpha=1),
    )
    expected = distillation_loss(
        s[:, :-1][valid],
        t[:, :-1][valid],
        config=DistillationConfig(alpha=1),
    )
    torch.testing.assert_close(result.total, expected.total)
    assert result.valid_count == 3
    masked = distillation_loss(
        s[0], t[0], attention_mask=mask[0], config=DistillationConfig(alpha=1)
    )
    reference = distillation_loss(
        s[0, 1:], t[0, 1:], config=DistillationConfig(alpha=1)
    )
    torch.testing.assert_close(masked.total, reference.total)


@pytest.mark.parametrize("length", [0, 1])
def test_causal_empty_sequences(length):
    logits = torch.empty(2, length, 3)
    with pytest.raises(ValueError, match="nonempty|length"):
        distillation_loss(
            logits, logits, config=DistillationConfig(task="causal_lm", alpha=1)
        )


@pytest.mark.parametrize("alpha", [0, 1])
def test_endpoint_skips_unused_math(monkeypatch, alpha):
    def unused(*args, **kwargs):
        raise AssertionError("unused endpoint computation")

    if alpha == 1:
        monkeypatch.setattr(torch.nn.functional, "cross_entropy", unused)
    else:
        monkeypatch.setattr(torch.nn.functional, "log_softmax", unused)
    result = distillation_loss(
        torch.randn(2, 3),
        torch.randn(2, 3),
        labels=torch.tensor([0, 2]),
        config=DistillationConfig(alpha=alpha),
    )
    assert torch.isfinite(result.total)
    assert (result.hard_loss if alpha == 1 else result.soft_loss).item() == 0


class Toy(torch.nn.Module):
    def __init__(self, kind="tensor"):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.dropout = torch.nn.Dropout(0.9)
        self.register_buffer("counter", torch.tensor(0.0))
        self.kind = kind

    def forward(self, pixel_values):
        assert not self.training or torch.is_grad_enabled()
        logits = self.linear(self.dropout(pixel_values))
        if self.kind == "dict":
            return {"logits": logits}
        if self.kind == "object":
            return SimpleNamespace(logits=logits)
        return logits


@pytest.mark.parametrize("kind", ["tensor", "dict", "object"])
def test_online_teacher_frozen_deterministic_student_updates(kind):
    student, teacher = Toy(kind), Toy(kind)
    for p in teacher.parameters():
        p.grad = torch.ones_like(p)
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    keys = set(student.state_dict())
    loss_fn = OnlineDistillationLoss(student, teacher, DistillationConfig())
    batch = {
        "pixel_values": torch.randn(4, 3),
        "labels": torch.tensor([0, 1, 1, 0]),
        "metadata": "omit",
    }
    student.eval()
    output = student(batch["pixel_values"])
    teacher.train()
    loss = loss_fn(output, batch)
    torch.testing.assert_close(loss, loss_fn(output, batch))
    assert not teacher.training
    old = student.linear.weight.detach().clone()
    loss.backward()
    torch.optim.SGD(student.parameters(), lr=0.1).step()
    assert not torch.equal(old, student.linear.weight)
    assert set(student.state_dict()) == keys
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert set(loss_fn.last_metrics) == {
        "hard_loss",
        "soft_loss",
        "total",
        "valid_count",
    }
    assert all(isinstance(x, (float, int)) for x in loss_fn.last_metrics.values())


@pytest.mark.parametrize("alias", ["identity", "parameter", "storage"])
def test_reject_alias_before_freeze(alias):
    student, teacher = Toy(), Toy()
    if alias == "identity":
        teacher = student
    if alias == "parameter":
        teacher.linear.weight = student.linear.weight
    if alias == "storage":
        teacher.linear.weight = torch.nn.Parameter(student.linear.weight.detach())
    with pytest.raises(ValueError, match="shar|alias"):
        OnlineDistillationLoss(student, teacher, DistillationConfig())
    assert all(p.requires_grad for p in student.parameters())


def test_online_clear_batch_and_output_errors():
    fn = OnlineDistillationLoss(Toy(), Toy(), DistillationConfig())
    for output, batch in [
        (torch.randn(2, 2), []),
        (torch.randn(2, 2), {"labels": torch.tensor([0, 1])}),
        ({"loss": 1}, {"pixel_values": torch.randn(2, 3)}),
    ]:
        with pytest.raises((ValueError, TypeError)):
            fn(output, batch)


def test_online_snapshots_caller_config():
    student, teacher = Toy(), Toy()
    config = DistillationConfig(alpha=0.3, temperature=3)
    fn = OnlineDistillationLoss(student, teacher, config)
    batch = {"pixel_values": torch.randn(2, 3), "labels": torch.tensor([0, 1])}
    output = student(batch["pixel_values"])
    before = fn(output, batch)
    config.alpha = 1
    config.temperature = 7
    config.input_keys = ["not_an_input"]
    torch.testing.assert_close(fn(output, batch), before)
    assert fn.config is not config
    assert fn.config.alpha == 0.3 and fn.config.temperature == 3


@pytest.mark.parametrize(
    "alpha,temperature", [(0, 1), (0.5, 1), (1, 1), (1, 2), (1, 1e-18)]
)
def test_reject_nonfinite_fp32_loss_computation(alpha, temperature):
    s = torch.tensor([[3e38, -3e38]], requires_grad=True)
    t = -s.detach()
    with pytest.raises(ValueError, match="finite.*FP32|FP32.*finite"):
        distillation_loss(
            s,
            t,
            labels=torch.tensor([1]),
            config=DistillationConfig(alpha=alpha, temperature=temperature),
        )
    assert s.grad is None


@pytest.mark.parametrize("temperature", [1.1e-19, 1.8e19])
def test_supported_temperature_boundaries_have_finite_backward(temperature):
    s = torch.tensor([[1.0, -1.0]], requires_grad=True)
    t = torch.tensor([[-1.0, 1.0]])
    result = distillation_loss(
        s, t, config=DistillationConfig(alpha=1, temperature=temperature)
    )
    assert torch.isfinite(result.total)
    result.total.backward()
    assert torch.isfinite(s.grad).all()


@pytest.mark.parametrize("sample_mask", [None, [1, 0]])
def test_online_classification_token_attention_mask(sample_mask):
    class TextClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.seen_mask = None

        def forward(self, input_ids, attention_mask):
            self.seen_mask = attention_mask
            return self.linear(input_ids.float() * attention_mask)

    student, teacher = TextClassifier(), TextClassifier()
    config = DistillationConfig()
    fn = OnlineDistillationLoss(student, teacher, config)
    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "labels": torch.tensor([0, 1]),
    }
    if sample_mask is not None:
        batch["loss_mask"] = torch.tensor(sample_mask)
    output = student(batch["input_ids"], batch["attention_mask"])
    output.retain_grad()
    result = fn(output, batch)
    assert teacher.seen_mask is batch["attention_mask"]
    with torch.no_grad():
        expected_teacher = teacher(batch["input_ids"], batch["attention_mask"])
    expected = distillation_loss(
        output,
        expected_teacher,
        labels=batch["labels"],
        attention_mask=batch.get("loss_mask"),
        config=config,
    )
    torch.testing.assert_close(result, expected.total)
    assert fn.last_metrics["valid_count"] == (2 if sample_mask is None else 1)
    result.backward()
    if sample_mask is not None:
        assert torch.count_nonzero(output.grad[1]) == 0
