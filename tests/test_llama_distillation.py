# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Offline verification of Llama online distillation and its evidence report."""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def validator():
    path = Path(__file__).parent / "benchmarks/scripts/validate_llama_distillation.py"
    assert path.is_file(), "Llama distillation validator is not implemented"
    spec = importlib.util.spec_from_file_location("llama_distillation_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_report_and_student_roundtrip(validator, tmp_path):
    args = validator.parser().parse_args(
        [
            "--tiny",
            "--seqlen",
            "8",
            "--batches",
            "2",
            "--steps",
            "2",
            "--output",
            str(tmp_path),
        ]
    )
    report = validator.run(args)
    assert report["passed"]
    assert report["parameter_counts"]["teacher"] > report["parameter_counts"]["student"]
    assert report["teacher_state"]["before"] == report["teacher_state"]["after"]
    assert report["data"]["train_batch_hashes"] != report["data"]["eval_batch_hashes"]
    for name in ("distillation", "ce_baseline"):
        result = report[name]
        assert result["training"]["nonzero_gradient_steps"] == 2
        assert result["training"]["changed_parameters"] > 0
        assert result["after"]["valid_count"] == 14
        assert result["after"]["ce"] > 0
        assert result["after"]["kl_teacher_to_student"] >= -1e-6
    assert report["distillation"]["before"] == report["ce_baseline"]["before"]
    assert report["student_export"]["max_abs_error"] <= 1e-6
    assert (tmp_path / "student_state_dict.pt").is_file()
    assert json.loads((tmp_path / "report.json").read_text())["passed"]


def test_tokenizer_mapping_not_just_size(validator):
    class Tokenizer:
        special_tokens_map = {"eos_token": "a"}

        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    with pytest.raises(ValueError, match="mapping"):
        validator.check_tokenizers(
            Tokenizer({"a": 0, "b": 1}), Tokenizer({"a": 1, "b": 0}), 2
        )


def test_pretrained_requires_separate_text(validator, tmp_path):
    args = validator.parser().parse_args(
        [
            "--teacher-model",
            "unused",
            "--student-model",
            "unused",
            "--output",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError, match="calibration.*evaluation"):
        validator.run(args)


def test_loss_matches_shifted_masked_fp32_reference():
    from parascale.distillation import DistillationConfig, distillation_loss

    torch.manual_seed(19)
    student = torch.randn(2, 5, 7, requires_grad=True)
    teacher = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[0, 1, -100, 3, 4], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    config = DistillationConfig(
        enabled=True, task="causal_lm", temperature=2.0, alpha=0.5
    )
    result = distillation_loss(
        student, teacher, labels=labels, attention_mask=mask, config=config
    )
    valid = (labels[:, 1:] != -100) & mask[:, 1:].bool()
    s, t = student[:, :-1][valid], teacher[:, :-1][valid]
    hard = torch.nn.functional.cross_entropy(s, labels[:, 1:][valid])
    soft = (
        torch.nn.functional.kl_div(
            (s / 2).log_softmax(-1), (t / 2).softmax(-1), reduction="batchmean"
        )
        * 4
    )
    torch.testing.assert_close(result.hard_loss, hard)
    torch.testing.assert_close(result.soft_loss, soft)
    torch.testing.assert_close(result.total, 0.5 * hard + 0.5 * soft)
    assert int(result.valid_count) == int(valid.sum())
    result.total.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_online_llama_teacher_is_external_and_no_grad():
    from transformers import LlamaConfig, LlamaForCausalLM

    from parascale.distillation import DistillationConfig, OnlineDistillationLoss

    def model(hidden):
        return LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=hidden,
                intermediate_size=hidden * 2,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                use_cache=False,
            )
        )

    student, teacher = model(32), model(64)
    keys = set(student.state_dict())
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    calls = []
    handle = teacher.register_forward_pre_hook(
        lambda module, inputs: calls.append((torch.is_grad_enabled(), module.training))
    )
    objective = OnlineDistillationLoss(
        student,
        teacher,
        DistillationConfig(
            enabled=True,
            task="causal_lm",
            input_keys=["input_ids", "attention_mask"],
        ),
    )
    ids = torch.arange(8).reshape(1, -1)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    loss = objective(student(input_ids=ids), batch)
    loss.backward()
    optimizer.step()
    handle.remove()
    assert calls == [(False, False)]
    assert set(student.state_dict()) == keys
    assert not any(
        id(t) == id(s) for t in teacher.parameters() for s in student.parameters()
    )
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(
        torch.equal(v, teacher_before[k]) for k, v in teacher.state_dict().items()
    )
    assert set(objective.last_metrics) == {
        "total",
        "hard_loss",
        "soft_loss",
        "valid_count",
    }


def test_reproducible_tiny_and_existing_output_rejected(validator, tmp_path):
    reports = []
    for name in ("first", "second"):
        args = validator.parser().parse_args(
            [
                "--tiny",
                "--seqlen",
                "4",
                "--batches",
                "1",
                "--steps",
                "1",
                "--output",
                str(tmp_path / name),
            ]
        )
        reports.append(validator.run(args))
    for key in ("student_initial_sha256", "data", "teacher_state"):
        assert reports[0][key] == reports[1][key]
    for name in ("distillation", "ce_baseline"):
        assert reports[0][name]["after"] == reports[1][name]["after"]
    with pytest.raises(ValueError, match="empty"):
        validator.run(args)


@pytest.mark.parametrize(
    "option,value",
    [("--steps", "0"), ("--seqlen", "1"), ("--temperature", "nan"), ("--alpha", "2")],
)
def test_invalid_bounds_rejected(validator, tmp_path, option, value):
    args = validator.parser().parse_args(
        [
            "--tiny",
            option,
            value,
            "--output",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError):
        validator.run(args)


def test_evaluation_requires_unmasked_source_and_target(validator):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_cache=False,
    )
    student = LlamaForCausalLM(config).eval()
    teacher = LlamaForCausalLM(config).eval()
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 1, 1, 0, 1, 1]])
    batch = {"input_ids": ids, "attention_mask": mask}
    result = validator.evaluate(student, teacher, [batch], torch.device("cpu"))
    assert result["valid_count"] == 2
    valid = mask[:, :-1].bool() & mask[:, 1:].bool()
    with torch.no_grad():
        s = student(**batch).logits[:, :-1].float()[valid]
        t = teacher(**batch).logits[:, :-1].float()[valid]
        ce = torch.nn.functional.cross_entropy(s, ids[:, 1:][valid])
        kl = torch.nn.functional.kl_div(
            s.log_softmax(-1), t.softmax(-1), reduction="batchmean"
        )
    assert result["ce"] == pytest.approx(ce.item())
    assert result["kl_teacher_to_student"] == pytest.approx(kl.item())
