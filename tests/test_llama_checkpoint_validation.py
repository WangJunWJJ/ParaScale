# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午1:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Fingerprint controls for the fresh-process Llama checkpoint validator."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_stage_recovery_catalog_validation_is_explicit(monkeypatch):
    from tests.benchmarks.scripts.validate_stage_recovery import parse_args

    argv = [
        "validate",
        "--mode",
        "baseline",
        "--model-path",
        "model",
        "--corpus-path",
        "corpus",
        "--checkpoint-dir",
        "checkpoint",
        "--output-dir",
        "output",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert parse_args().validate_catalog is False
    monkeypatch.setattr(sys, "argv", [*argv, "--validate-catalog"])
    assert parse_args().validate_catalog is True


def test_state_fingerprint_tracks_values_layout_and_type():
    from tests.benchmarks.scripts.validate_llama_checkpoint import state_fingerprint

    value = torch.arange(4, dtype=torch.float32)
    original = state_fingerprint({"weight": value, "step": 1})
    assert original == state_fingerprint({"step": 1, "weight": value.clone()})
    for changed in (value + 1, value.reshape(2, 2), value.double()):
        assert original != state_fingerprint({"weight": changed, "step": 1})
    assert original != state_fingerprint({"weight": value, "step": True})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_state_fingerprint_rejects_nonfinite_tensors(value):
    from tests.benchmarks.scripts.validate_llama_checkpoint import state_fingerprint

    with pytest.raises(ValueError, match="nonfinite"):
        state_fingerprint({"weight": torch.tensor([value])})


@pytest.mark.parametrize(
    "damage", ["none", "no_op", "nan", "missing", "stale_step", "unchanged"]
)
def test_stage_recovery_requires_real_finite_optimizer_updates(damage):
    from tests.benchmarks.scripts.validate_llama_checkpoint import state_fingerprint
    from tests.benchmarks.scripts.validate_stage_recovery import training_fingerprint

    for _ in range(2):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        initial = state_fingerprint(dict(model.named_parameters()))
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 2)).square().sum().backward()
            if damage != "no_op":
                optimizer.step()
        first = next(model.parameters())
        if damage == "nan":
            optimizer.state[first]["exp_avg"].fill_(float("nan"))
        elif damage == "missing":
            optimizer.state[first].pop("exp_avg")
        elif damage == "stale_step":
            optimizer.state[first]["step"].fill_(2)
        elif damage == "unchanged":
            initial = state_fingerprint(dict(model.named_parameters()))
        if damage == "none":
            assert training_fingerprint(model, optimizer, steps=3, initial=initial)
        else:
            with pytest.raises((ValueError, AssertionError)):
                training_fingerprint(model, optimizer, steps=3, initial=initial)


def test_full_hf_configuration_uses_its_json_contract():
    transformers = pytest.importorskip("transformers")
    from parascale.parallel.checkpoint_context import canonical_checkpoint_context
    from tests.benchmarks.scripts.validate_llama_checkpoint import model_configuration

    config = transformers.LlamaConfig(id2label={0: "zero", 1: "one"})
    actual = model_configuration(config)
    assert actual["id2label"] == {"0": "zero", "1": "one"}
    assert actual["hidden_size"] == config.hidden_size
    canonical_checkpoint_context(
        dict(
            model=actual, loss="causal", data="test", tokenizer="test", precision="fp32"
        )
    )


def test_rng_only_zero_step_restore_cannot_pass_continuation():
    from tests.benchmarks.scripts.validate_llama_checkpoint import (
        load_resume_checkpoint,
    )

    with torch.random.fork_rng(devices=[]):
        saved = torch.get_rng_state().clone()

        def broken_loader(directory, optimizer):
            torch.set_rng_state(saved)
            return 0

        trainer = SimpleNamespace(load_checkpoint=broken_loader)
        with pytest.raises(AssertionError, match="step"):
            load_resume_checkpoint(trainer, "unused", None, expected_step=2, steps=4)


def test_complete_real_model_context_accepts_pinned_library_metadata():
    transformers = pytest.importorskip("transformers")
    from parascale.parallel.checkpoint_context import canonical_checkpoint_context
    from tests.benchmarks.scripts.validate_llama_checkpoint import build_context

    value = build_context(
        transformers.LlamaConfig(),
        {"tokenizer.json": "test-digest"},
        b"text",
        [1, 2, 3, 4],
        global_batch=2,
        sequence_length=2,
    )
    canonical_checkpoint_context(value)
    assert type(value["model"]["torch"]) is str
    assert value["model"]["config"]["id2label"] == {"0": "LABEL_0", "1": "LABEL_1"}


def test_checkpoint_validator_declares_actual_execution_precision():
    from transformers import LlamaConfig

    from tests.benchmarks.scripts.validate_llama_checkpoint import build_context

    value = build_context(
        LlamaConfig(),
        {"tokenizer.json": "test"},
        b"text",
        [1, 2],
        global_batch=1,
        sequence_length=2,
        precision="bf16",
    )
    assert value["precision"] == "bf16"


@pytest.mark.parametrize("mode", ["baseline", "resume"])
def test_recovery_evaluation_is_explicit_and_resume_only(monkeypatch, mode):
    from tests.benchmarks.scripts.validate_stage_recovery import parse_args

    argv = [
        "validate",
        "--mode",
        mode,
        "--model-path",
        "model",
        "--corpus-path",
        "train",
        "--checkpoint-dir",
        "checkpoint",
        "--output-dir",
        "output",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert parse_args().evaluation_corpus is None
    monkeypatch.setattr(sys, "argv", [*argv, "--evaluation-corpus", "validation"])
    if mode == "baseline":
        with pytest.raises(SystemExit):
            parse_args()
    else:
        args = parse_args()
        assert args.evaluation_corpus == Path("validation")
        assert args.evaluation_max_batches == 1
        for invalid in ("0", "-1"):
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    *argv,
                    "--evaluation-corpus",
                    "validation",
                    "--evaluation-max-batches",
                    invalid,
                ],
            )
            with pytest.raises(SystemExit):
                parse_args()


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "parameter",
        "buffer",
        "optimizer",
        "gradient",
        "mode",
        "rng",
        "train_cursor",
        "validation_cursor",
        "runtime",
    ],
)
def test_recovery_evaluation_rejects_state_mutation(monkeypatch, damage):
    from parascale.runtime.training import stage_evaluation
    from tests.benchmarks.scripts.validate_stage_recovery import evaluate_restored_state

    model = torch.nn.Linear(2, 2)
    model.register_buffer("counter", torch.tensor(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    trainer = SimpleNamespace(
        module=model, execution_module=model, device=torch.device("cpu")
    )
    positions = {"train_cursor": 4, "validation_cursor": 0, "runtime": 2}
    training = SimpleNamespace(
        state_dict=lambda: {"position": positions["train_cursor"]}
    )
    validation = SimpleNamespace(
        state_dict=lambda: {"position": positions["validation_cursor"]}
    )
    runtime = SimpleNamespace(
        dataloader=training,
        engine=SimpleNamespace(state=SimpleNamespace(global_step=2)),
        capture=lambda: {"step": positions["runtime"]},
    )
    calls = []

    def evaluate(actual, cursor, *, max_batches, global_step):
        calls.append((actual, cursor, max_batches, global_step))
        first = next(model.parameters())
        with torch.no_grad():
            if damage == "parameter":
                first.add_(1)
            elif damage == "buffer":
                model.counter.add_(1)
            elif damage == "optimizer":
                optimizer.state[first]["exp_avg"].add_(1)
            elif damage == "gradient":
                first.grad.add_(1)
            elif damage == "mode":
                model.eval()
            elif damage == "rng":
                torch.rand(())
            elif damage in positions:
                positions[damage] += 1
        return {"global_step": global_step, "scored_tokens": 12, "nll": 1.5}

    monkeypatch.setattr(stage_evaluation, "evaluate_stage", evaluate)
    with torch.random.fork_rng(devices=[]):
        if damage == "none":
            result = evaluate_restored_state(
                trainer, optimizer, runtime, validation, max_batches=1
            )
            assert result["state_before"] == result["state_after"]
            assert result["evaluation"]["scored_tokens"] == 12
        else:
            with pytest.raises(
                (AssertionError, RuntimeError), match="evaluation changed"
            ):
                evaluate_restored_state(
                    trainer, optimizer, runtime, validation, max_batches=1
                )
    assert calls == [(trainer, validation, 1, 2)]
