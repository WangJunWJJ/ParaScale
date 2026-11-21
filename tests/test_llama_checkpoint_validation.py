# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午1:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Fingerprint controls for the fresh-process Llama checkpoint validator."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


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
