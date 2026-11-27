# -*- coding: utf-8 -*-
# @Time : 2026/11/26 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Held-out corpus contracts and finite, independent evaluation cursors."""

import json
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from parascale.data.text import (
    TokenBatchCursor,
    TokenCorpus,
    prepare_token_corpus,
)
from parascale.workloads.specs import causal_lm
from tests.test_text_corpus_no_torch import Tokenizer


def evaluation_spec(**overrides):
    return causal_lm.CausalLmEvaluationSpec.from_config(
        {"evaluation": {"data_path": "validation", "max_batches": 2, **overrides}}
    )


def build(spec, training, **overrides):
    from parascale.workloads.causal_lm_evaluation import (
        build_causal_lm_evaluation_cursor,
    )

    return build_causal_lm_evaluation_cursor(
        spec, training, **{"microbatches": 2, "vocab_size": 8, **overrides}
    )


def prepare(tmp_path, name, text, tokenizer=None):
    source = tmp_path / f"{name}.txt"
    source.write_text(text, encoding="utf-8")
    path = tmp_path / name
    prepare_token_corpus(source, path, tokenizer or Tokenizer())
    return path


@pytest.fixture
def corpora(tmp_path):
    train = prepare(tmp_path, "train", "1 2 3 4 5 6\n" * 25)
    valid = prepare(tmp_path, "valid", "6 5 4 3 2 1\n" * 25)
    return train, valid


def training_cursor(path, **overrides):
    return TokenBatchCursor(
        TokenCorpus(path),
        **{
            "batch_size": 2,
            "sequence_length": 4,
            "rank": 1,
            "world_size": 2,
            "seed": 29,
            "shuffle_window_blocks": 19,
            **overrides,
        },
    )


def test_spec_absence_disables_and_valid_mapping_is_frozen():
    assert causal_lm.CausalLmEvaluationSpec.from_config({}) is None
    spec = causal_lm.CausalLmEvaluationSpec.from_config(
        {
            "evaluation": MappingProxyType(
                {"data_path": "E:/data/valid", "max_batches": 2}
            )
        }
    )
    assert spec.data_path == "E:/data/valid"
    assert spec.max_batches == 2
    with pytest.raises(FrozenInstanceError):
        spec.max_batches = 3


@pytest.mark.parametrize(
    "section",
    [
        None,
        {},
        [],
        "validation",
        False,
        {"data_path": "validation"},
        {"max_batches": 1},
        {"data_path": "validation", "max_batches": 1, "enabled": True},
    ],
)
def test_spec_rejects_invalid_sections(section):
    with pytest.raises(ValueError, match="evaluation"):
        causal_lm.CausalLmEvaluationSpec.from_config({"evaluation": section})


@pytest.mark.parametrize("path", [None, "", "  ", 1, True, "s3://data"])
def test_spec_rejects_invalid_paths(path):
    with pytest.raises(ValueError, match="data_path"):
        evaluation_spec(data_path=path)


@pytest.mark.parametrize("budget", [None, 0, -1, True, 1.5, "2"])
def test_spec_rejects_invalid_budgets(budget):
    with pytest.raises(ValueError, match="max_batches"):
        evaluation_spec(max_batches=budget)


def test_builder_preserves_training_and_settings(corpora):
    train, valid = corpora
    training = training_cursor(train, pin_memory=True)
    training.position = 3
    before = training.state_dict()
    cursor = build(evaluation_spec(data_path=str(valid)), training)
    assert training.state_dict() == before
    assert cursor is not training
    assert cursor.position == 0
    for name in (
        "batch_size",
        "sequence_length",
        "rank",
        "world_size",
        "seed",
        "pin_memory",
        "window_blocks",
    ):
        assert getattr(cursor, name) == getattr(training, name)
    assert cursor.identity["shuffle_window_blocks"] == 19
    assert cursor.shuffle is False
    assert cursor.repeat is False
    assert cursor.corpus.manifest["tokenizer"] == training.corpus.manifest["tokenizer"]
    assert cursor.corpus.manifest["eos_token_id"] == 7


@pytest.mark.parametrize("world", [1, 2, 4])
def test_finite_disjoint_drop_tail_and_fresh_restart(corpora, world):
    pytest.importorskip("torch")
    train, valid = corpora
    spec = evaluation_spec(data_path=str(valid), max_batches=1)
    all_blocks = []
    for rank in range(world):
        training = training_cursor(train, rank=rank, world_size=world)
        next(iter(training))
        before = training.state_dict()
        cursor = build(spec, training)
        batches = list(cursor)
        assert len(batches) == 43 // world // 2
        assert list(cursor) == []
        blocks = [block for batch in batches for block in batch["block_ids"]]
        assert blocks == sorted(blocks)
        all_blocks.extend(blocks)
        fresh = build(spec, training)
        assert next(iter(fresh))["block_ids"] == batches[0]["block_ids"]
        assert training.state_dict() == before
    usable = 43 // (world * 2) * world * 2
    assert len(all_blocks) == len(set(all_blocks)) == usable
    assert set(all_blocks) == set(range(usable))


@pytest.mark.parametrize("budget, microbatches", [(5, 2), (2, 5)])
def test_exact_full_window_capacity_is_accepted(corpora, budget, microbatches):
    train, valid = corpora
    cursor = build(
        evaluation_spec(data_path=str(valid), max_batches=budget),
        training_cursor(train),
        microbatches=microbatches,
    )
    assert cursor.batches_per_epoch == budget * microbatches


def test_insufficient_full_windows_rejected_without_training_mutation(corpora):
    train, valid = corpora
    training = training_cursor(train)
    before = training.state_dict()
    with pytest.raises(ValueError, match="max_batches|windows"):
        build(
            evaluation_spec(data_path=str(valid), max_batches=4),
            training,
            microbatches=3,
        )
    assert training.state_dict() == before


@pytest.mark.parametrize("name", ["microbatches", "vocab_size"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2", None])
def test_invalid_builder_arguments(corpora, name, value):
    train, valid = corpora
    with pytest.raises(ValueError, match=name):
        build(
            evaluation_spec(data_path=str(valid)),
            training_cursor(train),
            **{name: value},
        )


def test_model_vocabulary_checked(corpora):
    train, valid = corpora
    with pytest.raises(ValueError, match="vocabulary"):
        build(
            evaluation_spec(data_path=str(valid)),
            training_cursor(train),
            vocab_size=7,
        )


@pytest.mark.parametrize("kind", ["same_path", "same_payload", "same_source"])
def test_training_identity_rejected(tmp_path, corpora, kind):
    train, valid = corpora
    if kind == "same_path":
        valid = train
    elif kind == "same_payload":
        valid = prepare(tmp_path, "copy", "  1 2 3 4 5 6  \n" * 25)
    else:
        manifest_path = valid / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_sha256"] = TokenCorpus(train).manifest["source_sha256"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="training|held-out"):
        build(evaluation_spec(data_path=str(valid)), training_cursor(train))


@pytest.mark.parametrize("fault", ["tokenizer", "eos", "document_policy"])
def test_same_tokenizer_and_eos_policy_required(tmp_path, corpora, fault):
    train, valid = corpora
    if fault == "eos":
        tokenizer = Tokenizer()
        tokenizer.eos_token_id = 0
        valid = prepare(tmp_path, "other_eos", "6 5 4 3 2 1\n" * 25, tokenizer)
    else:
        manifest_path = valid / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[fault] = "different"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer|policy"):
        build(evaluation_spec(data_path=str(valid)), training_cursor(train))


def test_optional_source_hash_absence_is_not_overlap(corpora):
    train, valid = corpora
    for path in (train, valid):
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["source_sha256"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert build(evaluation_spec(data_path=str(valid)), training_cursor(train))
