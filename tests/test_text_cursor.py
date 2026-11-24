# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Deterministic token batches, DP ownership and bounded cursor resume."""

import pytest

torch = pytest.importorskip("torch")

from parascale.data.text import (  # noqa: E402
    TokenBatchCursor,
    TokenCorpus,
    prepare_token_corpus,
)
from tests.test_text_corpus_no_torch import Tokenizer  # noqa: E402


def corpus(tmp_path):
    source = tmp_path / "data.txt"
    source.write_text("1 2 3 4 5 6\n" * 25, encoding="utf-8")
    prepare_token_corpus(source, tmp_path / "corpus", Tokenizer())
    return TokenCorpus(tmp_path / "corpus")


@pytest.mark.parametrize("world", [1, 2, 4])
def test_dp_batches_have_disjoint_block_ownership_and_equal_counts(tmp_path, world):
    data = corpus(tmp_path)
    rank_blocks = []
    counts = []
    for rank in range(world):
        loader = TokenBatchCursor(
            data,
            batch_size=2,
            sequence_length=4,
            rank=rank,
            world_size=world,
            seed=29,
            shuffle=True,
            repeat=False,
            shuffle_window_blocks=16,
        )
        batches = list(loader)
        counts.append(len(batches))
        rank_blocks.append({index for batch in batches for index in batch["block_ids"]})
        assert all(batch["input_ids"].shape == (2, 4) for batch in batches)
        assert all(batch["tokens"] == 6 for batch in batches)
    assert len(set(counts)) == 1
    assert sum(map(len, rank_blocks)) == len(set.union(*rank_blocks))


def test_resume_across_epochs_reproduces_shuffled_token_blocks(tmp_path):
    data = corpus(tmp_path)
    kwargs = dict(
        batch_size=2,
        sequence_length=4,
        rank=1,
        world_size=2,
        seed=27,
        shuffle_window_blocks=8,
    )
    loader = TokenBatchCursor(data, **kwargs)
    iterator = iter(loader)
    for _ in range(13):
        next(iterator)
    state = loader.state_dict()
    expected = [next(iterator) for _ in range(6)]
    resumed = TokenBatchCursor(data, **kwargs)
    resumed.load_state_dict(state)
    for actual, reference in zip(iter(resumed), expected):
        assert actual["block_ids"] == reference["block_ids"]
        assert torch.equal(actual["input_ids"], reference["input_ids"])
    assert set(state) == {"identity", "position"}


def test_resume_rejects_changed_batch_topology_seed_or_invalid_position(tmp_path):
    data = corpus(tmp_path)
    state = TokenBatchCursor(data, batch_size=2, sequence_length=4).state_dict()
    for change in (
        {"batch_size": 1},
        {"sequence_length": 8},
        {"world_size": 2},
        {"seed": 17},
        {"shuffle": False},
    ):
        options = {"batch_size": 2, "sequence_length": 4, **change}
        with pytest.raises(ValueError, match="identity"):
            TokenBatchCursor(data, **options).load_state_dict(state)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="position"):
            TokenBatchCursor(data, batch_size=2, sequence_length=4).load_state_dict(
                {**state, "position": invalid}
            )


def test_small_corpus_fails_before_distributed_training_can_diverge(tmp_path):
    with pytest.raises(ValueError, match="full batch"):
        TokenBatchCursor(
            corpus(tmp_path), batch_size=16, sequence_length=128, world_size=4
        )


def test_sampler_allocations_are_bounded_by_shuffle_window(tmp_path, monkeypatch):
    from torch.utils.data.distributed import DistributedSampler

    original = DistributedSampler.__iter__
    lengths = []

    def observe(sampler):
        lengths.append(len(sampler.dataset))
        return original(sampler)

    monkeypatch.setattr(DistributedSampler, "__iter__", observe)
    loader = TokenBatchCursor(
        corpus(tmp_path),
        batch_size=2,
        sequence_length=4,
        world_size=2,
        shuffle_window_blocks=8,
        repeat=False,
    )
    list(loader)
    assert lengths and max(lengths) <= 8


@pytest.mark.parametrize("shuffle", [False, True])
def test_dp_window_boundaries_and_short_tail_resume_without_loss(tmp_path, shuffle):
    data = corpus(tmp_path)
    rank_blocks = []
    for rank in range(2):
        options = dict(
            batch_size=2,
            sequence_length=4,
            world_size=2,
            rank=rank,
            shuffle=shuffle,
            shuffle_window_blocks=16,
            repeat=False,
        )
        loader = TokenBatchCursor(data, **options)
        expected = list(loader)
        # 43 blocks become 40 usable: two full windows and one eight-block tail.
        assert len(expected) == 10
        blocks = [block for batch in expected for block in batch["block_ids"]]
        assert len(blocks) == len(set(blocks)) == 20
        rank_blocks.append(set(blocks))
        for position in (3, 4, 5, 7, 8, 9, 10):
            source = TokenBatchCursor(data, **options)
            iterator = iter(source)
            for _ in range(position):
                next(iterator)
            restored = TokenBatchCursor(data, **options)
            restored.load_state_dict(source.state_dict())
            actual = list(restored)
            assert len(actual) == len(expected) - position
            for batch, reference in zip(actual, expected[position:]):
                assert batch["block_ids"] == reference["block_ids"]
                assert torch.equal(batch["input_ids"], reference["input_ids"])
    assert rank_blocks[0].isdisjoint(rank_blocks[1])
    assert set.union(*rank_blocks) == set(range(40))
