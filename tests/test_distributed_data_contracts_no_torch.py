# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午12:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Distributed batching contracts without requiring torch."""

import builtins
import random
from pathlib import Path

import pytest

from parascale.data.sampler import DistributedTokenBudgetBatchSampler
from parascale.workloads import ground_dino
from parascale.workloads.specs.ground_dino import GroundDinoSpec


def _sampler(lengths, rank=0, world_size=1, **kwargs):
    return DistributedTokenBudgetBatchSampler(
        lengths,
        10,
        rank=rank,
        world_size=world_size,
        length_fn=lambda value: value,
        **kwargs,
    )


@pytest.mark.parametrize("drop_last,expected", [(True, [1, 1]), (False, [2, 2])])
def test_three_batches_two_ranks(drop_last, expected):
    samplers = [
        _sampler([6] * 3, rank, 2, shuffle=False, drop_last=drop_last)
        for rank in range(2)
    ]
    assert [len(list(sampler)) for sampler in samplers] == expected
    assert [len(sampler) for sampler in samplers] == expected


@pytest.mark.parametrize("drop_last,expected", [(True, 0), (False, 1)])
def test_one_batch_four_ranks(drop_last, expected):
    for rank in range(4):
        sampler = _sampler([1], rank, 4, shuffle=False, drop_last=drop_last)
        assert len(list(sampler)) == len(sampler) == expected


@pytest.mark.parametrize("lengths", [[], [6, 6, 6], [21, 21], [1, 9, 9, 9]])
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("drop_last", [True, False])
def test_length_matches_actual_batches(lengths, world_size, drop_last):
    for rank in range(world_size):
        sampler = _sampler(lengths, rank, world_size, drop_last=drop_last)
        assert len(sampler) == len(list(sampler))


def test_rank_partition_is_independent_of_process_rng():
    state = random.getstate()
    try:
        rank_batches = []
        for rank in range(4):
            random.seed(100 + rank)
            rank_batches.extend(list(_sampler([6] * 24, rank, 4)))
        assert sorted(index for batch in rank_batches for index in batch) == list(
            range(24)
        )
    finally:
        random.setstate(state)


def test_iteration_and_length_do_not_consume_process_rng():
    sampler = _sampler([6] * 24)
    state = random.getstate()
    try:
        len(sampler)
        list(sampler)
        assert random.getstate() == state
    finally:
        random.setstate(state)


def test_epoch_is_repeatable_and_shared_across_ranks():
    sampler = _sampler([6] * 24)
    first = list(sampler)
    assert list(sampler) == first
    sampler.set_epoch(1)
    second = list(sampler)
    assert second != first
    for rank in range(4):
        sharded = _sampler([6] * 24, rank, 4)
        sharded.set_epoch(1)
        assert list(sharded) == second[rank::4]
    sampler.set_epoch(0)
    assert list(sampler) == first


class _Loader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs

    def __iter__(self):
        indices = list(self.kwargs.get("sampler", range(len(self.dataset))))
        size = self.kwargs["batch_size"]
        for start in range(0, len(indices), size):
            yield self.kwargs["collate_fn"](
                [self.dataset[index] for index in indices[start : start + size]]
            )


def _sample_ids(samples):
    return [sample.sample_id for sample in samples]


def _ground_batches(monkeypatch, count, rank, world_size, path):
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    samples = [
        {"image": Path(f"{index}.jpg"), "label": Path(f"{index}.txt")}
        for index in range(count)
    ]
    spec = GroundDinoSpec(batch_size=2, num_batches=3)
    if path == "sync":
        return list(
            ground_dino._iter_ground_dino_sync_batches(samples, _sample_ids, spec)
        )
    if path == "fallback":
        original_import = builtins.__import__

        def without_loader(name, *args, **kwargs):
            if name == "torch.utils.data":
                raise ImportError("test: torch unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_loader)
        with pytest.warns(RuntimeWarning, match="synchronous"):
            return list(
                ground_dino._iter_ground_dino_batches(
                    dataset=ground_dino._GroundDinoDataset(samples, prompt=spec.prompt),
                    samples=samples,
                    collator=_sample_ids,
                    spec=spec,
                )
            )
    return list(
        ground_dino._iter_ground_dino_batches(
            dataset=ground_dino._GroundDinoDataset(samples, prompt=spec.prompt),
            samples=samples,
            collator=_sample_ids,
            spec=spec,
            dataloader_cls=_Loader,
        )
    )


@pytest.mark.parametrize("path", ["loader", "sync", "fallback"])
@pytest.mark.parametrize("count,world_size", [(8, 2), (5, 2), (1, 4), (2, 5)])
def test_ground_dino_rank_sample_ids(monkeypatch, count, world_size, path):
    per_rank = (count + world_size - 1) // world_size
    padded = [index % count for index in range(per_rank * world_size)]
    for rank in range(world_size):
        local = [str(index) for index in padded[rank::world_size]]
        epoch = [local[start : start + 2] for start in range(0, len(local), 2)]
        expected = [epoch[index % len(epoch)] for index in range(3)]
        assert _ground_batches(monkeypatch, count, rank, world_size, path) == expected


@pytest.mark.parametrize("path", ["loader", "sync", "fallback"])
def test_ground_dino_single_process_semantics(monkeypatch, path):
    expected = (
        [["0", "1"], ["2"], ["0", "1"]]
        if path == "loader"
        else [["0", "1"], ["2", "0"], ["1", "2"]]
    )
    assert _ground_batches(monkeypatch, 3, 0, 1, path) == expected


@pytest.mark.parametrize("path", ["loader", "sync"])
def test_ground_dino_empty_dataset_terminates(monkeypatch, path):
    assert _ground_batches(monkeypatch, 0, 0, 4, path) == []
