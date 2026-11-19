# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午12:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Check actual CPU DataLoader rank partitions without model downloads."""

from pathlib import Path

import pytest

from parascale.workloads import ground_dino
from parascale.workloads.specs.ground_dino import GroundDinoSpec


@pytest.mark.parametrize("count,world_size", [(8, 2), (5, 2), (1, 4), (2, 5)])
def test_real_loader_matches_sync_rank_ids(monkeypatch, count, world_size):
    pytest.importorskip("torch")
    samples = [
        {"image": Path(f"{index}.jpg"), "label": Path(f"{index}.txt")}
        for index in range(count)
    ]
    spec = GroundDinoSpec(batch_size=2, num_batches=4, pin_memory=False)
    dataset = ground_dino._GroundDinoDataset(samples, prompt=spec.prompt)
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    for rank in range(world_size):
        monkeypatch.setenv("RANK", str(rank))
        actual = list(
            ground_dino._iter_ground_dino_batches(
                dataset=dataset,
                samples=samples,
                collator=lambda batch: [sample.sample_id for sample in batch],
                spec=spec,
            )
        )
        expected = list(
            ground_dino._iter_ground_dino_sync_batches(
                samples,
                lambda batch: [sample.sample_id for sample in batch],
                spec,
            )
        )
        assert actual == expected
        assert len(actual) == spec.num_batches
