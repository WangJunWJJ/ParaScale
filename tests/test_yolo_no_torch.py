# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:37
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest

from parascale.workloads.yolo import _iter_yolo_official_batches


def test_yolo_official_dataloader_propagates_iteration_errors():
    class Spec:
        batch_size = 2
        num_workers = 0
        pin_memory = False
        prefetch_factor = 2
        persistent_workers = False
        num_batches = 1

    class Device:
        type = "cpu"

    class BrokenDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def __iter__(self):
            raise RuntimeError("worker collate failed")

    with pytest.warns(RuntimeWarning, match="YOLO DataLoader failed"):
        with pytest.raises(RuntimeError, match="worker collate failed"):
            list(
                _iter_yolo_official_batches(
                    dataset=[object()],
                    samples=[object(), object()],
                    collator=lambda samples: {"samples": samples},
                    spec=Spec(),
                    device=Device(),
                    dataloader_cls=BrokenDataLoader,
                )
            )
