# -*- coding: utf-8 -*-
# @Time : 2026/6/11 上午9:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest

torch = pytest.importorskip("torch")

from parascale import VisionCollator


def test_vision_collator_stacks_equal_size_tensors():
    collator = VisionCollator(patch_size=16)
    batch = collator(
        [
            {
                "pixel_values": torch.ones(3, 32, 32),
                "label": 1,
                "height": 32,
                "width": 32,
            },
            {
                "pixel_values": torch.zeros(3, 32, 32),
                "label": 2,
                "height": 32,
                "width": 32,
            },
        ]
    )

    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["labels"].tolist() == [1, 2]
    assert batch["num_images"] == 2
    assert batch["patch_tokens"] == 8
    assert batch["per_sample_patch_tokens"] == [4, 4]


def test_vision_collator_pads_variable_size_tensors():
    collator = VisionCollator(patch_size=16, pad_to_multiple=16)
    batch = collator(
        [
            {"pixel_values": torch.ones(3, 32, 48), "label": 0},
            {"pixel_values": torch.ones(3, 48, 32), "label": 1},
        ]
    )

    assert batch["pixel_values"].shape == (2, 3, 48, 48)
    assert batch["image_sizes"] == [(32, 48), (48, 32)]
    assert batch["labels"].dtype == torch.long
