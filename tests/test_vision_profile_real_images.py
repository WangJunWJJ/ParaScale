# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from parascale.cli import cmd_vision_profile
from parascale.data.vision import find_image_files, profile_image_folder


def _write_images(root: Path):
    class_dir = root / "class-a"
    class_dir.mkdir(parents=True)
    for index, size in enumerate([(32, 32), (48, 64), (64, 48), (40, 40)]):
        image = Image.new("RGB", size, color=(index * 30, 20, 120))
        image.save(class_dir / f"sample-{index}.png")


def test_profile_image_folder_reads_real_pngs(tmp_path):
    _write_images(tmp_path)

    profile = profile_image_folder(
        tmp_path,
        batch_size=2,
        max_batches=2,
        image_size=32,
        patch_size=16,
        device="cpu",
    )
    payload = profile.to_dict()

    assert len(find_image_files(tmp_path)) == 4
    assert payload["images"] == 4
    assert payload["batches"] == 2
    assert payload["patch_tokens"] > 0
    assert payload["images_per_second"] > 0
    assert payload["decode_time_ms"] >= 0


def test_cli_vision_profile_writes_json(tmp_path):
    _write_images(tmp_path)
    output = tmp_path / "profile.json"

    args = Namespace(
        data_dir=str(tmp_path),
        batch_size=2,
        max_batches=1,
        image_size=32,
        patch_size=16,
        device="cpu",
        output=str(output),
    )
    assert cmd_vision_profile(args) == 0
    text = output.read_text(encoding="utf-8")
    assert '"mode": "vision_profile"' in text
