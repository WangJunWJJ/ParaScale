# -*- coding: utf-8 -*-
# @Time : 2026/6/22 上午9:47
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import warnings

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from parascale.data.vision import (
    VisionBatchCollator,
    VisionPreprocessor,
    VisionSample,
    VisionTransformConfig,
)
from parascale.workloads.adapters.yolo import (
    YoloDetectionTargetAdapter,
    YoloOfficialBatchAdapter,
)


def _write_yolo_sample(root, name="000001"):
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image = Image.new("RGB", (12, 10), color=(20, 40, 60))
    image_path = image_dir / f"{name}.jpg"
    image.save(image_path)
    label_path = label_dir / f"{name}.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    return {"image": image_path, "label": label_path}


def test_yolo_official_batch_tensor_cache_records_hit_ratio(tmp_path):
    sample = _write_yolo_sample(tmp_path)
    cache_dir = tmp_path / "tensor-cache"
    collator = VisionBatchCollator(
        preprocessor=VisionPreprocessor(
            transform=VisionTransformConfig(
                image_size=32,
                cache_format="vision_tensor_yolo_official_v1",
            ),
            target_adapter=YoloDetectionTargetAdapter(),
            tensor_cache=True,
            tensor_cache_dir=str(cache_dir),
        ),
        batch_adapter=YoloOfficialBatchAdapter(image_size=32),
    )
    vision_sample = VisionSample(
        image=sample["image"],
        annotation=sample["label"],
        sample_id=sample["image"].stem,
    )

    first = collator([vision_sample])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        second = collator([vision_sample])

    assert first["img"].shape == (1, 3, 32, 32)
    assert first["cls"].shape == (1, 1)
    assert first["bboxes"].shape == (1, 4)
    assert first["pipeline_profile"]["tensor_cache_hit_ratio"] == 0.0
    assert first["pipeline_profile"]["cache_hit"] == 0.0
    assert first["pipeline_profile"]["sample_tensor_build_ms"] >= 0.0
    assert first["pipeline_profile"]["label_build_ms"] >= 0.0
    assert second["pipeline_profile"]["tensor_cache_hit_ratio"] == 1.0
    assert second["pipeline_profile"]["cache_hit"] == 1.0
    assert list(cache_dir.glob("*.pt"))
