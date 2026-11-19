# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Prepare a deterministic COCO128 96/32 split for bounded detector validation."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def prepare(assets):
    import yaml
    from ultralytics import YOLO

    assets = assets.resolve()
    with zipfile.ZipFile(assets / "coco128.zip") as archive:
        for name in archive.namelist():
            if not (assets / name).resolve().is_relative_to(assets):
                raise ValueError("Archive member outside asset directory")
        archive.extractall(assets)
    root = assets / "coco128"
    images = sorted((root / "images/train2017").glob("*.jpg"))
    if len(images) != 128:
        raise ValueError("Expected exactly 128 COCO128 images")
    train = assets / "train96"
    (train / "images").mkdir(parents=True, exist_ok=True)
    (train / "labels").mkdir(exist_ok=True)
    for image in images[:96]:
        shutil.copy2(image, train / "images" / image.name)
        label = root / "labels/train2017" / f"{image.stem}.txt"
        if label.exists():
            shutil.copy2(label, train / "labels")
        else:
            (train / "labels" / label.name).write_text("", encoding="utf-8")
    for split, paths in (("train", images[:96]), ("val", images[96:])):
        (assets / f"{split}.txt").write_text(
            "".join(f"{path}\n" for path in paths), encoding="utf-8"
        )
    names = YOLO(str(assets / "yolov8n.pt")).model.names
    (assets / "evaluation.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(assets),
                "train": "train.txt",
                "val": "val.txt",
                "names": names,
            }
        ),
        encoding="utf-8",
    )
    (assets / "split.json").write_text(
        json.dumps(
            {
                "train": [p.name for p in images[:96]],
                "val": [p.name for p in images[96:]],
                "scope": "Disjoint fine-tuning split, not unseen during COCO pretraining.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    prepare(parser.parse_args().assets)
