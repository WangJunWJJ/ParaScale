# -*- coding: utf-8 -*-
# @Time : 2026/6/12 上午9:35
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Build a small YOLO-format Object365 cache for ParaScale detection benchmarks."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Object365 Tiny COCO-style annotations and image zips into "
            "a cached YOLO directory with images/ and labels/."
        )
    )
    parser.add_argument("--annotations", help="Path to objects365_Tiny_train.json.")
    parser.add_argument(
        "--objects365-tar",
        help="Optional Objects365_v1.tar.gz. Used when annotations or image zips are inside the tarball.",
    )
    parser.add_argument(
        "--image-zip",
        action="append",
        default=[],
        help="Image zip path. May be passed multiple times.",
    )
    parser.add_argument(
        "--image-zip-dir",
        help="Directory containing train_part*.zip or other Object365 image zips.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output cache directory. images/, labels/ and cache_manifest.json are written here.",
    )
    parser.add_argument(
        "--limit", type=int, default=256, help="Maximum images to cache."
    )
    return parser.parse_args()


def load_annotations(args: argparse.Namespace) -> Dict[str, Any]:
    if args.annotations:
        with open(args.annotations, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if not args.objects365_tar:
        raise ValueError("--annotations or --objects365-tar is required.")
    with tarfile.open(args.objects365_tar, "r:*") as archive:
        member = next(
            (
                item
                for item in archive.getmembers()
                if item.name.endswith("objects365_Tiny_train.json")
            ),
            None,
        )
        if member is None:
            raise ValueError("objects365_Tiny_train.json was not found in the tarball.")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"Could not extract {member.name}.")
        return json.load(extracted)


def build_label_index(data: Dict[str, Any]) -> Tuple[Dict[str, List[str]], int]:
    images = {int(item["id"]): item for item in data["images"]}
    category_ids = sorted({int(item["id"]) for item in data["categories"]})
    category_to_zero_based = {
        category_id: idx for idx, category_id in enumerate(category_ids)
    }
    labels: Dict[str, List[str]] = defaultdict(list)
    for annotation in data["annotations"]:
        if int(annotation.get("iscrowd", 0)):
            continue
        image = images.get(int(annotation["image_id"]))
        if image is None:
            continue
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        image_width = max(float(image["width"]), 1.0)
        image_height = max(float(image["height"]), 1.0)
        x_center = (x + width * 0.5) / image_width
        y_center = (y + height * 0.5) / image_height
        norm_width = width / image_width
        norm_height = height / image_height
        cls_id = category_to_zero_based[int(annotation["category_id"])]
        labels[str(image["file_name"])].append(
            f"{cls_id} {x_center:.8f} {y_center:.8f} {norm_width:.8f} {norm_height:.8f}"
        )
    return labels, len(category_ids)


def image_zip_paths(args: argparse.Namespace) -> List[Path]:
    paths = [Path(path) for path in args.image_zip]
    if args.image_zip_dir:
        paths.extend(sorted(Path(args.image_zip_dir).glob("*.zip")))
    return paths


def cache_from_zip_paths(
    zip_paths: Iterable[Path],
    labels: Dict[str, List[str]],
    output_dir: Path,
    limit: int,
) -> Tuple[int, List[str]]:
    copied = 0
    missing = set(labels)
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            copied += copy_matching_images(
                archive, labels, output_dir, limit - copied, missing
            )
        if copied >= limit:
            break
    return copied, sorted(missing)


def cache_from_tarball(
    tar_path: str,
    labels: Dict[str, List[str]],
    output_dir: Path,
    limit: int,
) -> Tuple[int, List[str]]:
    copied = 0
    missing = set(labels)
    with tarfile.open(tar_path, "r:*") as tar:
        members = [
            item
            for item in tar.getmembers()
            if item.isfile()
            and item.name.endswith(".zip")
            and "train_part" in item.name
        ]
        for member in sorted(members, key=lambda item: item.name):
            with tempfile.NamedTemporaryFile(suffix=".zip") as temp_zip:
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                shutil.copyfileobj(extracted, temp_zip)
                temp_zip.flush()
                with zipfile.ZipFile(temp_zip.name) as archive:
                    copied += copy_matching_images(
                        archive,
                        labels,
                        output_dir,
                        limit - copied,
                        missing,
                    )
            if copied >= limit:
                break
    return copied, sorted(missing)


def copy_matching_images(
    archive: zipfile.ZipFile,
    labels: Dict[str, List[str]],
    output_dir: Path,
    remaining: int,
    missing: set[str],
) -> int:
    copied = 0
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    names = {
        Path(name).name: name
        for name in archive.namelist()
        if Path(name).suffix.lower() in IMAGE_SUFFIXES
    }
    for file_name, rows in labels.items():
        if copied >= remaining:
            break
        archive_name = names.get(file_name)
        if archive_name is None:
            continue
        target_image = image_dir / file_name
        target_label = label_dir / f"{Path(file_name).stem}.txt"
        with archive.open(archive_name) as source, target_image.open("wb") as target:
            shutil.copyfileobj(source, target)
        target_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
        missing.discard(file_name)
        copied += 1
    return copied


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    data = load_annotations(args)
    labels, category_count = build_label_index(data)
    selected_labels = dict(list(labels.items())[: max(1, int(args.limit))])
    zip_paths = image_zip_paths(args)
    if zip_paths:
        copied, missing = cache_from_zip_paths(
            zip_paths, selected_labels, output_dir, args.limit
        )
    elif args.objects365_tar:
        copied, missing = cache_from_tarball(
            args.objects365_tar, selected_labels, output_dir, args.limit
        )
    else:
        raise ValueError(
            "--image-zip, --image-zip-dir or --objects365-tar is required."
        )
    manifest = {
        "images": copied,
        "labels": copied,
        "missing": missing,
        "categories": category_count,
        "format": "yolo_normalized_xywh",
    }
    (output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if copied > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
