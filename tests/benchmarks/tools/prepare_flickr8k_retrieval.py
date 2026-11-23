# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Verify pinned Flickr8k parquet and prepare disjoint WDS retrieval inputs."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

REVISION = "56f58c967835f7c508d684f36bd7897cca9d7634"
FILES = {
    "train-0.parquet": "68bc2d97c6021e575bdfd9aa8d364f865c8ad7840cf3ddc06a1811c2a0accc7e",
    "train-1.parquet": "55afbb11f6916aa636ce34e727c3a0194ea40381e02c20f9fa612fde5b280500",
    "validation.parquet": "2470e49d96fb37e19c8a1d29056623c84ed41c3760a12747e120695aa0c44ad4",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_unique(split, byte_hash, pixel_hash, seen):
    identities = {("bytes", byte_hash), ("pixels", pixel_hash)}
    if seen.intersection(identities):
        raise ValueError(f"duplicate image within/across splits at {split}")
    seen.update(identities)


def prepare(source, output):
    import pyarrow
    import pyarrow.parquet as pq
    from PIL import Image

    for name, expected in FILES.items():
        if file_hash(source / name) != expected:
            raise ValueError(f"source checksum mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    seen, splits, shards = set(), {}, {}
    for split, names, expected_count in (
        ("train", ["train-0.parquet", "train-1.parquet"], 6000),
        ("validation", ["validation.parquet"], 1000),
    ):
        records, archive, raw_count = [], None, 0
        (output / split).mkdir()
        try:
            for name in names:
                for batch in pq.ParquetFile(source / name).iter_batches(batch_size=32):
                    for row in batch.to_pylist():
                        raw = row["image"]["bytes"]
                        captions = [row[f"caption_{i}"] for i in range(5)]
                        if any(
                            not isinstance(c, str) or not c.strip() for c in captions
                        ):
                            raise ValueError("requires five nonempty captions")
                        captions = [c.strip() for c in captions]
                        byte_hash = hashlib.sha256(raw).hexdigest()
                        with Image.open(io.BytesIO(raw)) as image:
                            rgb = image.convert("RGB")
                            pixel_hash = hashlib.sha256(
                                str(rgb.size).encode() + rgb.tobytes()
                            ).hexdigest()
                        source_row = raw_count
                        raw_count += 1
                        # The pinned release has one exact train duplicate with
                        # different captions. Merge annotations, never validation.
                        if split == "train" and source_row == 1359:
                            canonical = records[1027]
                            if (byte_hash, pixel_hash) != (
                                canonical["image_sha256"],
                                canonical["pixel_sha256"],
                            ):
                                raise ValueError("known duplicate identity changed")
                            canonical["source_rows"].append(source_row)
                            canonical["captions"].extend(captions)
                            continue
                        check_unique(split, byte_hash, pixel_hash, seen)
                        index = len(records)
                        if index % 1500 == 0:
                            if archive is not None:
                                archive.close()
                            shard = f"{split}/{index // 1500:05d}.tar"
                            archive = tarfile.open(output / shard, "w")
                        key = f"{index:06d}"
                        for suffix, payload in (
                            ("jpg", raw),
                            ("txt", captions[0].encode("utf-8")),
                        ):
                            entry = tarfile.TarInfo(f"{key}.{suffix}")
                            entry.size = len(payload)
                            archive.addfile(entry, io.BytesIO(payload))
                        records.append(
                            {
                                "key": key,
                                "shard": shard,
                                "image_sha256": byte_hash,
                                "pixel_sha256": pixel_hash,
                                "captions": captions,
                                "source_rows": [source_row],
                            }
                        )
        finally:
            if archive is not None:
                archive.close()
        unique_count = expected_count - (1 if split == "train" else 0)
        if raw_count != expected_count or len(records) != unique_count:
            raise ValueError(f"unexpected {split} size: {len(records)}")
        splits[split] = records
        for path in sorted((output / split).glob("*.tar")):
            shards[path.relative_to(output).as_posix()] = file_hash(path)
    manifest = {
        "schema_version": 1,
        "source": "jxie/flickr8k",
        "revision": REVISION,
        "source_sha256": FILES,
        "shards_sha256": shards,
        "splits": splits,
        "pyarrow_version": pyarrow.__version__,
        "policy": "published splits; pinned train row1359 merged into1027 with all annotations retained; other byte/RGB duplicates rejected; canonical caption_0 training; all five captions for validation; no test split accessed",
        "limitations": "dataset card lacks full provenance/licensing; internal evaluation only; pretraining overlap and near-duplicates not established",
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return {split: len(records) for split, records in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output)))


if __name__ == "__main__":
    main()
