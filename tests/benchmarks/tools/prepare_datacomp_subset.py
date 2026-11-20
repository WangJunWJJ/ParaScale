# -*- coding: utf-8 -*-
# @Time : 2026/11/20 上午11:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded, CPU-only DataComp small acquisition; not a small-track reproduction.

Requires Pillow and pyarrow. curl provides hard transfer time/size bounds.
The output directory must not exist. Images retain their original copyrights.
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

REPO = "mlfoundations/datacomp_small"
REVISION = "b7dd54c19199236fe33f367ce7f88f412244fab8"
PARQUET = "0020f0cbd157d470aa56bea08e304b90.parquet"
METADATA_SHA256 = "aaaa9c99f1ab5c8e16c91499268ecfb30f1b3e9304f3405281f3288e39d32568"
OFFICIAL = "https://huggingface.co/datasets/" + REPO


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def create_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def fetch(url, temporary_dir, timeout=20, max_bytes=8 * 1024 * 1024):
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("Only HTTP(S) resources are allowed")
    with tempfile.NamedTemporaryFile(dir=temporary_dir, delete=False) as handle:
        path = Path(handle.name)
    try:
        command = [
            "curl.exe" if os.name == "nt" else "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--proto",
            "=http,https",
            "--proto-redir",
            "=http,https",
            "--max-redirs",
            "5",
            "--connect-timeout",
            str(min(timeout, 10)),
            "--max-time",
            str(timeout),
            "--max-filesize",
            str(max_bytes),
            "--output",
            str(path),
            url,
        ]
        result = subprocess.run(command, capture_output=True, timeout=timeout + 5)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[:1000])
        if path.stat().st_size > max_bytes:
            raise ValueError("Resource exceeds size bound")
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def validate_row(row):
    if not re.fullmatch("[0-9a-f]{32}", row["uid"]):
        raise ValueError("Invalid UID")
    if not isinstance(row["text"], str) or not row["text"].strip():
        raise ValueError("Empty caption")
    if len(row["text"].encode("utf-8")) > 1024 * 1024:
        raise ValueError("Caption too large")


def prepare_image(blob, boxes):
    from PIL import Image, ImageFilter

    if not isinstance(boxes, list):
        raise ValueError("Missing face_bboxes; refusing unblurred output")
    with Image.open(io.BytesIO(blob)) as source:
        if source.width * source.height > 25_000_000:
            raise ValueError("Image exceeds 25 million pixels")
        image = source.convert("RGB")
    width, height = image.size
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box):
            raise ValueError("Invalid normalized face bbox")
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Degenerate face bbox")
        diagonal = max((x1 - x0) * width, (y1 - y0) * height)
        margin = 0.1 * diagonal
        bounds = (
            max(0, math.floor(x0 * width - margin)),
            max(0, math.floor(y0 * height - margin)),
            min(width, math.ceil(x1 * width + margin)),
            min(height, math.ceil(y1 * height + margin)),
        )
        # Same normalized boxes and expanded regions; Pillow implementation,
        # not a claim of byte-equivalence with img2dataset's blur kernel.
        blurred = image.filter(ImageFilter.GaussianBlur(max(1, diagonal * 0.1)))
        image.paste(blurred.crop(bounds), bounds)
    image.thumbnail((512, 512))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=95)
    return output.getvalue()


def add_sample(archive, row, blob):
    validate_row(row)
    for suffix, value in [("jpg", blob), ("txt", row["text"].encode("utf-8"))]:
        info = tarfile.TarInfo(row["uid"] + "." + suffix)
        info.size = len(value)
        archive.addfile(info, io.BytesIO(value))
    return dict(
        row,
        status="success",
        image_sha256=digest(blob),
        image_bytes=len(blob),
        face_blur_applied=bool(row["face_bboxes"]),
    )


def acquire_row(row, output, timeout):
    try:
        validate_row(row)
        raw = fetch(row["url"], output, timeout)
        blob = prepare_image(raw, row.get("face_bboxes"))
        return (
            row,
            blob,
            dict(
                download_sha256=digest(raw),
                original_sha256_matches=digest(raw) == row.get("sha256"),
            ),
        )
    except Exception as error:
        return row, None, dict(status="failed", error=str(error)[:1000])


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")


def run(args):
    output = Path(args.output).resolve()
    create_output(output)
    started = time.monotonic()
    summary = dict(
        dataset=REPO,
        revision=REVISION,
        requested=args.target,
        successful=0,
        attempted=0,
        resources=[],
        scope="Bounded sequential subset, availability-biased; NOT full small track",
        license="Metadata CC-BY-4.0; individual images retain their copyrights",
        face_policy="Normalized face_bboxes always blurred with expanded regions; Pillow GaussianBlur, not byte-identical to img2dataset",
        limits=dict(
            max_rows=args.max_rows,
            seconds=args.seconds,
            image_bytes=8 * 1024 * 1024,
            workers=args.workers,
        ),
        official_source=OFFICIAL,
        transport=args.endpoint,
    )

    def resource(name, url, limit=8 * 1024 * 1024, timeout=25):
        entry = dict(file=name, url=url)
        summary["resources"].append(entry)
        try:
            blob = fetch(url, output, timeout, limit)
            (output / name).write_bytes(blob)
            entry.update(status="success", sha256=digest(blob), bytes=len(blob))
            return blob
        except Exception as error:
            entry.update(status="failed", error=str(error))
            raise

    try:
        base = args.endpoint.rstrip("/") + "/datasets/" + REPO
        try:
            resource(
                "dataset-card.md", base + "/raw/" + REVISION + "/README.md", timeout=10
            )
        except Exception:
            pass  # License/source already recorded; card availability is optional.
        # Verify mirrored bytes against the pinned resolver's LFS object hash.
        summary["expected_metadata_sha256"] = METADATA_SHA256
        summary["hash_source"] = (
            "Pinned resolve response X-Linked-Etag and X-Repo-Commit"
        )
        metadata = resource(
            PARQUET,
            base + "/resolve/" + REVISION + "/" + PARQUET,
            limit=128 * 1024 * 1024,
            timeout=180,
        )
        if digest(metadata) != METADATA_SHA256:
            raise ValueError("Metadata LFS SHA256 mismatch")
        summary["metadata_sha256_verified"] = True
        import pyarrow.parquet as pq

        rows = []
        for batch in pq.ParquetFile(output / PARQUET).iter_batches(batch_size=256):
            rows.extend(batch.to_pylist()[: args.max_rows - len(rows)])
            if len(rows) >= args.max_rows:
                break
        summary["candidate_rows"] = len(rows)
        with (output / "candidates.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        seen = set()
        (output / "wds").mkdir()
        summary["shards"] = ["wds/subset-00000.tar"]
        with (
            tarfile.open(output / summary["shards"][0], "w") as archive,
            (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
            ThreadPoolExecutor(max_workers=args.workers) as pool,
        ):
            offset = 0
            while offset < len(rows):
                if (
                    summary["successful"] >= args.target
                    or time.monotonic() - started >= args.seconds
                ):
                    break
                batch = rows[
                    offset : offset
                    + min(args.workers, args.target - summary["successful"])
                ]
                offset += len(batch)
                jobs = [
                    pool.submit(acquire_row, row, output, args.timeout) for row in batch
                ]
                for job in jobs:
                    row, blob, detail = job.result()
                    summary["attempted"] += 1
                    if row["uid"] in seen:
                        blob, detail = (
                            None,
                            dict(status="failed", error="duplicate UID"),
                        )
                    if blob is not None:
                        result = add_sample(archive, row, blob)
                        seen.add(row["uid"])
                        summary["successful"] += 1
                    else:
                        result = dict(row)
                    result.update(detail, revision=REVISION, metadata_file=PARQUET)
                    manifest.write(json.dumps(result) + "\n")
                    manifest.flush()
                print(
                    json.dumps({k: summary[k] for k in ("attempted", "successful")}),
                    flush=True,
                )
        summary["status"] = (
            "complete" if summary["successful"] >= args.target else "partial"
        )
    except Exception as error:
        summary.update(status="blocked", error=str(error))
    summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
    summary.setdefault("shards", [])
    save_json(output / "summary.json", summary)
    save_json(output / "manifest.json", summary)
    with (output / "SHA256SUMS").open("w", encoding="ascii", newline="\n") as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                with path.open("rb") as stream:
                    checksum = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(chunk)
                    sha = checksum.hexdigest()
                handle.write(f"{sha}  {path.relative_to(output).as_posix()}\n")
    print(json.dumps(summary), flush=True)
    return 0 if summary["successful"] >= args.target else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--target", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=12)
    args = parser.parse_args()
    if not (
        1 <= args.target <= args.max_rows <= 10000
        and 1 <= args.workers <= 32
        and 1 <= args.seconds <= 1800
        and 1 <= args.timeout <= 60
    ):
        parser.error("Bounds exceeded")
    if args.endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        parser.error("Only official HF or hf-mirror transport is supported")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
