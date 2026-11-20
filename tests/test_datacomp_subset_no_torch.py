# -*- coding: utf-8 -*-
# @Time : 2026/11/20 上午11:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

PATH = Path(__file__).parent / "benchmarks/tools/prepare_datacomp_subset.py"


def module():
    assert PATH.exists(), "bounded subset tool must exist"
    spec = importlib.util.spec_from_file_location("subset_tool", PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def picture():
    image = Image.new("RGB", (64, 64))
    image.putdata([(255, 255, 255) if i % 3 else (0, 0, 0) for i in range(4096)])
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_blur_changes_pixels_and_rejects_unknown_boxes():
    tool = module()
    plain = tool.prepare_image(picture(), [])
    blurred = tool.prepare_image(picture(), [[0.1, 0.1, 0.9, 0.9]])
    assert plain != blurred
    assert Image.open(io.BytesIO(blurred)).format == "JPEG"
    for boxes in (None, [[0, 0, 2, 1]], [[0.8, 0, 0.1, 1]]):
        with pytest.raises(ValueError):
            tool.prepare_image(picture(), boxes)
    with pytest.raises(Exception):
        tool.prepare_image(b"not an image", [])


def test_tar_contains_matching_text_and_hash(tmp_path):
    tool = module()
    row = {
        "uid": "a" * 32,
        "url": "https://example.com/a",
        "text": "a caption",
        "face_bboxes": [],
    }
    blob = tool.prepare_image(picture(), [])
    with tarfile.open(tmp_path / "data.tar", "w") as archive:
        result = tool.add_sample(archive, row, blob)
    assert result["image_sha256"] == hashlib.sha256(blob).hexdigest()
    with tarfile.open(tmp_path / "data.tar") as archive:
        assert archive.extractfile(row["uid"] + ".txt").read() == b"a caption"
        assert archive.extractfile(row["uid"] + ".jpg").read() == blob
    with pytest.raises(ValueError):
        tool.validate_row(dict(row, uid="../bad"))


def test_fetch_enforces_limits_and_no_tls_bypass(tmp_path, monkeypatch):
    tool = module()
    captured = []

    def run(command, **kwargs):
        captured.append((command, kwargs))
        Path(command[command.index("--output") + 1]).write_bytes(b"ok")

        class Result:
            returncode = 0
            stderr = b""

        return Result()

    monkeypatch.setattr(tool.subprocess, "run", run)
    assert tool.fetch("https://example.com/x", tmp_path, 13, 100) == b"ok"
    command, kwargs = captured[0]
    assert command[command.index("--max-time") + 1] == "13"
    assert command[command.index("--max-filesize") + 1] == "100"
    assert "--insecure" not in command and "-k" not in command
    assert kwargs["timeout"] == 18
    with pytest.raises(ValueError):
        tool.fetch("file:///etc/passwd", tmp_path, 1, 10)


def test_existing_output_is_never_reused(tmp_path):
    tool = module()
    marker = tmp_path / "keep"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError):
        tool.create_output(tmp_path)
    assert marker.read_text() == "untouched"


def test_existing_wds_preflight_rejects_unpaired_members(tmp_path):
    from tests.benchmarks.scripts.validate_clip_recovery import inspect_wds

    tool = module()
    row = {"uid": "a" * 32, "text": "caption", "face_bboxes": []}
    blob = tool.prepare_image(picture(), [])
    with tarfile.open(tmp_path / "data.tar", "w") as archive:
        tool.add_sample(archive, row, blob)
    records = inspect_wds(tmp_path)
    assert records[0]["caption"] == "caption"
    assert records[0]["image_sha256"] == hashlib.sha256(blob).hexdigest()
    with tarfile.open(tmp_path / "bad.tar", "w") as archive:
        info = tarfile.TarInfo("missing.jpg")
        info.size = len(blob)
        archive.addfile(info, io.BytesIO(blob))
    with pytest.raises(ValueError, match="Unpaired"):
        inspect_wds(tmp_path)


def test_pipeline_exports_wds_manifest_and_hashes(tmp_path, monkeypatch):
    import argparse

    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    tool = module()
    row = {
        "uid": "b" * 32,
        "url": "https://example.com/b",
        "text": "caption",
        "face_bboxes": [],
        "sha256": hashlib.sha256(picture()).hexdigest(),
    }
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist([row]), buffer)
    parquet = buffer.getvalue()
    monkeypatch.setattr(tool, "METADATA_SHA256", hashlib.sha256(parquet).hexdigest())

    def fetch(url, *args, **kwargs):
        if "/raw/" in url and url.endswith(".parquet"):
            return ("oid sha256:" + hashlib.sha256(parquet).hexdigest()).encode()
        if url.endswith(".parquet"):
            return parquet
        if url.endswith("README.md"):
            return b"license: cc-by-4.0"
        return picture()

    monkeypatch.setattr(tool, "fetch", fetch)
    output = tmp_path / "output"
    args = argparse.Namespace(
        output=str(output),
        endpoint="https://hf-mirror.com",
        target=1,
        max_rows=2,
        seconds=10,
        timeout=1,
        workers=1,
    )
    assert tool.run(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["successful"] == 1
    assert manifest["shards"] == ["wds/subset-00000.tar"]
    assert (output / manifest["shards"][0]).exists()
    assert b"\r" not in (output / "SHA256SUMS").read_bytes()
    from tests.benchmarks.scripts.validate_clip_recovery import validate_subset

    folder, records = validate_subset(output)
    assert folder == output / "wds"
    assert len(records) == 1
    (folder / "subset-00000.tar").write_bytes(b"corrupt")
    with pytest.raises((ValueError, tarfile.ReadError)):
        validate_subset(output)
    # Restore the archive before checking the acquisition manifest.
    with tarfile.open(folder / "subset-00000.tar", "w") as archive:
        tool.add_sample(archive, row, tool.prepare_image(picture(), []))
    for line in (output / "SHA256SUMS").read_text().splitlines():
        sha, filename = line.split("  ")
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == sha


def test_card_failure_does_not_block_metadata_and_failure_is_recorded(
    tmp_path, monkeypatch
):
    import argparse

    tool = module()
    calls = []

    def fetch(url, *args, **kwargs):
        calls.append(url)
        raise RuntimeError("HTTP 503" if url.endswith("README.md") else "HTTP 429")

    monkeypatch.setattr(tool, "fetch", fetch)
    output = tmp_path / "blocked"
    args = argparse.Namespace(
        output=str(output),
        endpoint="https://hf-mirror.com",
        target=1,
        max_rows=2,
        seconds=10,
        timeout=1,
        workers=1,
    )
    assert tool.run(args) == 2
    assert len(calls) == 2 and calls[-1].endswith(".parquet")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["successful"] == 0 and manifest["shards"] == []
    assert manifest["error"] == "HTTP 429"
    assert not list(output.rglob("*.tar"))


def test_short_download_batches_do_not_skip_candidates(tmp_path, monkeypatch):
    from types import SimpleNamespace

    pq = pytest.importorskip("pyarrow.parquet")
    tool = module()
    rows = [
        dict(
            uid=f"{index:032x}",
            text="caption",
            url=f"https://example.com/{index}",
            face_bboxes=[],
        )
        for index in range(5)
    ]
    monkeypatch.setattr(
        pq,
        "ParquetFile",
        lambda path: SimpleNamespace(
            iter_batches=lambda **kwargs: [SimpleNamespace(to_pylist=lambda: rows)]
        ),
    )
    monkeypatch.setattr(tool, "METADATA_SHA256", tool.digest(b"metadata"))

    def fetch(url, *args, **kwargs):
        if url.endswith(".parquet"):
            return b"metadata"
        if url in (rows[0]["url"], rows[1]["url"]):
            raise RuntimeError("unavailable")
        return picture()

    monkeypatch.setattr(tool, "fetch", fetch)
    args = SimpleNamespace(
        output=str(tmp_path / "subset"),
        endpoint="https://hf-mirror.com",
        target=2,
        max_rows=5,
        seconds=30,
        timeout=1,
        workers=4,
    )
    assert tool.run(args) == 0
    summary = json.loads((tmp_path / "subset/summary.json").read_text())
    assert summary["attempted"] == 4
    assert summary["successful"] == 2
