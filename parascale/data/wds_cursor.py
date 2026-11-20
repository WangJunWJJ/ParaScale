# -*- coding: utf-8 -*-
# @Time : 2026/11/20 上午11:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Seekable, single-worker image/text batches with content-checked resume."""

import hashlib
import os
import tarfile
from contextlib import ExitStack
from pathlib import Path


class WdsCursorBatches:
    """Index uncompressed tar offsets, never retain decoded images or past batches.

    State is captured after consumption, not prefetch. One iterator per instance
    and immutable shards are required. Corrupt/unpaired samples fail closed.
    """

    def __init__(self, spec, transform, *, paths=None):
        self.spec = spec
        self.transform = transform
        self.position = 0
        self.samples = []
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        paths = (
            sorted(Path(spec.data_dir).glob("*.tar")) if paths is None else list(paths)
        )
        hashes = []
        ordinal = 0
        for shard, path in enumerate(paths):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes.append((path.name, digest.hexdigest()))
            pairs = {}
            with tarfile.open(path, "r:") as archive:
                for member in archive:
                    key, _, suffix = member.name.rpartition(".")
                    if not member.isfile() or suffix not in {
                        "jpg",
                        "jpeg",
                        "png",
                        "webp",
                        "txt",
                    }:
                        continue
                    role = "text" if suffix == "txt" else "image"
                    pair = pairs.setdefault(key, {})
                    if role in pair:
                        raise ValueError(f"Duplicate WDS component: {path}:{key}")
                    pair[role] = (member.offset_data, member.size)
            for key, pair in pairs.items():
                if set(pair) != {"image", "text"}:
                    raise ValueError(f"Unpaired WDS sample: {path}:{key}")
                selected = (
                    shard % world == rank
                    if len(paths) >= world
                    else ordinal % world == rank
                )
                if selected:
                    self.samples.append((path, pair))
                ordinal += 1
        if not self.samples:
            raise ValueError("No WDS samples assigned to this rank")
        self.identity = {
            "version": 1,
            "shards": [list(item) for item in hashes],
            "rank": rank,
            "world_size": world,
            "batch_size": spec.batch_size,
            "model": spec.pretrained_model_name_or_path,
            "text_length": spec.text_length,
        }

    def state_dict(self):
        """Return next-sample position and immutable source identity."""
        return {"identity": self.identity, "position": self.position}

    def load_state_dict(self, state):
        """Reject changed data/topology before applying a saved cursor."""
        if state.get("identity") != self.identity:
            raise ValueError(
                "WDS resume identity mismatch: dataset, topology or preprocessing changed"
            )
        position = state.get("position")
        if not isinstance(position, int) or position < 0:
            raise ValueError("Invalid WDS resume position")
        self.position = position

    def __iter__(self):
        remaining = min(
            self.spec.num_batches * self.spec.batch_size,
            getattr(
                self.spec, "num_samples", self.spec.num_batches * self.spec.batch_size
            ),
        )
        for _ in range(self.spec.num_batches):
            batch_size = min(self.spec.batch_size, remaining)
            if batch_size <= 0:
                break
            batch = []
            with ExitStack() as stack:
                streams = {}
                for offset in range(batch_size):
                    path, pair = self.samples[
                        (self.position + offset) % len(self.samples)
                    ]
                    if path not in streams:
                        streams[path] = stack.enter_context(path.open("rb"))
                    stream = streams[path]
                    values = {}
                    for role, (start, size) in pair.items():
                        stream.seek(start)
                        blob = stream.read(size)
                        if len(blob) != size:
                            raise ValueError("WDS shard truncated after indexing")
                        values[role] = (
                            blob.decode("utf-8").strip() if role == "text" else blob
                        )
                    if not values["text"]:
                        raise ValueError("Empty WDS caption")
                    batch.append(values)
            result = self.transform(batch)
            self.position += batch_size
            remaining -= batch_size
            yield result
