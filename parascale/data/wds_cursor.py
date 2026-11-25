# -*- coding: utf-8 -*-
# @Time : 2026/11/20 上午11:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Seekable, single-worker image/text batches with content-checked resume."""

import hashlib
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path


class WdsCursorBatches:
    """Index uncompressed tar offsets, never retain decoded images or past batches.

    State is captured after consumption, not prefetch. One iterator per instance
    and immutable shards are required. Corrupt/unpaired samples fail closed.
    """

    def __init__(self, spec, transform, *, paths=None, prefetch=False):
        self.spec = spec
        self.transform = transform
        # Lookahead requires a CPU transform that never consumes global RNG.
        self.prefetch = prefetch
        self._active = False
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
        if self._active:
            raise RuntimeError("Close the active WDS iterator before restoring state")
        self.validate_state_dict(state)
        self.position = state["position"]

    def validate_state_dict(self, state):
        """Validate a saved cursor without changing data or RNG state."""
        if state.get("identity") != self.identity:
            raise ValueError(
                "WDS resume identity mismatch: dataset, topology or preprocessing changed"
            )
        position = state.get("position")
        if not isinstance(position, int) or position < 0:
            raise ValueError("Invalid WDS resume position")

    def __iter__(self):
        if self._active:
            raise RuntimeError("Only one WDS iterator may be active")
        self._active = True
        try:
            with ExitStack() as stack:
                executor = (
                    ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="parascale-wds"
                    )
                    if self.prefetch
                    else None
                )
                if executor is not None:
                    stack.callback(executor.shutdown, wait=True, cancel_futures=True)
                yield from self._batches(executor)
        finally:
            self._active = False

    def _batches(self, executor):
        remaining = min(
            self.spec.num_batches * self.spec.batch_size,
            getattr(
                self.spec, "num_samples", self.spec.num_batches * self.spec.batch_size
            ),
        )
        pending = None
        for index in range(self.spec.num_batches):
            batch_size = min(self.spec.batch_size, remaining)
            if batch_size <= 0:
                break
            if executor is not None and pending is None:
                pending = executor.submit(self._read_batch, self.position, batch_size)
            result = (
                pending.result()
                if pending is not None
                else self._read_batch(self.position, batch_size)
            )
            next_position = self.position + batch_size
            remaining -= batch_size
            if (
                executor is not None
                and remaining > 0
                and index + 1 < self.spec.num_batches
            ):
                pending = executor.submit(
                    self._read_batch,
                    next_position,
                    min(self.spec.batch_size, remaining),
                )
            self.position = next_position
            yield result

    def _read_batch(self, position, batch_size):
        batch = []
        with ExitStack() as stack:
            streams = {}
            for offset in range(batch_size):
                path, pair = self.samples[(position + offset) % len(self.samples)]
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
        return self.transform(batch)
