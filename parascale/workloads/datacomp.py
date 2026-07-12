# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:55
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""DataComp WebDataset and metadata adapters for CLIP/VLM workloads."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from parascale.data.vision import estimate_patch_tokens
from parascale.workloads.specs import ClipContrastiveSpec

from .common import _require_pil_image


class _DataCompWdsIterableDataset:
    def __init__(self, torch: Any, spec: ClipContrastiveSpec) -> None:
        self.torch = torch
        self.spec = spec
        if not spec.data_dir:
            raise ValueError(
                "streaming datacomp_wds workload requires data.data_dir or data.wds_dir."
            )
        root = Path(spec.data_dir)
        if not root.exists():
            raise FileNotFoundError(f"DataComp WDS directory does not exist: {root}")
        self.tar_paths = _datacomp_tar_paths(spec)
        if not self.tar_paths:
            raise FileNotFoundError(f"DataComp WDS directory has no .tar files: {root}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_id, num_workers = _torch_worker_shard()
        rank = int(os.environ.get("RANK", "0") or 0)
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
        shard_index = rank * num_workers + worker_id
        shard_count = max(1, world_size * num_workers)
        produced = 0
        if len(self.tar_paths) >= shard_count:
            tar_paths = [
                tar_path
                for tar_ordinal, tar_path in enumerate(self.tar_paths)
                if tar_ordinal % shard_count == shard_index
            ]
            sample_shard_count = 1
            sample_shard_index = 0
        else:
            tar_paths = self.tar_paths
            sample_shard_count = shard_count
            sample_shard_index = shard_index
        while produced < self.spec.num_samples:
            cycle_produced = 0
            ordinal = 0
            for tar_path in tar_paths:
                for stem, entry in _iter_datacomp_tar_entries(tar_path):
                    if ordinal % sample_shard_count != sample_shard_index:
                        ordinal += 1
                        continue
                    ordinal += 1
                    if "image_bytes" not in entry or "text" not in entry:
                        continue
                    if not _looks_like_supported_image(entry["image_bytes"]):
                        continue
                    try:
                        sample = _datacomp_entry_to_sample(
                            self.torch, stem, entry, self.spec
                        )
                    except Exception:
                        continue
                    yield sample
                    produced += 1
                    cycle_produced += 1
                    if produced >= self.spec.num_samples:
                        return
            if cycle_produced == 0:
                raise RuntimeError(
                    "DataComp worker shard contains no usable image-text samples: "
                    f"shard_index={shard_index}, shard_count={shard_count}."
                )


def _stream_datacomp_wds_batches(
    torch: Any, spec: ClipContrastiveSpec
) -> Iterable[List[Dict[str, Any]]]:
    try:
        from torch.utils.data import DataLoader, IterableDataset
    except Exception:
        yield from _stream_datacomp_wds_batches_inline(torch, spec)
        return

    class TorchDataCompWdsIterableDataset(_DataCompWdsIterableDataset, IterableDataset):
        pass

    dataset = TorchDataCompWdsIterableDataset(torch, spec)
    kwargs: Dict[str, Any] = {
        "batch_size": spec.batch_size,
        "num_workers": max(0, spec.num_workers),
        "collate_fn": lambda samples: list(samples),
    }
    if spec.num_workers > 0:
        kwargs["prefetch_factor"] = max(1, spec.prefetch_factor)
        kwargs["persistent_workers"] = bool(spec.persistent_workers)
    loader = DataLoader(dataset, **kwargs)
    yielded = 0
    for samples in loader:
        if samples:
            yield list(samples)
            yielded += 1
        if yielded >= spec.num_batches:
            break


def _stream_datacomp_wds_batches_inline(
    torch: Any, spec: ClipContrastiveSpec
) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    yielded = 0
    for sample in _DataCompWdsIterableDataset(torch, spec):
        batch.append(sample)
        if len(batch) >= spec.batch_size:
            yield batch
            yielded += 1
            batch = []
            if yielded >= spec.num_batches:
                return
    if batch and yielded < spec.num_batches:
        yield batch


def _torch_worker_shard() -> tuple[int, int]:
    try:
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
    except Exception:
        worker = None
    if worker is None:
        return 0, 1
    return int(worker.id), int(worker.num_workers)


def _build_datacomp_wds_dataset(
    torch: Any, spec: ClipContrastiveSpec
) -> List[Dict[str, Any]]:
    if not spec.data_dir:
        raise ValueError(
            "datacomp_wds workload requires data.data_dir or data.wds_dir."
        )
    root = Path(spec.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"DataComp WDS directory does not exist: {root}")
    tar_paths = _datacomp_tar_paths(spec)
    if not tar_paths:
        raise FileNotFoundError(f"DataComp WDS directory has no .tar files: {root}")

    dataset: List[Dict[str, Any]] = []
    for tar_path in tar_paths:
        for sample in _read_datacomp_tar_stream(
            torch, tar_path, spec, spec.num_samples - len(dataset)
        ):
            dataset.append(sample)
        if len(dataset) >= spec.num_samples:
            break
    if not dataset:
        raise RuntimeError(f"No usable image-text samples found in {root}")
    return dataset


def _read_datacomp_tar(
    torch: Any, tar_path: Path, spec: ClipContrastiveSpec, limit: int
) -> List[Dict[str, Any]]:
    return list(_read_datacomp_tar_stream(torch, tar_path, spec, limit))


def _read_datacomp_tar_stream(
    torch: Any, tar_path: Path, spec: ClipContrastiveSpec, limit: int
) -> Iterator[Dict[str, Any]]:
    emitted = 0
    last_error: str | None = None
    for stem, entry in _iter_datacomp_tar_entries(tar_path):
        if emitted >= limit:
            break
        if "image_bytes" not in entry or "text" not in entry:
            continue
        if not _looks_like_supported_image(entry["image_bytes"]):
            continue
        try:
            yield _datacomp_entry_to_sample(torch, stem, entry, spec)
            emitted += 1
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
    if emitted == 0 and last_error is not None:
        raise RuntimeError(
            f"Failed to decode DataComp samples from {tar_path}: {last_error}"
        )


def _datacomp_tar_paths(spec: ClipContrastiveSpec) -> List[Path]:
    if not spec.data_dir:
        return []
    source_paths = sorted(Path(spec.data_dir).glob("*.tar"))
    if not spec.dataset_local_cache_dir:
        return source_paths
    cache_root = Path(spec.dataset_local_cache_dir)
    cached_paths: List[Path] = []
    for source_path in source_paths:
        cached_paths.append(_cache_datacomp_tar_path(source_path, cache_root))
    return cached_paths


def _cache_datacomp_tar_path(source_path: Path, cache_root: Path) -> Path:
    try:
        stat = source_path.stat()
    except OSError:
        return source_path
    fingerprint = hashlib.sha256(
        f"{source_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    cached_path = cache_root / f"{source_path.stem}-{fingerprint}{source_path.suffix}"
    try:
        cached_stat = cached_path.stat()
        if cached_stat.st_size == stat.st_size:
            return cached_path
    except OSError:
        pass
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_path = cached_path.with_suffix(f"{cached_path.suffix}.{os.getpid()}.tmp")
        shutil.copy2(source_path, tmp_path)
        os.replace(tmp_path, cached_path)
        return cached_path
    except OSError:
        try:
            if "tmp_path" in locals():
                tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return source_path


def _iter_datacomp_tar_entries(tar_path: Path) -> Iterator[tuple[str, Dict[str, Any]]]:
    open_start = time.perf_counter()
    with tarfile.open(tar_path, "r") as archive:
        tar_open_ms = (time.perf_counter() - open_start) * 1000.0
        current_stem: str | None = None
        current_entry: Dict[str, Any] = {}
        emitted_stems: set[str] = set()
        for member in archive:
            if not member.isfile():
                continue
            stem, suffix = _split_wds_member(member.name)
            if suffix not in {"jpg", "jpeg", "png", "webp", "txt", "json"}:
                continue
            if current_stem is None:
                current_stem = stem
            elif stem != current_stem:
                yield current_stem, current_entry
                emitted_stems.add(current_stem)
                if stem in emitted_stems:
                    raise RuntimeError(
                        "DataComp WDS members for stem "
                        f"'{stem}' are not contiguous in {tar_path}; "
                        "streaming mode requires grouped WebDataset samples."
                    )
                current_stem = stem
                current_entry = {}
            entry = current_entry
            metadata = entry.setdefault(
                "metadata",
                {
                    "source_tar": str(tar_path),
                    "wds_tar_open_ms": tar_open_ms,
                    "wds_shard_read_ms": 0.0,
                },
            )
            if entry.get("stem") not in (None, stem):
                raise RuntimeError(
                    f"DataComp WDS stem order changed within {tar_path}: {entry.get('stem')} -> {stem}"
                )
            entry["stem"] = stem
            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue
            read_start = time.perf_counter()
            payload = file_obj.read()
            metadata["wds_shard_read_ms"] = (
                float(metadata.get("wds_shard_read_ms", 0.0) or 0.0)
                + (time.perf_counter() - read_start) * 1000.0
            )
            if suffix in {"jpg", "jpeg", "png", "webp"}:
                entry["image_bytes"] = payload
                entry["image_ext"] = suffix
            elif suffix == "txt":
                entry["text"] = payload.decode("utf-8", errors="replace").strip()
            elif suffix == "json":
                try:
                    metadata.update(json.loads(payload.decode("utf-8")))
                except json.JSONDecodeError:
                    metadata["json_decode_error"] = True
        if current_stem is not None:
            yield current_stem, current_entry


def _looks_like_supported_image(payload: Any) -> bool:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return False
    data = bytes(payload[:16])
    return (
        data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"\x89PNG\r\n\x1a\n")
        or (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


def _datacomp_entry_to_sample(
    torch: Any, stem: str, entry: Dict[str, Any], spec: ClipContrastiveSpec
) -> Dict[str, Any]:
    sample_start = time.perf_counter()
    text = str(entry.get("text", ""))
    metadata = dict(entry.get("metadata", {}))
    image_bytes = bytes(entry["image_bytes"])
    original_width = int(
        metadata.get("original_width", spec.image_size) or spec.image_size
    )
    original_height = int(
        metadata.get("original_height", spec.image_size) or spec.image_size
    )
    decode_ms = 0.0
    tensor_ms = 0.0
    sample_image: Any = image_bytes
    image_key = "image"
    if spec.wds_image_mode != "bytes":
        Image = _require_pil_image()
        decode_start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
                    category=UserWarning,
                )
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_width, original_height = image.size
            image = image.resize((spec.image_size, spec.image_size))
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            tensor_start = time.perf_counter()
            byte_tensor = torch.frombuffer(
                bytearray(image.tobytes()), dtype=torch.uint8
            )
            sample_image = (
                byte_tensor.view(spec.image_size, spec.image_size, 3)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            tensor_ms = (time.perf_counter() - tensor_start) * 1000.0
        except Exception as exc:
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            tensor_start = time.perf_counter()
            sample_image = torch.zeros(
                spec.channels, spec.image_size, spec.image_size, dtype=torch.float32
            )
            tensor_ms = (time.perf_counter() - tensor_start) * 1000.0
            metadata["wds_image_decode_error"] = f"{type(exc).__name__}: {exc}"
        image_key = "pixel_values"
    metadata.update(
        {
            "sample_id": stem,
            "text": text,
            "original_width": int(
                metadata.get("original_width", original_width) or original_width
            ),
            "original_height": int(
                metadata.get("original_height", original_height) or original_height
            ),
            "data_source": "datacomp_wds",
            "wds_image_decode_ms": decode_ms,
            "wds_tensor_build_ms": tensor_ms,
            "wds_sample_build_ms": (time.perf_counter() - sample_start) * 1000.0,
        }
    )
    sample = {
        "input_ids": _simple_text_tokenize(text, spec),
        "attention_mask": [1] * min(max(1, len(text.split())), spec.text_length),
        "text": text,
        "height": spec.image_size,
        "width": spec.image_size,
        "patch_tokens": estimate_patch_tokens(
            spec.image_size, spec.image_size, spec.patch_size
        ),
        "metadata": metadata,
    }
    sample[image_key] = sample_image
    return sample


def _build_datacomp_metadata_dataset(
    torch: Any, spec: ClipContrastiveSpec
) -> List[Dict[str, Any]]:
    if not spec.metadata_path:
        raise ValueError("datacomp_metadata workload requires data.metadata_path.")
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError(
            "DataComp metadata training requires pandas/pyarrow."
        ) from exc
    frame = pd.read_parquet(spec.metadata_path).head(spec.num_samples)
    generator = torch.Generator()
    generator.manual_seed(spec.seed)
    dataset: List[Dict[str, Any]] = []
    patch_tokens = estimate_patch_tokens(
        spec.image_size, spec.image_size, spec.patch_size
    )
    for index, row in frame.iterrows():
        text = str(row.get("text", ""))
        image = torch.randn(
            spec.channels, spec.image_size, spec.image_size, generator=generator
        )
        dataset.append(
            {
                "pixel_values": image,
                "input_ids": _simple_text_tokenize(text, spec),
                "attention_mask": [1]
                * min(max(1, len(text.split())), spec.text_length),
                "text": text,
                "height": spec.image_size,
                "width": spec.image_size,
                "patch_tokens": patch_tokens,
                "metadata": {
                    "sample_index": int(index),
                    "uid": str(row.get("uid", "")),
                    "url": str(row.get("url", "")),
                    "data_source": "datacomp_metadata",
                },
            }
        )
    return dataset


def _split_wds_member(name: str) -> tuple[str, str]:
    path = Path(name)
    return path.stem, path.suffix.lower().lstrip(".")


def _simple_text_tokenize(text: str, spec: ClipContrastiveSpec) -> List[int]:
    tokens = []
    for raw_token in text.split()[: spec.text_length]:
        digest = hashlib.blake2b(
            raw_token.encode("utf-8", errors="ignore"), digest_size=4
        )
        tokens.append(
            int.from_bytes(digest.digest(), "little") % max(2, spec.vocab_size - 1) + 1
        )
    if not tokens:
        tokens = [1]
    return tokens
