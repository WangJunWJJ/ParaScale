# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:21
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free identity of trusted local causal-LM model assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from parascale.parallel.checkpoint_manifest import unique_json_object


def file_sha256(path: str | Path) -> str:
    """Hash a local file with bounded one-MiB reads, without deserialization."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_asset_hashes(root: Path, *, include_weights: bool = True) -> dict[str, str]:
    """Resolve HF's default local weight preference, then stream each file once.

    Trusted local assets may be symlinks (including HF cache blob links outside
    root). Their targets must remain immutable during loading and fingerprinting;
    this is an identity check, not a filesystem sandbox for untrusted assets.
    Config-only artifact reloads can omit float weights with include_weights=False.
    """
    config = root / "config.json"
    if not root.is_dir() or not config.is_file():
        raise ValueError("Stage model requires a local directory with config.json")
    assets = {"config.json": config}
    if not include_weights:
        return {"config.json": file_sha256(config)}
    # Match from_pretrained(use_safetensors=None), without guessing shard names.
    for name in (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        candidate = root / name
        if not candidate.is_file():
            continue
        assets[name] = candidate
        if name.endswith(".index.json"):
            try:
                index = json.loads(
                    candidate.read_text(encoding="utf-8"),
                    object_pairs_hook=unique_json_object,
                )
                weight_map = (
                    index.get("weight_map") if isinstance(index, dict) else None
                )
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("empty or invalid weight_map")
                for shard in weight_map.values():
                    if (
                        not isinstance(shard, str)
                        or not shard
                        or Path(shard).is_absolute()
                        or Path(shard).drive
                        or ".." in Path(shard).parts
                    ):
                        raise ValueError("invalid weight shard path")
                    path = root / shard
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"missing or empty weight shard: {shard}")
                    assets[shard] = path
            except (ValueError, OSError) as error:
                raise ValueError(
                    f"Invalid model weight index {name}: {error}"
                ) from error
        elif candidate.stat().st_size == 0:
            raise ValueError(f"Empty model weights: {name}")
        break
    else:
        raise ValueError("No supported local model weights or weight index found")
    return {name: file_sha256(path) for name, path in sorted(assets.items())}
