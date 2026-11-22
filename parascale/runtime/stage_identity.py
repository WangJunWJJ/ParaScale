# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Detached effective-model and local-asset identity for stage recovery."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from parascale.parallel.checkpoint_context import canonical_checkpoint_context
from parascale.parallel.checkpoint_manifest import unique_json_object
from parascale.parallel.distributed_checkpoint import _digest


def _model_assets(root: Path) -> dict[str, str]:
    """Resolve HF's default local weight preference, then stream each file once.

    Trusted local assets may be symlinks (including HF cache blob links outside
    root). Their targets must remain immutable during loading and fingerprinting;
    this is an identity check, not a filesystem sandbox for untrusted assets.
    """
    config = root / "config.json"
    if not root.is_dir() or not config.is_file():
        raise ValueError("Stage model requires a local directory with config.json")
    assets = {"config.json": config}
    # Match from_pretrained(use_safetensors=None), without guessing shard names.
    for name in (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
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
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("empty or invalid weight_map")
                for shard in weight_map.values():
                    if (
                        not isinstance(shard, str) or not shard
                        or Path(shard).is_absolute() or Path(shard).drive
                        or ".." in Path(shard).parts
                    ):
                        raise ValueError("invalid weight shard path")
                    path = root / shard
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"missing or empty weight shard: {shard}")
                    assets[shard] = path
            except (ValueError, OSError) as error:
                raise ValueError(f"Invalid model weight index {name}: {error}") from error
        elif candidate.stat().st_size == 0:
            raise ValueError(f"Empty model weights: {name}")
        break
    else:
        raise ValueError("No supported local model weights or weight index found")
    return {name: _digest(path) for name, path in sorted(assets.items())}


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = set((root / "parallel").rglob("*.py"))
    paths.update(root / relative for relative in (
        "workloads/causal_lm.py",
        "data/text/__init__.py", "data/text/artifact.py", "data/text/batches.py",
        "optimizers/__init__.py", "optimizers/factory.py", "optimizers/spec.py",
        "optimizers/schedule.py", "runtime/stage_identity.py", "runtime/stage_plan.py",
        "runtime/training/data_state.py", "runtime/training/random_state.py",
        "runtime/training/stage_checkpoint.py",
    ))
    for relative in ("runtime/stage_train_runner.py", "runtime/training/stage_loop.py"):
        if (root / relative).is_file():
            paths.add(root / relative)
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(paths)
    }


def build_stage_checkpoint_context(stage: Any, cursor: Any, plan: Any) -> dict[str, Any]:
    """Snapshot rank-independent semantics; never retain modules or load tensors.

    Model assets must remain immutable between source loading and this snapshot.
    Only the default local HF safetensors/bin formats admitted by the workload
    are supported. Topology and cursor position belong to the checkpoint runtime.
    """
    import torch
    import transformers

    config = getattr(stage, "effective_model_config", None)
    if not isinstance(config, dict) or not config:
        raise ValueError("Stage requires a nonempty effective model config")
    # Checkpoint loading restores saved hyperparameters; bind requested values too.
    optimizer = json.loads(
        json.dumps(asdict(plan.optimizer_spec), sort_keys=True, allow_nan=False)
    )
    context = {
        "optimizer": optimizer,
        "model": {
            "config": config,
            "assets_sha256": _model_assets(Path(plan.workload.model_path)),
        },
        "tokenizer": cursor.corpus.manifest["tokenizer"],
        "data": cursor.corpus.identity,
        "loss": {
            "type": "causal_next_token_v1",
            "shift": 1,
            "ignore_index": -100,
            "vocab_size": stage.vocab_size,
        },
        "precision": plan.config.precision,
        "execution": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "implementation_sha256": _implementation_hashes(),
        },
    }
    return json.loads(canonical_checkpoint_context(context))
