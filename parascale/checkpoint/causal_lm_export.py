# -*- coding: utf-8 -*-
# @Time : 2026/11/25 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit full native checkpoint publication as a standalone HF student."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from parascale.checkpoint.io import atomic_binary_writer
from parascale.checkpoint.manager import CheckpointManager
from parascale.checkpoint.recovery import _validate_training_contract
from parascale.workloads.causal_lm_assets import file_sha256


def _load_student(manager, manifest, model_path):
    import torch

    from parascale.workloads.causal_lm_model import (
        causal_lm_config_identity,
        load_causal_lm_model,
    )

    objective = manifest.metadata.get("objective", {}).get("workload", {})
    if objective.get("type") != "causal_next_token_v1":
        raise ValueError("export requires a causal_lm training objective")
    attention = objective.get("attention_backend")
    if attention not in {"sdpa", "eager"}:
        raise ValueError("unsupported checkpoint attention backend")
    model = load_causal_lm_model(
        model_path, from_config=True, attention_backend=attention
    )
    if model.config.model_type != "llama":
        raise ValueError(
            "standalone product inference currently requires a Llama model"
        )
    if objective.get("model_config_sha256") != causal_lm_config_identity(model.config):
        raise ValueError("model configuration does not match checkpoint")
    if objective.get("tokenizer_identity") != model.tokenizer_identity:
        raise ValueError("tokenizer identity does not match checkpoint")
    entry = next(item for item in manifest.files if item.get("role") == "backend_state")
    path = manager.resolve_payload_path(manifest, entry)
    from parascale.optimizers.torchao_adapter import prepare_checkpoint

    prepare_checkpoint(manifest.metadata.get("optimizer", {}))
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    backend = payload.get("backend_state", {})
    if (
        backend.get("backend") != manifest.backend
        or backend.get("adapter_only_checkpoint")
        or backend.get("adapter_state_dict") is not None
    ):
        raise ValueError("export requires full native student state")
    state = backend.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("missing model state")
    if manifest.backend == "native_ddp":
        if not all(isinstance(key, str) and key.startswith("module.") for key in state):
            raise ValueError("DDP state must have consistent module prefixes")
        state = {key.removeprefix("module."): value for key, value in state.items()}
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("student state keys do not match the model")
    for key, value in state.items():
        target = expected[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"invalid student tensor: {key}")
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        other = aliases.setdefault(id(parameter), name)
        if not torch.equal(state[name], state[other]):
            raise ValueError(f"inconsistent tied weights: {name}, {other}")
    model.load_state_dict(state, strict=True)
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("checkpoint changed during export")
    return model.eval(), entry["sha256"]


def export_causal_lm_checkpoint(
    checkpoint: str | Path, model_path: str | Path, output: str | Path
) -> dict[str, Any]:
    """Export trusted immutable native/native_ddp full student state on CPU.

    Args:
        checkpoint: Explicit manifest, step directory, or checkpoint root.
        model_path: Matching local config/tokenizer; original weights are unused.
        output: New directory owned by this operation, outside the input trees.

    Returns:
        Release provenance and file hashes, also stored as release.json.

    Raises:
        ValueError: Invalid, corrupt, unsupported or incompatible checkpoint.
        FileExistsError: Destination exists or another exporter owns its lock.
        OSError: Read, serialization or publication failure. No partial release
            is published before rename; inputs and existing outputs are untouched.

    Callers must keep inputs immutable and exclusively own the destination.
    The lock coordinates ParaScale exporters, not arbitrary external writers.
    """
    from transformers import AutoTokenizer

    from parascale.data.text import tokenizer_fingerprint
    from parascale.workloads.causal_lm_model import load_causal_lm_model

    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    locator = CheckpointManager(str(checkpoint))
    manifest_path = locator.resolve_manifest_path(checkpoint)
    manager = CheckpointManager(str(manifest_path.parent.parent))
    manifest_digest = file_sha256(manifest_path)
    manifest = manager.read_manifest_path(manifest_path)
    _validate_training_contract(manager, manifest, manifest.step)
    validation = manager.validate_manifest(manifest)
    if not validation.ok:
        raise ValueError(f"checkpoint integrity failed: {validation.to_dict()}")
    for source in (Path(model_path).resolve(), Path(manager.root).resolve()):
        if output.resolve().is_relative_to(source) or source.is_relative_to(
            output.resolve()
        ):
            raise ValueError("output must not overlap model/checkpoint inputs")
    model, checkpoint_digest = _load_student(manager, manifest, model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer_fingerprint(tokenizer) != model.tokenizer_identity:
        raise ValueError("tokenizer changed during export")
    if file_sha256(manifest_path) != manifest_digest:
        raise ValueError("checkpoint manifest changed during export")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}-export.lock")
    with lock.open("x", encoding="utf-8") as lock_stream:
        try:
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            with tempfile.TemporaryDirectory(
                prefix=f".{output.name}-", dir=output.parent
            ) as temporary:
                staging = Path(temporary) / "model"
                model.model.save_pretrained(
                    staging, safe_serialization=True, max_shard_size="2GB"
                )
                tokenizer.save_pretrained(staging)
                # Verify serialization and shared weights before publication.
                restored = load_causal_lm_model(staging, attention_backend="eager")
                import torch

                restored_state = restored.state_dict()
                if restored.tokenizer_identity != model.tokenizer_identity or any(
                    not torch.equal(value, restored_state[name])
                    for name, value in model.state_dict().items()
                ):
                    raise ValueError("serialized student failed exact roundtrip")
                del restored_state, restored
                files = []
                for path in sorted(staging.iterdir()):
                    if not path.is_file() or path.is_symlink():
                        raise ValueError("release must contain only regular files")
                    with path.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    files.append(
                        {
                            "path": path.name,
                            "size_bytes": path.stat().st_size,
                            "sha256": file_sha256(path),
                        }
                    )
                report = {
                    "format": "parascale.causal_lm.release.v1",
                    "step": manifest.step,
                    "source_manifest_sha256": manifest_digest,
                    "source_state_sha256": checkpoint_digest,
                    "tokenizer_identity": model.tokenizer_identity,
                    "model_config_sha256": manifest.metadata["objective"]["workload"][
                        "model_config_sha256"
                    ],
                    "files": files,
                }
                with atomic_binary_writer(staging / "release.json") as stream:
                    stream.write(
                        (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(
                            "utf-8"
                        )
                    )
                if output.exists() or output.is_symlink():
                    raise FileExistsError(output)
                staging.rename(output)
                return report
        finally:
            lock_stream.close()
            lock.unlink(missing_ok=True)
