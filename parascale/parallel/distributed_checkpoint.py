# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Committed rank-local checkpoints for explicit distributed stage execution."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from parascale.contracts.checkpoint import CheckpointRuntime

from .checkpoint_context import canonical_checkpoint_context


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous manifests instead of accepting the last duplicate key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Checkpoint manifest contains duplicate key: {key}")
        result[key] = value
    return result


@contextmanager
def _phase(name: str):
    """Exchange local exceptions before any rank enters the next phase."""
    import torch.distributed as dist

    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(
        errors, None if error is None else f"{type(error).__name__}: {error}"
    )
    failures = [
        f"rank {rank}: {message}" for rank, message in enumerate(errors) if message
    ]
    if failures:
        raise RuntimeError(
            f"Checkpoint {name} failed: {'; '.join(failures)}"
        ) from error


def _fsync_directory(directory: Path) -> None:
    # Windows does not expose directory descriptors through os.open.
    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            raise


def _agree_identity(identity: dict, operation: str) -> None:
    """Reject mixed destinations/snapshots before writing or applying state."""
    import torch.distributed as dist

    identities = [None] * dist.get_world_size()
    dist.all_gather_object(identities, identity)
    if any(candidate != identity for candidate in identities):
        raise RuntimeError(f"Checkpoint {operation} identity mismatch across ranks")


def _validate_execution(trainer: Any) -> None:
    import torch.distributed as dist

    trainer.config.validate_world_size(dist.get_world_size())
    if type(trainer.rank) is not int or trainer.rank != dist.get_rank():
        raise ValueError("Checkpoint rank does not match process group")


def _optimizer_identity(module: Any, optimizer: Any) -> dict[str, Any]:
    """Bind optimizer slots to ordered stage parameter names, including subsets.

    Shared parameters use their first registered name. External or repeated
    parameters cannot be mapped unambiguously and are rejected before saving.
    Hyperparameters are intentionally excluded: resume restores their saved values.
    """
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    seen, groups = set(), []
    for group in optimizer.param_groups:
        parameters = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in names or identity in seen:
                raise ValueError(
                    "Checkpoint optimizer has external or duplicate parameters"
                )
            seen.add(identity)
            parameters.append(names[identity])
        groups.append(parameters)
    cls = type(optimizer)
    return {"type": f"{cls.__module__}.{cls.__qualname__}", "parameter_groups": groups}


def _runtime_identity(runtime: CheckpointRuntime | None, optimizer: Any) -> str | None:
    if runtime is None:
        return None
    identity = runtime.identity(optimizer)
    serialized = json.dumps(identity, sort_keys=True, allow_nan=False)
    if (
        not isinstance(identity, dict)
        or not identity
        or json.loads(serialized) != identity
    ):
        raise ValueError("Checkpoint runtime identity must be a nonempty JSON object")
    return serialized


def _validate_manifest(
    metadata: Any, trainer: Any, runtime_identity: str | None = None
) -> None:
    keys = {
        "schema_version",
        "topology",
        "step",
        "files",
        "context",
    }
    if runtime_identity is not None:
        keys.add("runtime")
    if not isinstance(metadata, dict) or set(metadata) != keys:
        raise ValueError("Checkpoint manifest schema mismatch")
    topology = metadata["topology"]
    expected = asdict(trainer.config)
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != (5 if runtime_identity is not None else 4)
        or not isinstance(topology, dict)
        or topology != expected
        or any(
            type(topology[key]) is not type(value) for key, value in expected.items()
        )
    ):
        raise ValueError("Checkpoint topology/schema does not match execution")
    if (
        runtime_identity is not None
        and json.dumps(metadata["runtime"], sort_keys=True, allow_nan=False)
        != runtime_identity
    ):
        raise ValueError("Checkpoint runtime identity mismatch")
    if canonical_checkpoint_context(
        metadata["context"]
    ) != canonical_checkpoint_context(getattr(trainer, "checkpoint_context", None)):
        raise ValueError("Checkpoint training context mismatch")
    if type(metadata["step"]) is not int or metadata["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    entries = metadata["files"]
    if not isinstance(entries, list) or len(entries) != trainer.config.world_size:
        raise ValueError("Checkpoint rank count mismatch")
    for rank, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or entry["path"] != f"rank-{rank:05d}.pt"
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in entry["sha256"])
        ):
            raise ValueError("Checkpoint rank entry schema mismatch")


def _validate_optimizer_binding(state: dict, identity: Any) -> None:
    """Validate ordered parameter names and unique serialized optimizer slots."""
    if (
        not isinstance(identity, dict)
        or set(identity) != {"type", "parameter_groups"}
        or not isinstance(identity["type"], str)
        or not identity["type"]
        or not isinstance(identity["parameter_groups"], list)
        or len(identity["parameter_groups"]) != len(state["param_groups"])
    ):
        raise ValueError("Checkpoint optimizer identity schema mismatch")
    names, slots = set(), set()
    for group, parameters in zip(state["param_groups"], identity["parameter_groups"]):
        if (
            not isinstance(group, dict)
            or not isinstance(group.get("params"), list)
            or not isinstance(parameters, list)
            or len(group["params"]) != len(parameters)
        ):
            raise ValueError("Checkpoint optimizer identity group mismatch")
        for slot, name in zip(group["params"], parameters):
            if (
                type(slot) is not int
                or slot < 0
                or slot in slots
                or not isinstance(name, str)
                or not name
                or name in names
            ):
                raise ValueError(
                    "Checkpoint optimizer identity has invalid parameter slots"
                )
            slots.add(slot)
            names.add(name)
    if any(type(slot) is not int or slot not in slots for slot in state["state"]):
        raise ValueError("Checkpoint optimizer state has unbound parameter slots")


def _validate_payload(payload: Any, metadata: dict, device: Any) -> None:
    import torch

    keys = {
        "model",
        "optimizer",
        "optimizer_identity",
        "cpu_rng",
        "cuda_rng",
        "step",
    }
    if metadata["schema_version"] == 5:
        keys.add("runtime")
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Checkpoint payload schema mismatch")
    if type(payload["step"]) is not int or payload["step"] != metadata["step"]:
        raise ValueError("Checkpoint step mismatch")
    if not isinstance(payload["model"], Mapping):
        raise ValueError("Checkpoint model state must be a mapping")
    state = payload["optimizer"]
    if (
        not isinstance(state, dict)
        or set(state) != {"state", "param_groups"}
        or not isinstance(state["state"], dict)
        or not isinstance(state["param_groups"], list)
    ):
        raise ValueError("Checkpoint optimizer schema mismatch")
    _validate_optimizer_binding(state, payload["optimizer_identity"])
    # Validate RNG states on private generators without changing global RNG.
    torch.Generator(device="cpu").set_state(payload["cpu_rng"])
    if device.type == "cuda":
        torch.Generator(device=device).set_state(payload["cuda_rng"])
    elif payload["cuda_rng"] is not None:
        raise ValueError("CPU checkpoint must not contain CUDA RNG state")


def save_checkpoint(
    trainer: Any,
    directory: Any,
    optimizer: Any,
    *,
    step: int,
    runtime: CheckpointRuntime | None = None,
) -> None:
    """Publish a manifest only after every rank's atomic file is complete.

    A path is immutable after commit. This is a trusted, same-topology Python
    checkpoint, not a portable inference export. An explicit runtime participant
    includes training progress in schema5; without one, schema4 stores only
    model/optimizer/Torch RNG and must not be treated as full training recovery.
    All ranks must participate; concurrent writers to one directory are unsupported.
    Process death or failures inside native state-dict collectives require job-level
    teardown; Python exception coordination cannot recover a broken process group.
    """
    import torch
    import torch.distributed as dist

    with _phase("save preflight"):
        _validate_execution(trainer)
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        directory = Path(directory).resolve()
        manifest = directory / "manifest.json"
        if manifest.exists():
            raise FileExistsError(f"Checkpoint already committed: {directory}")
        device = torch.device(trainer.device)
        path = directory / f"rank-{trainer.rank:05d}.pt"
        context = canonical_checkpoint_context(
            getattr(trainer, "checkpoint_context", None)
        )
        optimizer_identity = _optimizer_identity(trainer.module, optimizer)
        optimizer_state = optimizer.state_dict()
        _validate_optimizer_binding(optimizer_state, optimizer_identity)
        runtime_identity = _runtime_identity(runtime, optimizer)
        if runtime is not None:
            runtime_state = runtime.capture()
            runtime.validate(runtime_state, step=step, optimizer_state=optimizer_state)
    _agree_identity(
        dict(
            directory=str(directory),
            topology=asdict(trainer.config),
            step=step,
            context=context,
            runtime=runtime_identity,
        ),
        "save",
    )
    with _phase("directory creation"):
        directory.mkdir(parents=True, exist_ok=True)
        _fsync_directory(directory.parent)
    with _phase("state capture"):
        payload = {
            "model": trainer.module.state_dict(),
            "optimizer": optimizer_state,
            "optimizer_identity": optimizer_identity,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
            "step": step,
        }
        if runtime is not None:
            payload["runtime"] = runtime_state
    with _phase("rank write"):
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(directory)
        entry = {"path": path.name, "sha256": _digest(path)}
    entries = [None] * dist.get_world_size()
    dist.all_gather_object(entries, entry)
    with _phase("manifest publication"):
        metadata = dict(
            schema_version=5 if runtime is not None else 4,
            topology=asdict(trainer.config),
            step=step,
            files=entries,
            context=json.loads(context),
        )
        if runtime_identity is not None:
            metadata["runtime"] = json.loads(runtime_identity)
        _validate_manifest(metadata, trainer, runtime_identity)
        if trainer.rank == 0:
            temporary_manifest = manifest.with_suffix(".tmp")
            with temporary_manifest.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_manifest, manifest)
            _fsync_directory(directory)


def load_checkpoint(
    trainer: Any,
    directory: Any,
    optimizer: Any,
    *,
    runtime: CheckpointRuntime | None = None,
) -> int:
    """Load trusted same-topology state after coordinated local validation.

    Native state-dict APIs are used without a partial-state fallback. An error
    during application is reported on every rank, but cannot be rolled back.
    """
    import torch

    with _phase("load preflight"):
        _validate_execution(trainer)
        device = torch.device(trainer.device)
        directory = Path(directory).resolve()
        metadata = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
        runtime_identity = _runtime_identity(runtime, optimizer)
        _validate_manifest(metadata, trainer, runtime_identity)
    _agree_identity(
        dict(
            directory=str(directory),
            manifest=json.dumps(metadata, sort_keys=True, allow_nan=False),
        ),
        "load",
    )
    with _phase("payload read"):
        entry = metadata["files"][trainer.rank]
        path = directory / entry["path"]
        if not path.is_file() or _digest(path) != entry["sha256"]:
            raise ValueError("Checkpoint rank file missing or checksum mismatch")
        payload = torch.load(path, weights_only=False)
        _validate_payload(payload, metadata, device)
        if payload["optimizer_identity"] != _optimizer_identity(
            trainer.module, optimizer
        ):
            raise ValueError("Checkpoint optimizer identity mismatch")
        if runtime is not None:
            runtime.validate(
                payload["runtime"],
                step=payload["step"],
                optimizer_state=payload["optimizer"],
            )
    with _phase("state inspection"):
        current = trainer.module.state_dict()
    with _phase("state validation"):
        if set(current) != set(payload["model"]):
            raise ValueError("Checkpoint model keys mismatch")
        for key, value in current.items():
            saved = payload["model"][key]
            if isinstance(value, torch.Tensor) and (
                not isinstance(saved, torch.Tensor)
                or saved.shape != value.shape
                or saved.dtype != value.dtype
            ):
                raise ValueError(f"Checkpoint model tensor mismatch: {key}")
        groups = payload["optimizer"]["param_groups"]
        if len(groups) != len(optimizer.param_groups) or any(
            len(saved["params"]) != len(live["params"])
            for saved, live in zip(groups, optimizer.param_groups)
        ):
            raise ValueError("Checkpoint optimizer parameter groups mismatch")
    with _phase("model application"):
        trainer.module.load_state_dict(payload["model"], strict=True)
    with _phase("optimizer application"):
        optimizer.load_state_dict(payload["optimizer"])
    with _phase("runtime application"):
        if runtime is not None:
            runtime.restore(payload["runtime"])
    with _phase("RNG application"):
        torch.set_rng_state(payload["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"], device)
    return payload["step"]
