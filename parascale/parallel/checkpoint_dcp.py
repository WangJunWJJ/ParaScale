# -*- coding: utf-8 -*-
# @Time : 2026/11/25 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""DCP storage for exact plain-tensor DP replicas and rank-local recovery state."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

from parascale.checkpoint.io import atomic_binary_writer

from .checkpoint_catalog import _safe_path
from .distributed_checkpoint import _agree_identity, _digest, _fsync_directory, _phase


def _map_tree(value: Any, tensor_map: Any, *, metadata: bool = False) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        if metadata:
            raise ValueError("DCP metadata attributes cannot contain tensors")
        if (
            type(value) is not torch.Tensor
            or value.layout != torch.strided
            or value.is_quantized
        ):
            raise ValueError("DCP requires plain strided tensors")
        return tensor_map(value)
    if type(value) in (dict, OrderedDict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError("DCP state keys must be strings or integers")
        attributes = getattr(value, "__dict__", {})
        if set(attributes) - {"_metadata"}:
            raise ValueError("DCP does not support custom mapping attributes")
        result = type(value)(
            (key, _map_tree(item, tensor_map, metadata=metadata))
            for key, item in value.items()
        )
        if "_metadata" in attributes:
            result._metadata = _map_tree(value._metadata, tensor_map, metadata=True)
        return result
    if type(value) in (tuple, list):
        return type(value)(
            _map_tree(item, tensor_map, metadata=metadata) for item in value
        )
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"DCP does not support state type/value: {type(value).__name__}")


def _copy_tree(value: Any, device: str) -> Any:
    """Build tensor targets without copying data; preserve container metadata."""
    import torch

    return _map_tree(value, lambda tensor: torch.empty_like(tensor, device=device))


def _describe(value: Any, *, devices: bool = False) -> tuple[str, list]:
    import torch

    value = _map_tree(value, lambda tensor: tensor)
    records, tensors = [], []

    def visit(item, path):
        if isinstance(item, torch.Tensor):
            records.append(
                (
                    path,
                    "tensor",
                    str(item.dtype),
                    list(item.shape),
                    item.device.type if devices else None,
                )
            )
            tensors.append(item)
        elif type(item) in (dict, OrderedDict):
            records.append((path, type(item).__name__, len(item)))
            if hasattr(item, "_metadata"):
                visit(item._metadata, (*path, "attribute:_metadata"))
            for key in sorted(item, key=repr):
                visit(item[key], (*path, f"key:{type(key).__name__}:{key!r}"))
        elif type(item) in (list, tuple):
            records.append((path, type(item).__name__, len(item)))
            for index, child in enumerate(item):
                visit(child, (*path, f"index:{index}"))
        else:
            records.append((path, type(item).__name__, item))

    visit(value, ())
    return json.dumps(records, allow_nan=False), tensors


def validate_destination(directory: Path) -> None:
    """Reject linked ancestors and nonempty uncommitted destinations."""
    if ".." in directory.parts:
        raise ValueError("DCP checkpoint parent traversal is not allowed")
    ancestor = directory.absolute()
    while not ancestor.exists() and not ancestor.is_symlink():
        ancestor = ancestor.parent
    _safe_path(ancestor)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("DCP checkpoint destination must be new or empty")


def _prepare_replica(tensor, device, scratch, comparison, is_source):
    import torch

    # Numeric equality would merge distinct representations such as signed zero.
    local = tensor.to(device).contiguous().reshape(-1).view(torch.uint8)
    incoming = scratch[: local.numel()]
    mask = comparison[: local.numel()]
    if is_source:
        incoming.copy_(local)
    return local, incoming, mask


def _verify_replicas(trainer: Any, common: dict) -> None:
    import torch
    import torch.distributed as dist

    with _phase("DCP replica preflight"):
        signature, tensors = _describe(common, devices=True)
        if not tensors or any(
            tensor.device.type not in ("cpu", "cuda") for tensor in tensors
        ):
            raise ValueError("DCP replicas require materialized CPU/CUDA tensors")
        group = (
            trainer.mesh["dp"].get_group()
            if trainer.config.pp_size > 1
            else dist.group.WORLD
        )
        if dist.get_world_size(group) != trainer.config.dp_size:
            raise ValueError("DCP replica group does not match topology")
        source = dist.get_global_rank(group, 0)
        is_source = dist.get_rank() == source
        tensors = [tensor for tensor in tensors if tensor.numel()]
        counts = [None] * dist.get_world_size()
        signatures = [None] * dist.get_world_size(group)
        device = torch.device(trainer.device)
        scratch = torch.empty(
            max((t.numel() * t.element_size() for t in tensors), default=0),
            dtype=torch.uint8,
            device=device,
        )
        comparison = torch.empty(
            scratch.numel(),
            dtype=torch.bool,
            device=device,
        )
    with _phase("DCP replica metadata"):
        dist.all_gather_object(signatures, signature, group=group)
        if any(item != signature for item in signatures):
            raise ValueError("DCP replica metadata differs")
    dist.all_gather_object(counts, len(tensors))
    # Pipeline stages may have different tensor counts, but share the error phases.
    with torch.no_grad():
        for index in range(max(counts)):
            active = index < len(tensors)
            with _phase("DCP replica preparation"):
                if active:
                    local, incoming, mask = _prepare_replica(
                        tensors[index], device, scratch, comparison, is_source
                    )
            with _phase("DCP replica comparison"):
                if active:
                    dist.broadcast(incoming, src=source, group=group)
                    torch.eq(local, incoming, out=mask)
                    if not mask.all().item():
                        raise ValueError("DCP replica tensors differ")


def _call_dcp(function: Any, *args: Any, **kwargs: Any) -> Any:
    from torch.distributed.checkpoint.api import CheckpointException

    try:
        return function(*args, **kwargs)
    except CheckpointException as error:
        raise RuntimeError(f"DCP operation failed: {error}") from error


def save_state(trainer: Any, directory: Path, payload: dict) -> list[dict]:
    """Write shared state and per-rank templates; caller publishes the manifest."""
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    with _phase("DCP state preparation"):
        common = {key: payload[key] for key in ("model", "optimizer")}
        local = {key: value for key, value in payload.items() if key not in common}
        local["common_template"] = _copy_tree(common, "meta")
        stage = trainer.config.coordinates(trainer.rank)[1]
        writer = dcp.FileSystemWriter(
            directory / "shared", sync_files=True, overwrite=False
        )
    _verify_replicas(trainer, common)
    with _phase("DCP serialization"):
        _call_dcp(
            dcp.save,
            {f"stage_{stage}": common},
            storage_writer=writer,
            planner=dcp.DefaultSavePlanner(),
        )
    with _phase("DCP rank state"):
        _fsync_directory(directory / "shared")
        with atomic_binary_writer(directory / f"rank-{trainer.rank:05d}.pt") as stream:
            torch.save(local, stream)
    with _phase("DCP inventory"):
        _safe_path(directory / "shared")
        paths = sorted(
            [directory / f"rank-{rank:05d}.pt" for rank in range(dist.get_world_size())]
            + list((directory / "shared").iterdir())
        )
        names = [path.relative_to(directory).as_posix() for path in paths]
        for path in paths:
            if not _safe_path(path).is_file():
                raise ValueError("DCP inventory requires regular files")
    _agree_identity({"files": names}, "DCP inventory")
    with _phase("DCP checksums"):
        entries = [
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": _digest(path),
                "size_bytes": path.stat().st_size,
            }
            for path in paths[trainer.rank :: dist.get_world_size()]
        ]
    all_entries = [None] * dist.get_world_size()
    dist.all_gather_object(all_entries, entries)
    return sorted(
        [entry for group in all_entries for entry in group], key=lambda row: row["path"]
    )


def _validate_tensor_index(target: dict, index: Any) -> None:
    import torch
    from torch.distributed.checkpoint import DefaultLoadPlanner, TensorStorageMetadata

    planner = DefaultLoadPlanner()
    planner.set_up_planner(target, index)
    for key, tensor in planner.state_dict.items():
        if isinstance(tensor, torch.Tensor):
            entry = index.state_dict_metadata.get(key)
            if (
                not isinstance(entry, TensorStorageMetadata)
                or entry.properties.dtype != tensor.dtype
                or entry.size != tensor.size()
            ):
                raise ValueError(f"DCP index tensor mismatch: {key}")


def load_state(trainer: Any, directory: Path, metadata: dict) -> dict:
    """Verify all files, load fresh CPU targets, retain module version metadata."""
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    with _phase("DCP integrity"):
        for entry in metadata["files"][trainer.rank :: dist.get_world_size()]:
            path = _safe_path(directory / entry["path"])
            if (
                not path.is_file()
                or path.stat().st_size != entry["size_bytes"]
                or _digest(path) != entry["sha256"]
            ):
                raise ValueError("DCP checkpoint file size/checksum mismatch")
    with _phase("DCP read preparation"):
        local = torch.load(
            directory / f"rank-{trainer.rank:05d}.pt",
            weights_only=False,
            map_location="cpu",
        )
        if not isinstance(local, dict) or "common_template" not in local:
            raise ValueError("DCP rank state is missing its template")
        template = local.pop("common_template")
        if not isinstance(template, dict) or set(template) != {"model", "optimizer"}:
            raise ValueError("DCP common template schema mismatch")
        signature, tensors = _describe(template)
        if any(tensor.device.type != "meta" for tensor in tensors):
            raise ValueError("DCP template must not contain materialized tensor data")
        common = _copy_tree(template, "cpu")
        reader = dcp.FileSystemReader(directory / "shared")
        index = reader.read_metadata()
        files = {
            entry["path"][len("shared/") :]: entry["size_bytes"]
            for entry in metadata["files"]
            if entry["path"].endswith(".distcp")
        }
        for item in index.storage_data.values():
            if (
                item.relative_path not in files
                or type(item.offset) is not int
                or type(item.length) is not int
                or item.offset < 0
                or item.length < 0
                or item.offset + item.length > files[item.relative_path]
            ):
                raise ValueError(
                    "DCP metadata references undeclared/out-of-bounds storage"
                )
        stage = trainer.config.coordinates(trainer.rank)[1]
        target = {f"stage_{stage}": common}
        _validate_tensor_index(target, index)
    with _phase("DCP loading"):
        _call_dcp(
            dcp.load,
            target,
            storage_reader=reader,
            planner=dcp.DefaultLoadPlanner(),
        )
    with _phase("DCP loaded structure"):
        if _describe(common)[0] != signature:
            raise ValueError("DCP loaded state differs from its rank template")
        if set(local) & set(common):
            raise ValueError("DCP rank/common state overlap")
    return {**local, **common}
