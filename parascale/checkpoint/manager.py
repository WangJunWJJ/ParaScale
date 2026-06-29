# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free checkpoint metadata layer."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CheckpointManifest:
    step: int
    format: str = "parascale_manifest_v1"
    schema_version: int = 2
    global_step: int | None = None
    consumed_samples: int = 0
    consumed_tokens: int = 0
    backend: str = "unknown"
    parallel_plan: Dict[str, Any] = field(default_factory=dict)
    rng_state: Dict[str, Any] = field(default_factory=dict)
    data_state: Dict[str, Any] = field(default_factory=dict)
    files: List[Dict[str, Any]] = field(default_factory=list)
    shards: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.global_step is None:
            self.global_step = int(self.step)
        self.validate()

    def validate(self) -> None:
        if int(self.step) < 0:
            raise ValueError("CheckpointManifest.step must be non-negative.")
        if int(self.global_step or 0) < 0:
            raise ValueError("CheckpointManifest.global_step must be non-negative.")
        if int(self.consumed_samples) < 0:
            raise ValueError(
                "CheckpointManifest.consumed_samples must be non-negative."
            )
        if int(self.consumed_tokens) < 0:
            raise ValueError("CheckpointManifest.consumed_tokens must be non-negative.")
        if self.schema_version < 1:
            raise ValueError("CheckpointManifest.schema_version must be >= 1.")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "format": self.format,
            "schema_version": int(self.schema_version),
            "step": self.step,
            "global_step": int(self.global_step or self.step),
            "consumed_samples": int(self.consumed_samples),
            "consumed_tokens": int(self.consumed_tokens),
            "backend": self.backend,
            "parallel_plan": dict(self.parallel_plan),
            "rng_state": dict(self.rng_state),
            "data_state": dict(self.data_state),
            "files": [dict(item) for item in self.files],
            "shards": list(self.shards),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointManifest":
        return cls(
            step=int(data["step"]),
            format=str(data.get("format", "parascale_manifest_v1")),
            schema_version=int(data.get("schema_version", 1)),
            global_step=int(data.get("global_step", data["step"])),
            consumed_samples=int(data.get("consumed_samples", 0)),
            consumed_tokens=int(data.get("consumed_tokens", 0)),
            backend=str(
                data.get("backend", data.get("metadata", {}).get("backend", "unknown"))
            ),
            parallel_plan=dict(data.get("parallel_plan", {})),
            rng_state=dict(data.get("rng_state", {})),
            data_state=dict(data.get("data_state", {})),
            files=[dict(item) for item in data.get("files", [])],
            shards=list(data.get("shards", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CheckpointManager:
    root: str

    def manifest_path(self, step: int) -> Path:
        return Path(self.root) / f"step-{int(step):08d}" / "manifest.json"

    def payload_path(self, step: int, filename: str) -> Path:
        return Path(self.root) / f"step-{int(step):08d}" / filename

    def resolve_payload_path(
        self, manifest: CheckpointManifest, file_entry: Dict[str, Any]
    ) -> Path:
        path = Path(str(file_entry["path"]))
        if path.is_absolute():
            return path
        return self.manifest_path(manifest.step).parent / path

    def resolve_manifest_path(self, checkpoint: str | Path) -> Path:
        path = Path(checkpoint)
        if path.is_file():
            return path
        if path.is_dir() and (path / "manifest.json").is_file():
            return path / "manifest.json"
        if path.is_dir():
            manifests = sorted(path.glob("step-*/manifest.json"))
            if manifests:
                return manifests[-1]
        raise FileNotFoundError(f"checkpoint manifest not found: {checkpoint}")

    def write_manifest(self, manifest: CheckpointManifest) -> Path:
        manifest.files = [
            self._with_file_metadata(manifest, file_entry)
            for file_entry in manifest.files
        ]
        manifest.validate()
        path = self.manifest_path(manifest.step)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
        return path

    def read_manifest(self, step: int) -> CheckpointManifest:
        data = json.loads(self.manifest_path(step).read_text(encoding="utf-8"))
        return CheckpointManifest.from_dict(data)

    def read_manifest_path(self, checkpoint: str | Path) -> CheckpointManifest:
        data = json.loads(
            self.resolve_manifest_path(checkpoint).read_text(encoding="utf-8")
        )
        return CheckpointManifest.from_dict(data)

    def validate_manifest(
        self, manifest: CheckpointManifest
    ) -> "CheckpointValidationReport":
        return CheckpointValidator(self).validate(manifest)

    def validate(self, step: int) -> "CheckpointValidationReport":
        return self.validate_manifest(self.read_manifest(step))

    def _with_file_metadata(
        self, manifest: CheckpointManifest, file_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        enriched = dict(file_entry)
        if enriched.get("error") or "path" not in enriched:
            return enriched
        path = self.resolve_payload_path(manifest, enriched)
        if path.is_file():
            enriched.setdefault("entry_type", "file")
            enriched.setdefault("size_bytes", path.stat().st_size)
            enriched.setdefault("sha256", _sha256_file(path))
        elif path.is_dir():
            enriched.setdefault("entry_type", "directory")
            enriched.setdefault("exists", True)
        return enriched


@dataclass
class CheckpointValidationReport:
    ok: bool
    checked_files: int = 0
    checked_directories: int = 0
    missing: List[str] = field(default_factory=list)
    checksum_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    size_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "checked_files": int(self.checked_files),
            "checked_directories": int(self.checked_directories),
            "missing": list(self.missing),
            "checksum_mismatches": [dict(item) for item in self.checksum_mismatches],
            "size_mismatches": [dict(item) for item in self.size_mismatches],
            "errors": list(self.errors),
        }


@dataclass
class CheckpointValidator:
    manager: CheckpointManager

    def validate(self, manifest: CheckpointManifest) -> CheckpointValidationReport:
        report = CheckpointValidationReport(ok=True)
        allow_errors = bool(
            manifest.metadata.get("allow_checkpoint_error_for_benchmark")
            or manifest.metadata.get("skip_final_checkpoint")
        )
        if not allow_errors:
            if manifest.metadata.get("backend_checkpoint_error"):
                report.errors.append(
                    "backend_checkpoint_error: "
                    f"{manifest.metadata['backend_checkpoint_error']}"
                )
            if manifest.metadata.get("backend_state_written") is False:
                report.errors.append("backend_state_written is false")
        for file_entry in manifest.files:
            if file_entry.get("error"):
                if not allow_errors:
                    report.errors.append(
                        f"{file_entry.get('role', file_entry.get('path', 'entry'))}: "
                        f"{file_entry['error']}"
                    )
                continue
            if "path" not in file_entry:
                continue
            try:
                self._validate_entry(manifest, file_entry, report)
            except Exception as exc:
                report.errors.append(f"{file_entry.get('path')}: {exc}")
        report.ok = not (
            report.missing
            or report.checksum_mismatches
            or report.size_mismatches
            or report.errors
        )
        return report

    def _validate_entry(
        self,
        manifest: CheckpointManifest,
        file_entry: Dict[str, Any],
        report: CheckpointValidationReport,
    ) -> None:
        path = self.manager.resolve_payload_path(manifest, file_entry)
        relative_path = str(file_entry["path"])
        if not path.exists():
            report.missing.append(relative_path)
            return
        if path.is_dir():
            report.checked_directories += 1
            return
        if not path.is_file():
            report.errors.append(f"{relative_path}: unsupported checkpoint entry type")
            return

        report.checked_files += 1
        expected_size = file_entry.get("size_bytes")
        if expected_size is not None and int(expected_size) != path.stat().st_size:
            report.size_mismatches.append(
                {
                    "path": relative_path,
                    "expected": int(expected_size),
                    "actual": path.stat().st_size,
                }
            )
        expected_sha = file_entry.get("sha256")
        if expected_sha is not None:
            actual_sha = _sha256_file(path)
            if actual_sha != expected_sha:
                report.checksum_mismatches.append(
                    {
                        "path": relative_path,
                        "expected": expected_sha,
                        "actual": actual_sha,
                    }
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
