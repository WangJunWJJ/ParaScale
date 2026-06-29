# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Checkpoint conversion planning helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .manager import CheckpointManifest


@dataclass
class CheckpointConversionPlan:
    source_format: str
    target_format: str = "parascale"
    source_path: str | None = None
    target_path: str | None = None
    steps: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_format": self.source_format,
            "target_format": self.target_format,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "steps": list(self.steps),
            "metadata": dict(self.metadata or {}),
        }


class CheckpointConverter:
    SUPPORTED_SOURCES = {"parascale", "fsdp", "deepspeed", "hf", "ascend"}
    SUPPORTED_TARGETS = {"parascale", "serve_manifest"}

    def build_plan(
        self,
        source_format: str,
        target_format: str = "parascale",
        source_path: str | None = None,
        target_path: str | None = None,
    ) -> CheckpointConversionPlan:
        source_format = source_format.lower()
        target_format = target_format.lower()
        if source_format not in self.SUPPORTED_SOURCES:
            raise ValueError(f"unsupported checkpoint source format: {source_format}")
        if target_format not in self.SUPPORTED_TARGETS:
            raise ValueError(f"unsupported checkpoint target format: {target_format}")
        metadata: Dict[str, Any] = {
            "requires_weight_rewrite": source_format != "parascale"
        }
        steps = [
            "inspect_source",
            "validate_manifest_or_state_dict",
            "write_parascale_manifest",
        ]
        if source_format != "parascale":
            steps.append("rewrite_or_reference_weight_shards")
        if target_format == "serve_manifest":
            steps.append("emit_serving_layout")
        if source_path is not None:
            metadata["source_exists"] = Path(source_path).exists()
        return CheckpointConversionPlan(
            source_format=source_format,
            target_format=target_format,
            source_path=source_path,
            target_path=target_path,
            steps=steps,
            metadata=metadata,
        )

    def convert(self, plan: CheckpointConversionPlan) -> Dict[str, Any]:
        if plan.source_format == "parascale":
            return self._convert_parascale_manifest(plan)
        if plan.source_format == "hf":
            return self._convert_hf_manifest(plan)
        return {
            "conversion_plan": plan.to_dict(),
            "converted": False,
            "reason": "conversion plan is validated; weight rewriting is not implemented in this lightweight runtime path",
        }

    def _convert_parascale_manifest(
        self, plan: CheckpointConversionPlan
    ) -> Dict[str, Any]:
        if plan.source_path is None:
            return {
                "conversion_plan": plan.to_dict(),
                "converted": False,
                "reason": "source_path is required for parascale manifest conversion",
            }
        source_path = self._resolve_manifest_path(Path(plan.source_path))
        manifest = CheckpointManifest.from_dict(
            json.loads(source_path.read_text(encoding="utf-8"))
        )
        target_path = self._target_manifest_path(plan, source_path, manifest)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        converted_manifest = manifest.to_dict()
        converted_manifest["metadata"] = dict(converted_manifest.get("metadata", {}))
        converted_manifest["metadata"]["converted_from"] = str(source_path)
        converted_manifest["metadata"]["conversion_target"] = plan.target_format

        if plan.target_format == "serve_manifest":
            converted_manifest["format"] = "parascale_serve_manifest_v1"
            converted_manifest["metadata"]["serve_ready"] = self._is_serve_ready(
                manifest
            )
            converted_manifest["metadata"]["serve_layout"] = {
                "loader": "parascale_manifest",
                "files": [dict(item) for item in manifest.files],
                "backend": manifest.backend,
            }
        else:
            converted_manifest["format"] = "parascale_manifest_v1"

        target_path.write_text(
            json.dumps(converted_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {
            "conversion_plan": plan.to_dict(),
            "converted": True,
            "source_manifest": str(source_path),
            "target_manifest": str(target_path),
            "serve_ready": converted_manifest["metadata"].get("serve_ready"),
        }

    def _convert_hf_manifest(self, plan: CheckpointConversionPlan) -> Dict[str, Any]:
        if plan.source_path is None:
            return {
                "conversion_plan": plan.to_dict(),
                "converted": False,
                "reason": "source_path is required for HF checkpoint inspection",
            }
        source_dir = Path(plan.source_path)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"HF checkpoint directory not found: {source_dir}")

        hf_config_path = source_dir / "config.json"
        hf_config: Dict[str, Any] = {}
        if hf_config_path.is_file():
            hf_config = json.loads(hf_config_path.read_text(encoding="utf-8"))

        weight_files = self._find_hf_weight_files(source_dir)
        if not weight_files:
            raise FileNotFoundError(f"no HF weight files found in: {source_dir}")

        target_path = self._target_hf_manifest_path(plan, source_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = CheckpointManifest(
            step=0,
            backend="hf",
            files=[
                {
                    "path": str(path.relative_to(source_dir)),
                    "role": "model",
                    "format": self._weight_format(path),
                    "bytes": path.stat().st_size,
                }
                for path in weight_files
            ],
            metadata={
                "source_format": "hf",
                "source_path": str(source_dir),
                "conversion_target": plan.target_format,
                "weight_rewrite_performed": False,
                "hf_config": hf_config,
            },
        )
        manifest_dict = manifest.to_dict()
        if plan.target_format == "serve_manifest":
            manifest_dict["format"] = "parascale_serve_manifest_v1"
            manifest_dict["metadata"]["serve_ready"] = True
            manifest_dict["metadata"]["serve_layout"] = {
                "loader": "hf_reference",
                "source_path": str(source_dir),
                "files": [dict(item) for item in manifest.files],
            }
        target_path.write_text(
            json.dumps(manifest_dict, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {
            "conversion_plan": plan.to_dict(),
            "converted": True,
            "source_checkpoint": str(source_dir),
            "target_manifest": str(target_path),
            "weight_files": len(weight_files),
            "weight_rewrite_performed": False,
            "serve_ready": manifest_dict["metadata"].get("serve_ready"),
        }

    @staticmethod
    def _resolve_manifest_path(path: Path) -> Path:
        if path.is_file():
            return path
        if path.is_dir() and (path / "manifest.json").is_file():
            return path / "manifest.json"
        if path.is_dir():
            manifests = sorted(path.glob("step-*/manifest.json"))
            if manifests:
                return manifests[-1]
        raise FileNotFoundError(f"checkpoint manifest not found: {path}")

    @staticmethod
    def _target_manifest_path(
        plan: CheckpointConversionPlan, source_path: Path, manifest: CheckpointManifest
    ) -> Path:
        if plan.target_path is not None:
            target = Path(plan.target_path)
            if target.suffix:
                return target
            return target / "manifest.json"
        suffix = (
            "serve-manifest.json"
            if plan.target_format == "serve_manifest"
            else "manifest.converted.json"
        )
        return source_path.parent / suffix

    @staticmethod
    def _target_hf_manifest_path(
        plan: CheckpointConversionPlan, source_dir: Path
    ) -> Path:
        if plan.target_path is not None:
            target = Path(plan.target_path)
            if target.suffix:
                return target
            return target / "manifest.json"
        suffix = (
            "serve-manifest.json"
            if plan.target_format == "serve_manifest"
            else "manifest.json"
        )
        return source_dir / ".parascale" / suffix

    @staticmethod
    def _find_hf_weight_files(source_dir: Path) -> list[Path]:
        patterns = [
            "model.safetensors",
            "*.safetensors",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
            "*.bin",
        ]
        files: list[Path] = []
        seen = set()
        for pattern in patterns:
            for path in sorted(source_dir.glob(pattern)):
                if path.is_file() and path not in seen:
                    files.append(path)
                    seen.add(path)
        return files

    @staticmethod
    def _weight_format(path: Path) -> str:
        if path.suffix == ".safetensors":
            return "safetensors"
        if path.suffix == ".bin":
            return "torch"
        return path.suffix.lstrip(".") or "unknown"

    @staticmethod
    def _is_serve_ready(manifest: CheckpointManifest) -> bool:
        return any(
            file_entry.get("role") == "backend_state" and not file_entry.get("error")
            for file_entry in manifest.files
        )
