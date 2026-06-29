# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午1:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Cache utilities for vision input pipelines."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass
class VisionMetadataCache:
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def put(self, key: str, metadata: Dict[str, Any]) -> None:
        self.entries[str(key)] = dict(metadata)

    def get(self, key: str) -> Dict[str, Any]:
        return dict(self.entries.get(str(key), {}))

    def clear(self) -> None:
        self.entries.clear()


class DiskTensorCache:
    """Small file cache for expensive CPU preprocessing outputs."""

    def __init__(self, root: str | Path | None, *, enabled: bool = True) -> None:
        self.root = Path(root).expanduser() if root else None
        self.enabled = bool(enabled and self.root)

    def key_for_paths(
        self,
        *paths: str | Path,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        hasher = hashlib.sha256()
        for path_value in paths:
            path = Path(path_value)
            hasher.update(str(path).encode("utf-8", errors="replace"))
            try:
                stat = path.stat()
                hasher.update(str(stat.st_size).encode("ascii"))
                hasher.update(str(stat.st_mtime_ns).encode("ascii"))
            except OSError:
                hasher.update(b"missing")
            hasher.update(b"\0")
        if extra:
            for key in sorted(extra):
                hasher.update(str(key).encode("utf-8", errors="replace"))
                hasher.update(b"=")
                hasher.update(str(extra[key]).encode("utf-8", errors="replace"))
                hasher.update(b"\0")
        return hasher.hexdigest()

    def path_for_key(self, key: str) -> Path:
        if self.root is None:
            raise ValueError("DiskTensorCache has no root directory.")
        return self.root / f"{key}.pt"

    def load(self, key: str, torch: Any) -> Any | None:
        if not self.enabled:
            return None
        path = self.path_for_key(key)
        try:
            if not path.exists():
                return None
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return None

    def save(self, key: str, value: Any, torch: Any) -> bool:
        if not self.enabled:
            return False
        path = self.path_for_key(key)
        tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, tmp_path)
            os.replace(tmp_path, path)
            return True
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
