# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:55
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Communication profile container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class CommunicationProfile:
    all_reduce_ms: float = 0.0
    reduce_scatter_ms: float = 0.0
    all_gather_ms: float = 0.0
    bucket_count: int = 0
    bucket_bytes: int = 0
    overlap_ratio: float = 0.0

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any]) -> "CommunicationProfile":
        return cls(
            all_reduce_ms=float(metrics.get("comm_all_reduce_ms", 0.0) or 0.0),
            reduce_scatter_ms=float(metrics.get("comm_reduce_scatter_ms", 0.0) or 0.0),
            all_gather_ms=float(metrics.get("comm_all_gather_ms", 0.0) or 0.0),
            bucket_count=int(metrics.get("comm_bucket_count", 0) or 0),
            bucket_bytes=int(metrics.get("comm_bucket_bytes", 0) or 0),
            overlap_ratio=float(metrics.get("comm_overlap_ratio", 0.0) or 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comm_all_reduce_ms": self.all_reduce_ms,
            "comm_reduce_scatter_ms": self.reduce_scatter_ms,
            "comm_all_gather_ms": self.all_gather_ms,
            "comm_bucket_count": self.bucket_count,
            "comm_bucket_bytes": self.bucket_bytes,
            "comm_overlap_ratio": self.overlap_ratio,
        }
