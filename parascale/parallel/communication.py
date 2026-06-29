# -*- coding: utf-8 -*-
# @Time : 2026/6/9 下午7:17
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Experimental communication primitives for ParaScale v1.

These primitives are adapted from the remote prototype, but kept independent
from torch.distributed so they can be tested locally and later plugged into
`CollectiveBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class CompressionStats:
    algorithm: str
    compression_ratio: float
    error_feedback: bool = False
    tracked_tensors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class GradientCompressor:
    algorithm = "base"

    def compress(self, tensor: Any, tensor_id: int = 0) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError

    def decompress(self, compressed: Any, metadata: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def stats(self) -> CompressionStats:
        return CompressionStats(self.algorithm, compression_ratio=1.0)


@dataclass
class TopKCompressor(GradientCompressor):
    compression_ratio: float = 0.01
    error_feedback: bool = True
    residual_errors: Dict[int, Any] = field(default_factory=dict)

    algorithm = "topk"

    def __post_init__(self) -> None:
        if not 0 < self.compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")

    def compress(self, tensor: Any, tensor_id: int = 0) -> Tuple[Any, Dict[str, Any]]:
        torch = _require_torch()
        working = tensor
        if self.error_feedback:
            residual = self.residual_errors.get(tensor_id)
            if residual is None:
                residual = torch.zeros_like(tensor)
            working = working + residual

        flat = working.flatten()
        k = max(1, int(flat.numel() * self.compression_ratio))
        _, indices = torch.topk(torch.abs(flat), k)
        values = flat[indices]

        if self.error_feedback:
            reconstructed = torch.zeros_like(flat)
            reconstructed[indices] = values
            self.residual_errors[tensor_id] = (flat - reconstructed).view_as(tensor)

        return values, {
            "algorithm": self.algorithm,
            "indices": indices,
            "shape": tuple(tensor.shape),
            "numel": int(flat.numel()),
            "dtype": str(tensor.dtype),
        }

    def decompress(self, compressed: Any, metadata: Dict[str, Any]) -> Any:
        torch = _require_torch()
        tensor = torch.zeros(
            int(metadata["numel"]), device=compressed.device, dtype=compressed.dtype
        )
        tensor[metadata["indices"]] = compressed
        return tensor.view(metadata["shape"])

    def stats(self) -> CompressionStats:
        return CompressionStats(
            self.algorithm,
            compression_ratio=self.compression_ratio,
            error_feedback=self.error_feedback,
            tracked_tensors=len(self.residual_errors),
        )


@dataclass
class OneBitCompressor(GradientCompressor):
    error_feedback: bool = True
    residual_errors: Dict[int, Any] = field(default_factory=dict)

    algorithm = "one_bit"

    def compress(self, tensor: Any, tensor_id: int = 0) -> Tuple[Any, Dict[str, Any]]:
        torch = _require_torch()
        working = tensor
        if self.error_feedback:
            residual = self.residual_errors.get(tensor_id)
            if residual is None:
                residual = torch.zeros_like(tensor)
            working = working + residual

        scale = torch.abs(working).mean()
        signs = (working >= 0).to(torch.uint8)
        if self.error_feedback:
            reconstructed = (signs.float() * 2 - 1).to(working.device) * scale
            self.residual_errors[tensor_id] = working - reconstructed
        return signs, {
            "algorithm": self.algorithm,
            "scale": scale,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
        }

    def decompress(self, compressed: Any, metadata: Dict[str, Any]) -> Any:
        return (compressed.float() * 2 - 1) * metadata["scale"]

    def stats(self) -> CompressionStats:
        return CompressionStats(
            self.algorithm,
            compression_ratio=1.0 / 32.0,
            error_feedback=self.error_feedback,
            tracked_tensors=len(self.residual_errors),
        )


def build_gradient_compressor(
    config: Dict[str, Any] | None = None,
) -> GradientCompressor:
    config = config or {}
    algorithm = str(config.get("algorithm", "none")).lower()
    if algorithm in {"none", "identity"}:
        return IdentityCompressor()
    if algorithm in {"topk", "top_k"}:
        return TopKCompressor(
            compression_ratio=float(config.get("compression_ratio", 0.01)),
            error_feedback=bool(config.get("error_feedback", True)),
        )
    if algorithm in {"one_bit", "1bit", "1-bit"}:
        return OneBitCompressor(error_feedback=bool(config.get("error_feedback", True)))
    raise ValueError(f"unknown gradient compression algorithm: {algorithm}")


class IdentityCompressor(GradientCompressor):
    algorithm = "identity"

    def compress(self, tensor: Any, tensor_id: int = 0) -> Tuple[Any, Dict[str, Any]]:
        return tensor, {"algorithm": self.algorithm, "shape": tuple(tensor.shape)}

    def decompress(self, compressed: Any, metadata: Dict[str, Any]) -> Any:
        return compressed


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise ImportError("communication compression requires PyTorch") from exc
    return torch
