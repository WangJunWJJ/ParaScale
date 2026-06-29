# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Base device backend contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DeviceBackend:
    name: str
    accelerator: str
    communication: str
    available: bool = False

    def is_available(self) -> bool:
        return self.available

    def device(self, local_rank: int = 0) -> str:
        return self.device_name(local_rank)

    def set_device(self, local_rank: int = 0) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def memory_allocated(self) -> int:
        return 0

    def max_memory_allocated(self) -> int:
        return 0

    def reset_peak_memory_stats(self) -> None:
        return None

    def supports_bf16(self) -> bool:
        return False

    def supports_flash_attention(self) -> bool:
        return False

    def device_name(self, index: int = 0) -> str:
        if self.accelerator == "cpu":
            return "cpu"
        return f"{self.accelerator}:{index}"

    def capability(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "accelerator": self.accelerator,
            "communication": self.communication,
            "available": self.available,
            "supports_bf16": self.supports_bf16(),
            "supports_flash_attention": self.supports_flash_attention(),
        }


__all__ = ["DeviceBackend"]
