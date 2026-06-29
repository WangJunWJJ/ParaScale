# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:26
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded KV cache manager for inference requests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class KVCacheBlock:
    request_id: str
    value: Any


@dataclass
class KVCacheManager:
    max_blocks: int = 1024
    blocks: Dict[str, KVCacheBlock] = field(default_factory=OrderedDict)

    def put(self, request_id: str, value: Any) -> None:
        if request_id in self.blocks:
            self.blocks.pop(request_id)
        self.blocks[request_id] = KVCacheBlock(request_id=request_id, value=value)
        while len(self.blocks) > self.max_blocks:
            self.blocks.pop(next(iter(self.blocks)))

    def get(self, request_id: str) -> Any:
        block = self.blocks.get(request_id)
        if block is not None:
            self.blocks.pop(request_id)
            self.blocks[request_id] = block
        return None if block is None else block.value

    def release(self, request_id: str) -> None:
        self.blocks.pop(request_id, None)

    def clear(self) -> None:
        self.blocks.clear()

    def stats(self) -> Dict[str, int]:
        return {"blocks": len(self.blocks), "max_blocks": self.max_blocks}
