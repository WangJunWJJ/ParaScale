# -*- coding: utf-8 -*-
# @Time : 2026/5/3 下午10:01
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Collective communication interfaces reused by training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class CollectiveBackend:
    name = "abstract"

    def init_process_group(self, **kwargs: Any) -> None:
        raise NotImplementedError

    def all_reduce(self, value: Any, op: str = "sum", group: Any = None) -> Any:
        raise NotImplementedError

    def reduce_scatter(
        self, output: Any, input: Any, op: str = "sum", group: Any = None
    ) -> Any:
        raise NotImplementedError

    def all_gather(self, output: Any, input: Any, group: Any = None) -> Any:
        raise NotImplementedError

    def all_to_all(self, output: Any, input: Any, group: Any = None) -> Any:
        raise NotImplementedError

    def broadcast(self, value: Any, src: int = 0, group: Any = None) -> Any:
        raise NotImplementedError

    def barrier(self, group: Any = None) -> None:
        raise NotImplementedError

    def new_group(self, ranks: List[int], name: Optional[str] = None) -> Any:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


@dataclass
class MockCollectiveBackend(CollectiveBackend):
    name: str = "mock"
    initialized: bool = False
    world_size: int = 1
    rank: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    groups: Dict[str, List[int]] = field(default_factory=dict)

    def init_process_group(self, **kwargs: Any) -> None:
        self.initialized = True
        self.world_size = int(kwargs.get("world_size", self.world_size) or 1)
        self.rank = int(kwargs.get("rank", self.rank) or 0)
        self.history.append({"op": "init_process_group", "kwargs": dict(kwargs)})

    def all_reduce(self, value: Any, op: str = "sum", group: Any = None) -> Any:
        self.history.append({"op": "all_reduce", "reduce_op": op, "group": group})
        return value

    def reduce_scatter(
        self, output: Any, input: Any, op: str = "sum", group: Any = None
    ) -> Any:
        self.history.append({"op": "reduce_scatter", "reduce_op": op, "group": group})
        return output if output is not None else input

    def all_gather(self, output: Any, input: Any, group: Any = None) -> Any:
        self.history.append({"op": "all_gather", "group": group})
        return output if output is not None else [input]

    def all_to_all(self, output: Any, input: Any, group: Any = None) -> Any:
        self.history.append({"op": "all_to_all", "group": group})
        return output if output is not None else input

    def broadcast(self, value: Any, src: int = 0, group: Any = None) -> Any:
        self.history.append({"op": "broadcast", "src": src, "group": group})
        return value

    def barrier(self, group: Any = None) -> None:
        self.history.append({"op": "barrier", "group": group})

    def new_group(self, ranks: List[int], name: Optional[str] = None) -> str:
        group_name = name or f"group-{len(self.groups)}"
        self.groups[group_name] = list(ranks)
        self.history.append(
            {"op": "new_group", "name": group_name, "ranks": list(ranks)}
        )
        return group_name

    def shutdown(self) -> None:
        self.history.append({"op": "shutdown"})
        self.initialized = False


@dataclass
class TorchDistributedCollectiveBackend(CollectiveBackend):
    name: str = "torch_distributed"
    backend: str = "auto"
    initialized: bool = False
    world_size: int = 1
    rank: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    groups: Dict[str, Any] = field(default_factory=dict)

    def init_process_group(self, **kwargs: Any) -> None:
        dist = self._require_dist()
        self.backend = str(kwargs.get("backend", self.backend))
        if self.backend == "auto":
            self.backend = self._select_backend()
        self.world_size = int(kwargs.get("world_size", self.world_size) or 1)
        self.rank = int(kwargs.get("rank", self.rank) or 0)
        if not dist.is_initialized():
            init_kwargs = dict(kwargs)
            init_kwargs["backend"] = self.backend
            init_kwargs.setdefault("world_size", self.world_size)
            init_kwargs.setdefault("rank", self.rank)
            dist.init_process_group(**init_kwargs)
        self.initialized = True
        self.history.append(
            {
                "op": "init_process_group",
                "backend": self.backend,
                "kwargs": dict(kwargs),
            }
        )

    def all_reduce(self, value: Any, op: str = "sum", group: Any = None) -> Any:
        dist = self._require_initialized()
        dist.all_reduce(value, op=self._reduce_op(op), group=group)
        self.history.append({"op": "all_reduce", "reduce_op": op, "group": group})
        return value

    def reduce_scatter(
        self, output: Any, input: Any, op: str = "sum", group: Any = None
    ) -> Any:
        dist = self._require_initialized()
        dist.reduce_scatter(output, input, op=self._reduce_op(op), group=group)
        self.history.append({"op": "reduce_scatter", "reduce_op": op, "group": group})
        return output

    def all_gather(self, output: Any, input: Any, group: Any = None) -> Any:
        dist = self._require_initialized()
        dist.all_gather(output, input, group=group)
        self.history.append({"op": "all_gather", "group": group})
        return output

    def all_to_all(self, output: Any, input: Any, group: Any = None) -> Any:
        dist = self._require_initialized()
        dist.all_to_all(output, input, group=group)
        self.history.append({"op": "all_to_all", "group": group})
        return output

    def broadcast(self, value: Any, src: int = 0, group: Any = None) -> Any:
        dist = self._require_initialized()
        dist.broadcast(value, src=src, group=group)
        self.history.append({"op": "broadcast", "src": src, "group": group})
        return value

    def barrier(self, group: Any = None) -> None:
        dist = self._require_initialized()
        dist.barrier(group=group)
        self.history.append({"op": "barrier", "group": group})

    def new_group(self, ranks: List[int], name: Optional[str] = None) -> Any:
        dist = self._require_initialized()
        group = dist.new_group(ranks=list(ranks))
        group_name = name or f"group-{len(self.groups)}"
        self.groups[group_name] = group
        self.history.append(
            {"op": "new_group", "name": group_name, "ranks": list(ranks)}
        )
        return group

    def shutdown(self) -> None:
        dist = self._require_dist()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        self.initialized = False
        self.history.append({"op": "shutdown"})

    @staticmethod
    def _require_dist() -> Any:
        try:
            import torch.distributed as dist
        except Exception as exc:
            raise ImportError(
                "TorchDistributedCollectiveBackend requires torch.distributed."
            ) from exc
        if not dist.is_available():
            raise RuntimeError(
                "torch.distributed is not available in this PyTorch build."
            )
        return dist

    def _require_initialized(self) -> Any:
        dist = self._require_dist()
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed process group is not initialized.")
        return dist

    @staticmethod
    def _select_backend() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "nccl"
        except Exception:
            pass
        return "gloo"

    @staticmethod
    def _reduce_op(name: str) -> Any:
        import torch.distributed as dist

        mapping = {
            "sum": dist.ReduceOp.SUM,
            "avg": getattr(dist.ReduceOp, "AVG", dist.ReduceOp.SUM),
            "mean": getattr(dist.ReduceOp, "AVG", dist.ReduceOp.SUM),
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
            "prod": dist.ReduceOp.PRODUCT,
        }
        if name not in mapping:
            raise ValueError(f"unsupported reduce op: {name}")
        return mapping[name]
