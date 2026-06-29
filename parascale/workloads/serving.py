# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:48
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Serving model loaders for built-in workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from parascale.runtime.specs import TinyTorchWorkloadSpec

from .common import _require_torch, _section
from .tiny import build_tiny_torch_components


def build_serving_model_from_checkpoint(
    config_data: Dict[str, Any], manifest: Any, checkpoint_manager: Any
):
    registry = default_serving_model_registry()
    serving = _section(config_data, "serving")
    training = _section(config_data, "training")
    workload = str(serving.get("workload", training.get("workload", "torch_tiny_mlp")))
    return registry.create(
        workload,
        config_data=config_data,
        manifest=manifest,
        checkpoint_manager=checkpoint_manager,
    )


@dataclass
class ServingModelRegistry:
    loaders: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, name: str, loader: Callable[..., Any]) -> None:
        self.loaders[name] = loader

    def create(self, name: str, **kwargs: Any) -> Any:
        if name not in self.loaders:
            raise ValueError(f"unsupported serving workload: {name}")
        return self.loaders[name](**kwargs)


def default_serving_model_registry() -> ServingModelRegistry:
    registry = ServingModelRegistry()
    for name in ["torch_tiny", "torch_tiny_mlp", "tiny_torch"]:
        registry.register(name, load_tiny_torch_serving_model)
    return registry


def load_tiny_torch_serving_model(
    config_data: Dict[str, Any], manifest: Any, checkpoint_manager: Any
):
    model, _optimizer, _dataloader, _loss_fn = build_tiny_torch_components(
        TinyTorchWorkloadSpec.from_config(config_data)
    )
    backend_entry = next(
        (
            file_entry
            for file_entry in getattr(manifest, "files", [])
            if file_entry.get("role") == "backend_state" and not file_entry.get("error")
        ),
        None,
    )
    if backend_entry is None:
        raise FileNotFoundError("checkpoint does not contain a backend_state payload.")
    if not hasattr(checkpoint_manager, "resolve_payload_path"):
        raise RuntimeError("checkpoint manager cannot resolve backend payload paths.")

    torch = _require_torch()
    payload_path = checkpoint_manager.resolve_payload_path(manifest, backend_entry)
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    backend_state = payload.get("backend_state", payload)
    model_state = backend_state.get("model_state_dict")
    if model_state is None:
        raise RuntimeError("backend_state payload does not contain model_state_dict.")
    model.load_state_dict(model_state)
    model.eval()
    return TinyTorchServingAdapter(model)


class TinyTorchServingAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model

    def generate(self, requests: Any):
        return self._run(requests)

    def embed(self, requests: Any):
        return self._run(requests)

    def _run(self, requests: Any):
        torch = _require_torch()
        tensor = self._to_tensor(requests, torch)
        try:
            device = next(self.model.parameters()).device
            tensor = tensor.to(device)
        except StopIteration:
            pass
        with torch.no_grad():
            output = self.model(input_ids=tensor)
        return output.detach().cpu().tolist()

    @staticmethod
    def _to_tensor(requests: Any, torch: Any):
        if isinstance(requests, dict):
            requests = requests.get("input_ids", requests.get("x"))
        if hasattr(requests, "detach"):
            return requests.float()
        if (
            isinstance(requests, (list, tuple))
            and requests
            and not isinstance(requests[0], (list, tuple))
        ):
            requests = [requests]
        return torch.tensor(requests, dtype=torch.float32)
