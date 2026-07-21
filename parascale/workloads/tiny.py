# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Tiny torch workload used by quickstart smoke tests."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from parascale.workloads.specs.tiny import TinyTorchWorkloadSpec

from .common import _require_torch


def build_tiny_torch_components(spec: TinyTorchWorkloadSpec):
    torch = _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(spec.seed)

    class TinyTorchMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(spec.input_dim, spec.hidden_dim),
                nn.ReLU(),
                nn.Linear(spec.hidden_dim, spec.output_dim),
            )

        def forward(self, input_ids=None, **kwargs):
            x = input_ids if input_ids is not None else kwargs["x"]
            return self.net(x.float())

    model = TinyTorchMLP()
    optimizer = optim.AdamW(model.parameters(), lr=spec.lr)

    def dataloader() -> Iterable[Dict[str, Any]]:
        generator = torch.Generator()
        generator.manual_seed(spec.seed)
        for _ in range(spec.num_batches):
            x = torch.randn(spec.batch_size, spec.input_dim, generator=generator)
            y = torch.randn(spec.batch_size, spec.output_dim, generator=generator)
            yield {"input_ids": x, "labels": y}

    def loss_fn(output, batch):
        return ((output - batch["labels"]) ** 2).mean()

    return model, optimizer, dataloader(), loss_fn
