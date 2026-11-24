# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Configuration owned by the local pretrained causal-LM workload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import _section


@dataclass(frozen=True)
class CausalLmSpec:
    model_path: str
    data_path: str
    batch_size: int = 1
    sequence_length: int = 512
    seed: int = 42
    data_seed: int = 42
    shuffle: bool = True
    repeat: bool = True
    activation_checkpointing: bool = False
    attention_backend: str = "sdpa"
    shuffle_window_blocks: int = 65536

    def __post_init__(self) -> None:
        if not self.model_path or not self.data_path:
            raise ValueError("causal_lm requires local model and token corpus paths")
        for name, minimum in (
            ("batch_size", 1),
            ("sequence_length", 2),
            ("seed", 0),
            ("data_seed", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid causal_lm {name}")
        for name in ("shuffle", "repeat", "activation_checkpointing"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"causal_lm {name} must be boolean")
        if self.attention_backend not in {"eager", "sdpa"}:
            raise ValueError("causal_lm attention_backend must be eager or sdpa")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CausalLmSpec":
        model, data = _section(config, "model"), _section(config, "data")
        training, runtime = _section(config, "training"), _section(config, "parascale")
        if data.get("type") != "token_corpus":
            raise ValueError(
                "causal_lm requires data.type=token_corpus; run prepare-text first"
            )
        if data.get("num_workers", 0) != 0:
            raise ValueError(
                "token_corpus reads mapped blocks directly; num_workers must be zero"
            )
        return cls(
            model_path=str(model.get("pretrained_model_name_or_path") or ""),
            data_path=str(data.get("path") or ""),
            batch_size=data.get("batch_size", training.get("batch_size", 1)),
            sequence_length=data.get("sequence_length", 512),
            seed=training.get("seed", 42),
            data_seed=data.get("seed", 42),
            shuffle=data.get("shuffle", True),
            repeat=data.get("repeat", True),
            activation_checkpointing=model.get(
                "activation_checkpointing",
                runtime.get("enable_activation_checkpointing", False),
            ),
            attention_backend=model.get("attention_backend", "sdpa"),
            shuffle_window_blocks=data.get("shuffle_window_blocks", 65536),
        )
