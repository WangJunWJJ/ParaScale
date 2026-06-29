# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:59
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""DeepSpeed training backend."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from parascale.configuration import build_deepspeed_final_config, resolve_config

from .base import TrainingBackend


class DeepSpeedTrainingBackend(TrainingBackend):
    name = "deepspeed"

    def __init__(
        self,
        model: Any = None,
        optimizer: Any = None,
        config: Any = None,
        local_rank: int = 0,
    ):
        super().__init__(model, optimizer, config, local_rank)
        self.engine = None

    def setup(self) -> Tuple[Any, Any]:
        if self.model is None:
            return self.model, self.optimizer
        try:
            import deepspeed
        except Exception as exc:
            raise ImportError(
                "DeepSpeed backend requires the optional 'deepspeed' package."
            ) from exc

        self._make_parameters_contiguous(self.model)
        engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            model_parameters=(
                None if self.optimizer is not None else self.model.parameters()
            ),
            config=self.build_deepspeed_config(),
        )
        self.engine = engine
        self.model = engine
        self.optimizer = optimizer
        return self.model, self.optimizer

    @staticmethod
    def _make_parameters_contiguous(model: Any) -> None:
        for parameter in getattr(model, "parameters", lambda: [])():
            data = getattr(parameter, "data", None)
            if data is not None and hasattr(data, "is_contiguous"):
                if not data.is_contiguous():
                    parameter.data = data.contiguous()

    def build_deepspeed_config(self) -> Dict[str, Any]:
        config = self.config
        config_data = {
            "parascale": config.to_dict() if hasattr(config, "to_dict") else {},
            "deepspeed_config": getattr(config, "deepspeed_config", None) or {},
        }
        resolved = resolve_config(config_data)
        ds_config = build_deepspeed_final_config(resolved)
        grad_clip = getattr(config, "grad_clip_norm", None)
        if grad_clip is not None:
            ds_config["gradient_clipping"] = grad_clip
        if getattr(config, "deepspeed_config", None):
            ds_config["_parascale"] = {
                **dict(ds_config.get("_parascale", {})),
                "merged_user_config": True,
                "enforced_keys": [
                    "train_micro_batch_size_per_gpu",
                    "gradient_accumulation_steps",
                    "precision",
                    "zero_optimization.stage",
                ],
            }
        return ds_config

    @classmethod
    def _merge_user_deepspeed_config(
        cls, user_config: Dict[str, Any], parascale_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = cls._deep_merge_dicts(user_config, parascale_config)
        merged["_parascale"] = {
            **dict(merged.get("_parascale", {})),
            "merged_user_config": True,
            "enforced_keys": [
                "train_micro_batch_size_per_gpu",
                "gradient_accumulation_steps",
                "precision",
                "zero_optimization.stage",
            ],
        }
        return merged

    @classmethod
    def _deep_merge_dicts(
        cls, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge_dicts(current, value)
            else:
                merged[key] = value
        return merged

    def backward(self, loss: Any) -> None:
        if self.engine is None:
            raise RuntimeError("DeepSpeed backend is not initialized")
        self.engine.backward(loss)

    def step(self, optimizer: Any = None) -> None:
        if self.engine is None:
            raise RuntimeError("DeepSpeed backend is not initialized")
        self.engine.step()

    def state_dict(self) -> Dict[str, Any]:
        return {"backend": self.name, "deepspeed_engine": self.engine is not None}

    def save_checkpoint(
        self,
        checkpoint_manager: Any,
        step: Any = None,
        client_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if self.engine is None:
            return super().save_checkpoint(
                checkpoint_manager, step, client_state, **kwargs
            )
        if hasattr(checkpoint_manager, "manifest_path"):
            checkpoint_step = int(step or 0)
            checkpoint_dir = checkpoint_manager.manifest_path(checkpoint_step).parent
            save_dir = checkpoint_dir / "deepspeed"
            tag = f"global_step{checkpoint_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            self.engine.save_checkpoint(
                str(save_dir), tag=tag, client_state=client_state or {}
            )
            return {
                "files": [
                    {
                        "path": "deepspeed",
                        "role": "deepspeed_checkpoint",
                        "format": "deepspeed",
                        "tag": tag,
                    }
                ],
                "metadata": {
                    "backend_checkpoint": self.name,
                    "deepspeed_tag": tag,
                },
            }
        self.engine.save_checkpoint(
            checkpoint_manager, tag=str(step), client_state=client_state or {}
        )
        return checkpoint_manager

    def load_checkpoint(
        self, checkpoint_manager: Any, step: Any = None, **kwargs: Any
    ) -> Dict[str, Any]:
        if self.engine is None:
            return super().load_checkpoint(checkpoint_manager, step, **kwargs)
        if hasattr(checkpoint_manager, "manifest_path"):
            checkpoint_step = int(step or 0)
            load_dir = str(
                checkpoint_manager.manifest_path(checkpoint_step).parent / "deepspeed"
            )
            load_tag = f"global_step{checkpoint_step}"
        else:
            load_dir, load_tag = self.parse_checkpoint_path(checkpoint_manager, step)
        load_path, client_state = self.engine.load_checkpoint(load_dir, tag=load_tag)
        if load_path is None:
            raise FileNotFoundError(
                f"DeepSpeed checkpoint not found: dir={load_dir}, tag={load_tag}"
            )
        return client_state or {}

    @staticmethod
    def parse_checkpoint_path(
        checkpoint_path: str, tag: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        if tag is not None:
            return checkpoint_path, tag
        base = os.path.basename(os.path.normpath(checkpoint_path))
        if base.startswith("checkpoint_"):
            return os.path.dirname(os.path.normpath(checkpoint_path)), base
        return checkpoint_path, None

    _parse_checkpoint_path = parse_checkpoint_path


__all__ = ["DeepSpeedTrainingBackend"]
