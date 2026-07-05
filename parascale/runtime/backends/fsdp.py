# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:58
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""FSDP training backend."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .base import TrainingBackend, _require_torch


class FSDPTrainingBackend(TrainingBackend):
    name = "fsdp"
    _DEFAULT_CHECKPOINT_MODULE_CLASSES = {
        "CLIPEncoderLayer",
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "Qwen2VLDecoderLayer",
        "Qwen2_5_VLDecoderLayer",
        "SiglipEncoderLayer",
        "ViTLayer",
    }

    def setup(self) -> Tuple[Any, Any]:
        if self.model is None:
            return self.model, self.optimizer
        torch = _require_torch()
        try:
            from torch.distributed.fsdp import (
                CPUOffload,
                FullyShardedDataParallel,
                MixedPrecision,
                ShardingStrategy,
            )
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        except Exception as exc:
            raise ImportError(
                "FSDP backend requires a PyTorch build with torch.distributed.fsdp."
            ) from exc

        config = self.config
        strategy_name = getattr(config, "fsdp_sharding_strategy", "full_shard")
        strategy_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        }
        precision = getattr(config, "precision", "fp32")
        dtype = (
            torch.float16
            if precision == "fp16"
            else torch.bfloat16 if precision == "bf16" else None
        )
        mixed_precision = (
            MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
            if dtype is not None
            else None
        )
        auto_wrap_policy = None
        if bool(getattr(config, "fsdp_auto_wrap", False)):
            min_params = int(getattr(config, "fsdp_min_num_params", 100_000_000))
            def auto_wrap_policy(module, recurse, nonwrapped_numel):
                return size_based_auto_wrap_policy(
                    module,
                    recurse,
                    nonwrapped_numel,
                    min_num_params=min_params,
                )

        self._apply_activation_checkpointing(self.model)
        device_id = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else None
        )
        self.model = FullyShardedDataParallel(
            self.model,
            sharding_strategy=strategy_map[strategy_name],
            cpu_offload=CPUOffload(
                offload_params=bool(getattr(config, "fsdp_cpu_offload", False))
            ),
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device_id,
            use_orig_params=bool(getattr(config, "fsdp_use_orig_params", True)),
        )
        return self.model, self.optimizer

    def _apply_activation_checkpointing(self, model: Any) -> None:
        if not bool(getattr(self.config, "enable_activation_checkpointing", False)):
            return
        policy = str(
            getattr(
                self.config,
                "fsdp_activation_checkpointing_policy",
                "transformer_auto",
            )
            or "transformer_auto"
        )
        if policy == "none":
            return
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
        except Exception as exc:
            raise ImportError(
                "FSDP activation checkpointing requires PyTorch checkpoint_wrapper."
            ) from exc

        import functools

        wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        min_params = int(getattr(self.config, "fsdp_min_num_params", 100_000_000))
        class_names = set(self._DEFAULT_CHECKPOINT_MODULE_CLASSES)
        class_names.update(
            str(name)
            for name in getattr(self.config, "fsdp_checkpoint_module_classes", []) or []
            if str(name)
        )

        def should_checkpoint(module: Any) -> bool:
            if getattr(module, "_parascale_activation_checkpointed", False):
                return False
            if policy == "size_based":
                return sum(param.numel() for param in module.parameters()) >= min_params
            name = module.__class__.__name__
            return name in class_names or name.endswith("DecoderLayer")

        matched = [
            module
            for module in model.modules()
            if module is not model and should_checkpoint(module)
        ]
        if not matched and policy == "transformer_auto":
            matched = [
                module
                for module in model.modules()
                if module is not model
                and sum(param.numel() for param in module.parameters()) >= min_params
            ]
            matched_ids = {id(module) for module in matched}
            def check_fn(module):
                return id(module) in matched_ids
        else:
            check_fn = should_checkpoint
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper,
            check_fn=check_fn,
        )
        for module in matched:
            setattr(module, "_parascale_activation_checkpointed", True)
        self.activation_checkpointed_modules = len(matched)

    def state_dict(self) -> Dict[str, Any]:
        if self.model is None:
            return super().state_dict()
        try:
            from torch.distributed.fsdp import (
                FullOptimStateDictConfig,
                FullStateDictConfig,
                LocalOptimStateDictConfig,
                LocalStateDictConfig,
                ShardedOptimStateDictConfig,
                ShardedStateDictConfig,
                StateDictType,
            )
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except Exception:
            return super().state_dict()

        state_type = getattr(self.config, "fsdp_state_dict_type", "full")
        if state_type == "sharded":
            state_dict_type = StateDictType.SHARDED_STATE_DICT
            state_config = ShardedStateDictConfig(offload_to_cpu=True)
            optim_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
        elif state_type == "local":
            state_dict_type = StateDictType.LOCAL_STATE_DICT
            state_config = LocalStateDictConfig()
            optim_config = LocalOptimStateDictConfig()
        else:
            state_dict_type = StateDictType.FULL_STATE_DICT
            state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            optim_config = FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )

        with FSDP.state_dict_type(
            self.model, state_dict_type, state_config, optim_config
        ):
            model_state = self.model.state_dict()
            optimizer_state = (
                FSDP.optim_state_dict(self.model, self.optimizer)
                if self.optimizer is not None
                else None
            )
        return {
            "backend": self.name,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "state_dict_type": state_type,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self.model is None:
            return super().load_state_dict(state)
        model_state = state.get("model_state_dict")
        if model_state is not None and hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(model_state)
        optimizer_state = state.get("optimizer_state_dict")
        if self.optimizer is None or optimizer_state is None:
            return None
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            optimizer_state = FSDP.optim_state_dict_to_load(
                self.model,
                self.optimizer,
                optimizer_state,
            )
        except Exception:
            pass
        try:
            self.optimizer.load_state_dict(optimizer_state)
            self.optimizer_state_loaded = True
        except ValueError as exc:
            self.optimizer_state_loaded = False
            self.optimizer_state_error = str(exc)
        return None

    def save_checkpoint(
        self,
        checkpoint_manager: Any,
        step: Any = None,
        client_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if not hasattr(checkpoint_manager, "payload_path"):
            return super().save_checkpoint(
                checkpoint_manager, step, client_state, **kwargs
            )

        torch = _require_torch()
        checkpoint_step = int(step or 0)
        state_type = getattr(self.config, "fsdp_state_dict_type", "full")
        rank = self._rank()
        filename = (
            f"rank-{rank:05d}/fsdp_state.pt"
            if state_type in {"sharded", "local"}
            else "fsdp_state.pt"
        )
        backend_state = self.state_dict()
        files = []
        if rank == 0 or state_type in {"sharded", "local"}:
            payload_path = checkpoint_manager.payload_path(checkpoint_step, filename)
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backend_state": backend_state,
                    "client_state": client_state or {},
                },
                payload_path,
            )
            files.append(
                {
                    "path": filename,
                    "role": "fsdp_state",
                    "format": "torch",
                    "state_dict_type": state_type,
                    "rank": rank,
                }
            )
        return {
            "files": files,
            "metadata": {
                "backend_checkpoint": self.name,
                "fsdp_state_dict_type": state_type,
                "rank": rank,
            },
        }

    def load_checkpoint(
        self, checkpoint_manager: Any, step: Any = None, **kwargs: Any
    ) -> Dict[str, Any]:
        if not hasattr(checkpoint_manager, "payload_path"):
            return super().load_checkpoint(checkpoint_manager, step, **kwargs)

        torch = _require_torch()
        checkpoint_step = int(step or 0)
        state_type = getattr(self.config, "fsdp_state_dict_type", "full")
        rank = self._rank()
        filename = (
            f"rank-{rank:05d}/fsdp_state.pt"
            if state_type in {"sharded", "local"}
            else "fsdp_state.pt"
        )
        payload_path = checkpoint_manager.payload_path(checkpoint_step, filename)
        if not payload_path.is_file():
            raise FileNotFoundError(
                f"FSDP checkpoint payload not found: {payload_path}"
            )
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        backend_state = payload.get("backend_state", payload)
        if isinstance(backend_state, dict):
            self.load_state_dict(backend_state)
        return payload.get("client_state", {})

    def _rank(self) -> int:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass
        return int(self.local_rank)


__all__ = ["FSDPTrainingBackend"]
