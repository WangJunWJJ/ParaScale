# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午5:49
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Checkpoint and resume controller for runtime engines."""

from __future__ import annotations

from typing import Any, Dict, Optional

from parascale.checkpoint.io import atomic_binary_writer

from .coordination import coordinated_call
from .random_state import capture_rng_state, restore_rng_state


class CheckpointController:
    """Persist and restore runtime state through a checkpoint manager."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def save(
        self,
        checkpoint_manager: Any = None,
        step: Optional[int] = None,
        *,
        scheduler: Any = None,
    ) -> Any:
        if checkpoint_manager is None:
            return {"step": step if step is not None else self.engine.state.global_step}

        policy = self._collective_call(
            self._native_training_policy, identity=lambda value: value
        )
        from parascale.checkpoint import CheckpointManifest

        if scheduler is None:
            scheduler = getattr(self.engine, "scheduler", None)
        prepare_scheduler = getattr(self.engine, "prepare_scheduler", None)
        if callable(prepare_scheduler):
            scheduler = prepare_scheduler(
                getattr(self.engine.training_backend, "optimizer", None), scheduler
            )
        checkpoint_step = step if step is not None else self.engine.state.global_step
        rank = self._rank()
        world_size = self._world_size()
        runtime_files = self._collective_call(
            lambda: self._save_runtime_state(checkpoint_manager, checkpoint_step)
        )
        files = []
        backend_metadata = {}
        backend_state_written = False
        backend_name = getattr(
            self.engine.training_backend,
            "name",
            getattr(self.engine.config, "training_backend", "unknown"),
        )
        state_dict_type = self._state_dict_type(backend_name)
        shard_mode = bool(
            backend_name == "fsdp" and state_dict_type in {"sharded", "local"}
        )
        all_rank_backend_save = backend_name == "deepspeed" or (
            backend_name == "fsdp" and state_dict_type == "full"
        )

        def prepare_backend():
            prepare = getattr(self.engine.training_backend, "prepare_checkpoint", None)
            if callable(prepare) and hasattr(checkpoint_manager, "payload_path"):
                prepare()

        self._collective_call(prepare_backend)

        def save_backend():
            if rank != 0 and not shard_mode and not all_rank_backend_save:
                return [], {}, False
            if self.engine.training_backend is None or not hasattr(
                checkpoint_manager, "payload_path"
            ):
                return [], {}, False
            if backend_name in {"fsdp", "deepspeed"}:
                result = self._save_backend_specific_checkpoint(
                    checkpoint_manager, checkpoint_step, scheduler, backend_name
                )
            else:
                result = self._save_portable_backend_state(
                    checkpoint_manager, checkpoint_step, scheduler
                )
            if result[1].get("backend_checkpoint_error") and not self._allow_errors():
                raise RuntimeError(
                    f"backend checkpoint save failed: {result[1]['backend_checkpoint_error']}"
                )
            return result

        files, backend_metadata, backend_state_written = self._collective_call(
            save_backend
        )
        if rank != 0:
            self._collective_call(lambda: None)
            return {
                "step": checkpoint_step,
                "rank": rank,
                "world_size": world_size,
                "skipped": True,
                "reason": (
                    "backend checkpoint written; manifest is written by rank 0"
                    if all_rank_backend_save
                    else (
                        "checkpoint shard written; manifest is written by rank 0"
                        if shard_mode
                        else "checkpoint manifest is written by rank 0 only"
                    )
                ),
                "files": files,
            }

        def publish_manifest():
            # Every rank has finished writing before rank 0 hashes/publishes files.
            if shard_mode and rank == 0:
                self._add_expected_fsdp_shards(files, world_size, state_dict_type)
            if runtime_files:
                files.extend(
                    self._runtime_entry(runtime_rank)
                    for runtime_rank in range(world_size)
                )
            data_resume = getattr(self.engine, "data_resume", None)
            data_state = data_resume.capture() if data_resume is not None else {}
            optimizer = getattr(self.engine.training_backend, "optimizer", None)
            optimizer_metadata = dict(
                getattr(optimizer, "_parascale_optimizer_metadata", {}) or {}
            )
            manifest = CheckpointManifest(
                step=checkpoint_step,
                global_step=self.engine.state.global_step,
                consumed_samples=int(
                    getattr(self.engine.state, "consumed_samples", 0) or 0
                ),
                backend=backend_name,
                parallel_plan=self.engine.plan().to_dict(),
                data_state=data_state,
                files=files,
                metadata={
                    "last_metrics": dict(self.engine.state.last_metrics),
                    "retention_managed": backend_name in {"native", "native_ddp"},
                    "backend_state_written": backend_state_written,
                    "backend_specific_checkpoint": backend_name in {"fsdp", "deepspeed"}
                    and backend_state_written,
                    "rank": rank,
                    "world_size": world_size,
                    "rank_count": world_size,
                    "runtime_state_version": 1 if runtime_files else None,
                    "optimizer": optimizer_metadata,
                    "scheduler": dict(
                        getattr(self.engine, "scheduler_metadata", {}) or {}
                    ),
                    "distributed_backend": self._distributed_backend(),
                    "shard_count": (
                        world_size
                        if shard_mode
                        or (all_rank_backend_save and backend_state_written)
                        else (1 if backend_state_written else 0)
                    ),
                    "state_dict_type": state_dict_type,
                    "checkpoint_write_policy": (
                        "rank_sharded" if shard_mode else "rank0_manifest"
                    ),
                    "scheduler_state_written": bool(
                        scheduler is not None and hasattr(scheduler, "state_dict")
                    ),
                    **backend_metadata,
                },
            )
            objective = dict(getattr(self.engine, "objective_metadata", {}) or {})
            if objective:
                manifest.metadata["objective"] = objective
            if policy is not None:
                manifest.metadata["native_training_policy"] = policy
            path = checkpoint_manager.write_manifest(manifest)
            if backend_name in {"native", "native_ddp"} and callable(
                getattr(checkpoint_manager, "prune", None)
            ):
                checkpoint_manager.prune(
                    keep_last=int(
                        getattr(self.engine.config, "checkpoint_keep_last", 3)
                    )
                )
            return path

        return self._collective_call(publish_manifest)

    def load(
        self,
        checkpoint_manager: Any,
        step: int,
        *,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        dataloader: Any = None,
        optimizer_builder: Any = None,
        scheduler_builder: Any = None,
    ) -> Any:
        from parascale.runtime.backends import create_runtime_training_backend

        def validate_runtime():
            manifest = checkpoint_manager.read_manifest(step)
            self._validate_before_resume(checkpoint_manager, manifest)
            objective = dict(getattr(self.engine, "objective_metadata", {}) or {})
            if manifest.metadata.get("objective", {}) != objective:
                raise ValueError(
                    "checkpoint training objective does not match the current teacher/configuration"
                )
            self._validate_optimizer_metadata(manifest, optimizer)
            schedule_identity = dict(getattr(self.engine, "scheduler_metadata", {}) or {})
            if manifest.metadata.get("scheduler", {}) != schedule_identity:
                raise ValueError("checkpoint scheduler configuration mismatch")
            requested = (
                scheduler is not None
                or scheduler_builder is not None
                or getattr(self.engine, "scheduler", None) is not None
            )
            if bool(manifest.metadata.get("scheduler_state_written", False)) != requested:
                raise ValueError("checkpoint scheduler presence mismatch")
            policy = self._native_training_policy()
            saved_policy = manifest.metadata.get("native_training_policy")
            if policy is not None and saved_policy is None:
                raise ValueError(
                    "checkpoint is missing native training policy; strict resume "
                    "requires a checkpoint written by this runtime. Load model "
                    "weights explicitly to start a new training run instead."
                )
            if saved_policy != policy:
                raise ValueError(
                    "native training policy mismatch: "
                    f"checkpoint={saved_policy!r}, requested={policy!r}"
                )
            runtime_state = self._read_runtime_state(checkpoint_manager, manifest)
            if dataloader is not None:
                data_state = (runtime_state or {}).get("data_state", manifest.data_state)
                self.engine.data_resume.validate_resume(dataloader, data_state)
            return manifest, runtime_state

        manifest, runtime_state = self._collective_call(validate_runtime)
        backend_state_loaded = False
        if model is not None or optimizer is not None:
            backend_name = getattr(self.engine.config, "training_backend", "native")
            if optimizer_builder is not None and backend_name == "deepspeed":
                optimizer = optimizer_builder(model)
            self.engine.training_backend = create_runtime_training_backend(
                model=model,
                optimizer=optimizer,
                config=self.engine.config,
                local_rank=self.engine._local_rank(),
            )
            model, optimizer = self.engine.training_backend.setup()
            if optimizer_builder is not None and backend_name != "deepspeed":
                optimizer = optimizer_builder(model)
                self.engine.training_backend.optimizer = (
                    self.engine.training_backend.setup_optimizer(optimizer)
                )

        self._collective_call(
            lambda: self._validate_optimizer_metadata(
                manifest, getattr(self.engine.training_backend, "optimizer", None)
            )
        )

        prepare_scheduler = getattr(self.engine, "prepare_scheduler", None)
        if callable(prepare_scheduler):
            scheduler = prepare_scheduler(
                getattr(self.engine.training_backend, "optimizer", None),
                scheduler,
                scheduler_builder,
            )

        backend_name = getattr(self.engine.training_backend, "name", manifest.backend)
        backend_specific_entry = self._find_backend_specific_entry(
            manifest,
            backend_name,
        )
        backend_entry = self._find_file_entry(manifest, "backend_state")
        if (
            backend_specific_entry is not None
            and self.engine.training_backend is not None
        ):
            backend_state_loaded = self._load_backend_specific_checkpoint(
                checkpoint_manager,
                manifest,
                backend_specific_entry,
                scheduler,
            )
        elif backend_entry is not None and self.engine.training_backend is not None:

            def load_portable():
                return self._load_portable_backend_state(
                    checkpoint_manager,
                    manifest,
                    backend_entry,
                    scheduler,
                )

            backend_state_loaded = (
                self._collective_call(load_portable)
                if backend_name in {"native", "native_ddp"}
                else load_portable()
            )

        self.engine.state.global_step = int(manifest.global_step or manifest.step)
        self.engine.state.last_metrics = dict(manifest.metadata.get("last_metrics", {}))
        data_resume = getattr(self.engine, "data_resume", None)
        if data_resume is not None:
            data_resume.restore_manifest(manifest)
            if runtime_state is not None and "data_state" in runtime_state:
                self.engine.state.data_state = dict(runtime_state["data_state"])
                self.engine.state.consumed_micro_batches = int(
                    runtime_state["data_state"].get("consumed_micro_batches", 0)
                )
        if runtime_state is not None:

            def restore_runtime():
                precision = getattr(self.engine, "precision", None)
                load_precision = getattr(precision, "load_state_dict", None)
                if (
                    callable(load_precision)
                    and runtime_state.get("precision_state_dict") is not None
                ):
                    load_precision(runtime_state["precision_state_dict"])
                import torch

                self._restore_rng_state(torch, runtime_state["rng_state"])

            self._collective_call(restore_runtime)
            manifest.metadata["runtime_state_loaded"] = True
        manifest.metadata["backend_state_loaded"] = backend_state_loaded
        if self.engine.training_backend is not None:
            if hasattr(self.engine.training_backend, "optimizer_state_loaded"):
                manifest.metadata["optimizer_state_loaded"] = bool(
                    getattr(self.engine.training_backend, "optimizer_state_loaded")
                )
            if hasattr(self.engine.training_backend, "optimizer_state_error"):
                manifest.metadata["optimizer_state_error"] = str(
                    getattr(self.engine.training_backend, "optimizer_state_error")
                )
        return manifest

    def _native_training_policy(self) -> dict[str, Any] | None:
        from parascale.runtime.backends.native import (
            NativeDdpTrainingBackend,
            NativeTrainingBackend,
        )

        active = self.engine.training_backend
        active_name = getattr(active, "name", None)
        backend_name = getattr(self.engine.config, "training_backend", active_name)
        native_classes = {
            "native": NativeTrainingBackend,
            "native_ddp": NativeDdpTrainingBackend,
        }
        if (
            active_name is not None
            and active_name != backend_name
            and (active_name in native_classes or backend_name in native_classes)
        ):
            raise ValueError("native training policy mismatch: configured backend changed")
        backend_cls = native_classes.get(backend_name)
        requested = (
            backend_cls(config=self.engine.config).checkpoint_policy()
            if backend_cls is not None
            else None
        )
        describe = getattr(active, "checkpoint_policy", None)
        if backend_cls is not None and callable(describe):
            if describe() != requested:
                raise ValueError("native training policy mismatch: engine/backend configuration differs")
        return requested

    @staticmethod
    def _validate_optimizer_metadata(manifest: Any, optimizer: Any) -> None:
        saved = dict(manifest.metadata.get("optimizer", {}) or {})
        current = dict(getattr(optimizer, "_parascale_optimizer_metadata", {}) or {})
        if current.get("type") == "torchao_adamw4bit" and not saved:
            raise ValueError("checkpoint optimizer metadata is required for torchao_adamw4bit")
        if not saved or not current:
            return
        for key in (
            "type",
            "state_schema_version",
            "group_size",
            "compensate_quant_error",
            "error_compensation_dtype",
            "error_compensation_mode",
            "implementation",
            "torch_version",
            "torchao_version",
            "amsgrad",
            "bf16_stochastic_round",
        ):
            if key == "error_compensation_mode" and key not in saved:
                if current.get(key, "absolute") == "absolute":
                    continue
                raise ValueError(
                    "checkpoint optimizer metadata mismatch for "
                    "error_compensation_mode: saved='absolute', "
                    f"current={current.get(key)!r}"
                )
            if saved.get(key) != current.get(key):
                raise ValueError(
                    "checkpoint optimizer metadata mismatch for "
                    f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
                )

    @staticmethod
    def _distributed_backend() -> str | None:
        try:
            import torch.distributed as dist
        except Exception:
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        return str(dist.get_backend())

    def _validate_before_resume(
        self,
        checkpoint_manager: Any,
        manifest: Any,
    ) -> None:
        validate = getattr(checkpoint_manager, "validate_manifest", None)
        if callable(validate):
            report = validate(manifest)
            if not bool(getattr(report, "ok", False)):
                details = (
                    report.to_dict() if hasattr(report, "to_dict") else {"ok": False}
                )
                raise RuntimeError(
                    f"Checkpoint validation failed before resume: {details}"
                )

        checkpoint_world_size = int(
            manifest.metadata.get(
                "rank_count",
                manifest.metadata.get("world_size", 1),
            )
            or 1
        )
        runtime_world_size = self._world_size()
        allow_change = bool(
            getattr(
                self.engine.config,
                "allow_world_size_change_on_resume",
                False,
            )
        )
        if checkpoint_world_size != runtime_world_size and not allow_change:
            raise ValueError(
                "Checkpoint world_size mismatch: "
                f"checkpoint={checkpoint_world_size}, runtime={runtime_world_size}. "
                "Set allow_world_size_change_on_resume=true only when the backend "
                "and checkpoint format explicitly support resharding."
            )

    def _save_backend_specific_checkpoint(
        self,
        checkpoint_manager: Any,
        checkpoint_step: int,
        scheduler: Any,
        backend_name: str,
    ) -> tuple[list[Dict[str, Any]], Dict[str, Any], bool]:
        try:
            client_state = self._build_client_state(scheduler)
            result = self.engine.training_backend.save_checkpoint(
                checkpoint_manager,
                step=checkpoint_step,
                client_state=client_state,
            )
            files = []
            metadata = {}
            if isinstance(result, dict):
                files.extend(result.get("files", []))
                metadata = dict(result.get("metadata", {}))
            return files, metadata, True
        except Exception as exc:
            if not self._allow_errors():
                raise
            return (
                [
                    {
                        "path": f"{backend_name}_checkpoint",
                        "role": f"{backend_name}_checkpoint",
                        "error": str(exc),
                    }
                ],
                {"backend_checkpoint_error": str(exc)},
                False,
            )

    def _save_portable_backend_state(
        self,
        checkpoint_manager: Any,
        checkpoint_step: int,
        scheduler: Any,
    ) -> tuple[list[Dict[str, Any]], Dict[str, Any], bool]:
        try:
            import torch

            payload_path = checkpoint_manager.payload_path(
                checkpoint_step,
                "backend_state.pt",
            )
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "backend_state": self.engine.training_backend.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                    if scheduler is not None and hasattr(scheduler, "state_dict")
                    else None
                ),
                "rng_state": self._capture_rng_state(torch),
            }
            with atomic_binary_writer(payload_path) as stream:
                torch.save(payload, stream)
            return (
                [
                    {
                        "path": payload_path.name,
                        "role": "backend_state",
                        "format": "torch",
                    }
                ],
                {},
                True,
            )
        except Exception as exc:
            if not self._allow_errors():
                raise
            return (
                [
                    {
                        "path": "backend_state.pt",
                        "role": "backend_state",
                        "error": str(exc),
                    }
                ],
                {"backend_checkpoint_error": str(exc)},
                False,
            )

    def _load_backend_specific_checkpoint(
        self,
        checkpoint_manager: Any,
        manifest: Any,
        backend_entry: Dict[str, Any],
        scheduler: Any,
    ) -> bool:
        state_dict_type = backend_entry.get("state_dict_type")
        if state_dict_type is None:
            state_dict_type = manifest.metadata.get(
                "state_dict_type",
                manifest.metadata.get("fsdp_state_dict_type"),
            )
        load_kwargs = {}
        if state_dict_type is not None:
            load_kwargs["state_dict_type"] = str(state_dict_type)
        client_state = self.engine.training_backend.load_checkpoint(
            checkpoint_manager,
            step=manifest.step,
            **load_kwargs,
        )
        scheduler_state = client_state.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(client_state["scheduler_state_dict"])
        rng_state = client_state.get("rng_state")
        if rng_state is not None and not manifest.metadata.get("runtime_state_version"):
            try:
                import torch

                self._restore_rng_state(torch, rng_state)
            except Exception:
                pass
        return True

    def _load_portable_backend_state(
        self,
        checkpoint_manager: Any,
        manifest: Any,
        backend_entry: Dict[str, Any],
        scheduler: Any,
    ) -> bool:
        if not hasattr(checkpoint_manager, "resolve_payload_path"):
            raise RuntimeError(
                "checkpoint manager cannot resolve backend payload paths."
            )

        import torch

        payload_path = checkpoint_manager.resolve_payload_path(manifest, backend_entry)
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        backend_state = payload.get("backend_state", payload)
        self.engine.training_backend.load_state_dict(backend_state)
        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        if payload.get("rng_state") is not None and not manifest.metadata.get(
            "runtime_state_version"
        ):
            self._restore_rng_state(torch, payload["rng_state"])
        return True

    def _build_client_state(self, scheduler: Any = None) -> Dict[str, Any]:
        client_state: Dict[str, Any] = {
            "global_step": self.engine.state.global_step,
            "last_metrics": dict(self.engine.state.last_metrics),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None and hasattr(scheduler, "state_dict")
                else None
            ),
        }
        try:
            import torch

            client_state["rng_state"] = self._capture_rng_state(torch)
        except Exception as exc:
            client_state["rng_state_error"] = str(exc)
        return client_state

    @staticmethod
    def _find_backend_specific_entry(manifest: Any, backend_name: str) -> Any:
        backend_specific_roles = {
            "fsdp": "fsdp_state",
            "deepspeed": "deepspeed_checkpoint",
        }
        return CheckpointController._find_file_entry(
            manifest,
            backend_specific_roles.get(backend_name),
        )

    @staticmethod
    def _find_file_entry(manifest: Any, role: Optional[str]) -> Any:
        if role is None:
            return None
        return next(
            (
                file_entry
                for file_entry in manifest.files
                if file_entry.get("role") == role and not file_entry.get("error")
            ),
            None,
        )

    @staticmethod
    def _capture_rng_state(torch: Any) -> Dict[str, Any]:
        return capture_rng_state(torch)

    @staticmethod
    def _restore_rng_state(torch: Any, state: Dict[str, Any]) -> None:
        restore_rng_state(state, torch)

    @staticmethod
    def _runtime_entry(rank: int) -> Dict[str, Any]:
        return {
            "path": f"rank-{rank:05d}/runtime_state.pt",
            "role": "runtime_state",
            "format": "torch",
            "rank": rank,
        }

    def _save_runtime_state(self, manager: Any, step: int) -> bool:
        if not callable(getattr(manager, "payload_path", None)):
            return False
        try:
            import torch
        except ImportError:
            return False  # Metadata-only use remains supported without Torch.
        precision = getattr(self.engine, "precision", None)
        capture_precision = getattr(precision, "state_dict", None)
        data_resume = getattr(self.engine, "data_resume", None)
        path = manager.payload_path(step, self._runtime_entry(self._rank())["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_binary_writer(path) as stream:
            torch.save(
                {
                    "version": 1,
                    "rank": self._rank(),
                    "world_size": self._world_size(),
                    "rng_state": self._capture_rng_state(torch),
                    "data_state": data_resume.capture()
                    if data_resume is not None
                    else {},
                    "precision_state_dict": capture_precision()
                    if callable(capture_precision)
                    else None,
                },
                stream,
            )
        return True

    def _read_runtime_state(self, manager: Any, manifest: Any) -> Any:
        entries = [
            entry for entry in manifest.files if entry.get("role") == "runtime_state"
        ]
        if not entries and not manifest.metadata.get("runtime_state_version"):
            return None  # Legacy checkpoints retain their original loading protocol.
        ranks = [entry.get("rank") for entry in entries]
        expected = int(
            manifest.metadata.get("rank_count", manifest.metadata.get("world_size", 1))
        )
        if len(ranks) != expected or set(ranks) != set(range(expected)):
            raise ValueError(
                "Checkpoint runtime_state entries must cover each rank exactly once"
            )
        entry = next(
            (entry for entry in entries if entry["rank"] == self._rank()), None
        )
        if entry is None:
            raise ValueError(f"Checkpoint has no runtime_state for rank {self._rank()}")
        import torch

        state = torch.load(
            manager.resolve_payload_path(manifest, entry),
            map_location="cpu",
            weights_only=True,
        )
        if (
            state.get("version") != 1
            or state.get("rank") != self._rank()
            or state.get("world_size") != expected
        ):
            raise ValueError("Checkpoint runtime_state identity/version mismatch")
        if (
            not isinstance(state.get("rng_state"), dict)
            or "torch_cpu" not in state["rng_state"]
        ):
            raise ValueError("Checkpoint runtime_state is missing RNG state")
        return state

    def _rank(self) -> int:
        rank_fn = getattr(self.engine, "_distributed_rank", None)
        if callable(rank_fn):
            try:
                return int(rank_fn())
            except Exception:
                return 0
        return 0

    def _collective_call(self, operation: Any, *, identity: Any = None) -> Any:
        """Finish a local I/O phase on every rank before propagating errors."""
        return coordinated_call(
            operation,
            phase="Checkpoint",
            identity=identity,
            fallback_barrier=self._barrier,
        )

    def _world_size(self) -> int:
        world_size_fn = getattr(self.engine, "_distributed_world_size", None)
        if callable(world_size_fn):
            try:
                return max(1, int(world_size_fn()))
            except Exception:
                return 1
        return 1

    def _barrier(self) -> None:
        barrier = getattr(self.engine, "_distributed_barrier", None)
        if callable(barrier):
            barrier()

    def _state_dict_type(self, backend_name: str) -> str:
        if backend_name == "fsdp":
            return str(getattr(self.engine.config, "fsdp_state_dict_type", "full"))
        return "portable"

    def _allow_errors(self) -> bool:
        return bool(
            getattr(self.engine.config, "allow_checkpoint_error_for_benchmark", False)
            or getattr(self.engine.config, "skip_final_checkpoint", False)
        )

    @staticmethod
    def _add_expected_fsdp_shards(
        files: list[Dict[str, Any]],
        world_size: int,
        state_dict_type: str,
    ) -> None:
        existing = {str(file_entry.get("path")) for file_entry in files}
        for shard_rank in range(int(world_size)):
            path = f"rank-{shard_rank:05d}/fsdp_state.pt"
            if path in existing:
                continue
            files.append(
                {
                    "path": path,
                    "role": "fsdp_state",
                    "format": "torch",
                    "state_dict_type": state_dict_type,
                    "rank": shard_rank,
                    "expected_shard": True,
                }
            )
