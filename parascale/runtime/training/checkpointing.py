# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午5:49
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Checkpoint and resume controller for runtime engines."""

from __future__ import annotations

from typing import Any, Dict, Optional


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

        from parascale.checkpoint import CheckpointManifest

        checkpoint_step = step if step is not None else self.engine.state.global_step
        rank = self._rank()
        world_size = self._world_size()
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
        if rank != 0 and not shard_mode and not all_rank_backend_save:
            self._barrier()
            return {
                "step": checkpoint_step,
                "rank": rank,
                "world_size": world_size,
                "skipped": True,
                "reason": "checkpoint manifest is written by rank 0 only",
            }
        if (
            self.engine.training_backend is not None
            and backend_name in {"fsdp", "deepspeed"}
            and hasattr(checkpoint_manager, "payload_path")
        ):
            files, backend_metadata, backend_state_written = (
                self._save_backend_specific_checkpoint(
                    checkpoint_manager,
                    checkpoint_step,
                    scheduler,
                    backend_name,
                )
            )
        elif self.engine.training_backend is not None and hasattr(
            checkpoint_manager, "payload_path"
        ):
            files, backend_metadata, backend_state_written = (
                self._save_portable_backend_state(
                    checkpoint_manager,
                    checkpoint_step,
                    scheduler,
                )
            )

        if (
            backend_metadata.get("backend_checkpoint_error")
            and not self._allow_errors()
        ):
            raise RuntimeError(
                "backend checkpoint save failed: "
                f"{backend_metadata['backend_checkpoint_error']}"
            )
        if rank != 0 and (shard_mode or all_rank_backend_save):
            self._barrier()
            return {
                "step": checkpoint_step,
                "rank": rank,
                "world_size": world_size,
                "skipped": True,
                "reason": (
                    "backend checkpoint written; manifest is written by rank 0"
                    if all_rank_backend_save
                    else "checkpoint shard written; manifest is written by rank 0"
                ),
                "files": files,
            }
        if shard_mode and rank == 0:
            self._add_expected_fsdp_shards(files, world_size, state_dict_type)
        data_resume = getattr(self.engine, "data_resume", None)
        data_state = data_resume.capture() if data_resume is not None else {}
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
                "backend_state_written": backend_state_written,
                "backend_specific_checkpoint": backend_name in {"fsdp", "deepspeed"}
                and backend_state_written,
                "rank": rank,
                "world_size": world_size,
                "rank_count": world_size,
                "shard_count": (
                    world_size
                    if shard_mode or (all_rank_backend_save and backend_state_written)
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
        path = checkpoint_manager.write_manifest(manifest)
        self._barrier()
        return path

    def load(
        self,
        checkpoint_manager: Any,
        step: int,
        *,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
    ) -> Any:
        from parascale.runtime.backends import create_runtime_training_backend

        manifest = checkpoint_manager.read_manifest(step)
        self._validate_before_resume(checkpoint_manager, manifest)
        backend_state_loaded = False
        if model is not None or optimizer is not None:
            self.engine.training_backend = create_runtime_training_backend(
                model=model,
                optimizer=optimizer,
                config=self.engine.config,
                local_rank=self.engine._local_rank(),
            )
            self.engine.training_backend.setup()

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
            backend_state_loaded = self._load_portable_backend_state(
                checkpoint_manager,
                manifest,
                backend_entry,
                scheduler,
            )

        self.engine.state.global_step = int(manifest.global_step or manifest.step)
        self.engine.state.last_metrics = dict(manifest.metadata.get("last_metrics", {}))
        data_resume = getattr(self.engine, "data_resume", None)
        if data_resume is not None:
            data_resume.restore_manifest(manifest)
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
                    report.to_dict()
                    if hasattr(report, "to_dict")
                    else {"ok": False}
                )
                raise RuntimeError(
                    "Checkpoint validation failed before resume: " f"{details}"
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
            torch.save(payload, payload_path)
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
        if rng_state is not None:
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
        if payload.get("rng_state") is not None:
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
        state: Dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
        try:
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception as exc:
            state["torch_cuda_error"] = str(exc)
        return state

    @staticmethod
    def _restore_rng_state(torch: Any, state: Dict[str, Any]) -> None:
        if state.get("torch_cpu") is not None:
            torch.set_rng_state(state["torch_cpu"])
        if state.get("torch_cuda") is not None:
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            except Exception:
                pass

    def _rank(self) -> int:
        rank_fn = getattr(self.engine, "_distributed_rank", None)
        if callable(rank_fn):
            try:
                return int(rank_fn())
            except Exception:
                return 0
        return 0

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
