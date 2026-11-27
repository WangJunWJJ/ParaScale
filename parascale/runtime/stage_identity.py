# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Detached effective-model and local-asset identity for stage recovery."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from parascale.parallel.checkpoint_context import canonical_checkpoint_context
from parascale.parallel.distributed_checkpoint import _digest
from parascale.workloads.causal_lm_assets import model_asset_hashes as _model_assets


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = set((root / "parallel").rglob("*.py"))
    paths.update(
        root / relative
        for relative in (
            "workloads/causal_lm.py",
            "workloads/causal_lm_model.py",
            "workloads/causal_lm_assets.py",
            "workloads/causal_lm_evaluation.py",
            "workloads/specs/causal_lm.py",
            "communication/hooks.py",
            "core/distributed/coordination.py",
            "data/text/__init__.py",
            "data/text/artifact.py",
            "data/text/batches.py",
            "optimizers/__init__.py",
            "optimizers/factory.py",
            "optimizers/spec.py",
            "optimizers/schedule.py",
            "runtime/stage_identity.py",
            "runtime/stage_plan.py",
            "runtime/training/data_state.py",
            "runtime/training/random_state.py",
            "runtime/training/stage_checkpoint.py",
            "runtime/training/stage_evaluation.py",
        )
    )
    for relative in ("runtime/stage_train_runner.py", "runtime/training/stage_loop.py"):
        if (root / relative).is_file():
            paths.add(root / relative)
    return {path.relative_to(root).as_posix(): _digest(path) for path in sorted(paths)}


def build_stage_checkpoint_context(
    stage: Any, cursor: Any, plan: Any
) -> dict[str, Any]:
    """Snapshot rank-independent semantics; never retain modules or load tensors.

    Model assets must remain immutable between source loading and this snapshot.
    Only the default local HF safetensors/bin formats admitted by the workload
    are supported. Topology and cursor position belong to the checkpoint runtime.
    """
    import torch
    import transformers

    config = getattr(stage, "effective_model_config", None)
    if not isinstance(config, dict) or not config:
        raise ValueError("Stage requires a nonempty effective model config")
    # Checkpoint loading restores saved hyperparameters; bind requested values too.
    optimizer = json.loads(
        json.dumps(asdict(plan.optimizer_spec), sort_keys=True, allow_nan=False)
    )
    context = {
        "optimizer": optimizer,
        "model": {
            "config": config,
            "assets_sha256": _model_assets(Path(plan.workload.model_path)),
        },
        "tokenizer": cursor.corpus.manifest["tokenizer"],
        "data": cursor.corpus.identity,
        "loss": {
            "type": "causal_next_token_v1",
            "shift": 1,
            "ignore_index": -100,
            "vocab_size": stage.vocab_size,
        },
        "precision": plan.config.precision,
        "execution": {
            "ddp_comm_hook": plan.ddp_comm_hook,
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "implementation_sha256": _implementation_hashes(),
        },
    }
    return json.loads(canonical_checkpoint_context(context))
