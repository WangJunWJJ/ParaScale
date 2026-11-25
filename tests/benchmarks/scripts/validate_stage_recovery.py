# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Fresh-process, exact same-topology recovery on a pretrained Llama token corpus.

Run baseline then resume in separate torchrun launches with identical arguments.
The baseline saves after update2 and records update3; resume must reproduce it.
This is a bounded recovery contract, not a convergence or throughput benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


def training_fingerprint(model, optimizer, *, steps, initial):
    """Reject empty/no-op/nonfinite training before accepting exact recovery."""
    import torch

    from tests.benchmarks.scripts.validate_llama_checkpoint import state_fingerprint

    parameters = dict(model.named_parameters())
    trainable = [
        parameter for parameter in parameters.values() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Recovery validation requires trainable parameters")
    for parameter in trainable:
        state = optimizer.state.get(parameter)
        if not state or not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError("Missing Adam optimizer state after training")
        counter = state["step"]
        if (
            not isinstance(counter, torch.Tensor)
            or counter.numel() != 1
            or counter.item() != steps
        ):
            raise ValueError("Optimizer update count mismatch")
        if any(
            state[key].shape != parameter.shape for key in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError("Optimizer moment shape mismatch")
    model_hash = state_fingerprint(parameters)
    if model_hash == initial:
        raise ValueError("Model weights did not change during training")
    return {"model": model_hash, "optimizer": state_fingerprint(optimizer.state_dict())}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "resume"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument(
        "--pipeline-schedule", choices=("gpipe", "1f1b"), default="gpipe"
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--validate-catalog", action="store_true")
    parser.add_argument("--checkpoint-format", choices=("rank", "dcp"), default="rank")
    parser.add_argument("--checkpoint-state-probe", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torch.distributed as dist

    from parascale.optimizers import build_optimizer
    from parascale.optimizers.schedule import SchedulerSpec
    from parascale.parallel import DistributedParallelConfig
    from parascale.parallel.checkpoint_catalog import (
        latest_stage_checkpoint,
        prune_stage_checkpoints,
    )
    from parascale.parallel.execution import DistributedTrainer
    from parascale.runtime.training.coordination import coordinated_call
    from parascale.runtime.training.data_state import DataResumeController
    from parascale.runtime.training.engine import TrainState
    from parascale.runtime.training.stage_checkpoint import StageTrainingCheckpoint
    from parascale.workloads import build_causal_lm_stage_components
    from tests.benchmarks.scripts.validate_llama_checkpoint import state_fingerprint

    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    layout = DistributedParallelConfig(
        args.dp,
        args.tp,
        args.pp,
        microbatches=2,
        pipeline_schedule=args.pipeline_schedule,
    )
    layout.validate_world_size(dist.get_world_size())
    try:

        def validate_catalog_destination():
            if (
                args.validate_catalog
                and args.mode == "baseline"
                and args.checkpoint_dir.exists()
            ):
                raise ValueError("Catalog validation requires a new checkpoint root")

        coordinated_call(validate_catalog_destination, phase="catalog destination")
        config = {
            "model": {"pretrained_model_name_or_path": str(args.model_path)},
            "data": {
                "type": "token_corpus",
                "path": str(args.corpus_path),
                "batch_size": 1,
                "sequence_length": args.sequence_length,
                "seed": 42,
            },
            "optimizer": {"type": "adamw", "lr": 1e-5},
        }
        stage, cursor, loss = build_causal_lm_stage_components(
            config, layout, rank=rank, world_size=dist.get_world_size()
        )
        stage.to(device)
        pipeline = layout.coordinates(rank)[1]
        example = torch.zeros(
            (1, args.sequence_length)
            if pipeline == 0
            else (1, args.sequence_length, stage.hidden_size),
            dtype=torch.int64 if pipeline == 0 else torch.float32,
            device=device,
        )
        model_hash = hashlib.sha256()
        for path in sorted(args.model_path.glob("*.safetensors")):
            model_hash.update(path.name.encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    model_hash.update(block)
        context = {
            "model": {
                "config": json.loads((args.model_path / "config.json").read_text()),
                "safetensors_sha256": model_hash.hexdigest(),
            },
            "data": cursor.corpus.identity,
            "tokenizer": cursor.corpus.manifest["tokenizer"],
            "loss": "causal_next_token_v1",
            "precision": "bf16",
        }
        trainer = DistributedTrainer(
            stage,
            layout,
            example,
            summed_loss=loss,
            tensor_plan=stage.tensor_parallel_plan() or None,
            precision="bf16",
            checkpoint_context=context,
        )
        optimizer = build_optimizer(stage, config)
        spec = SchedulerSpec("cosine", total_steps=8, warmup_steps=1)
        engine = SimpleNamespace(
            state=TrainState(),
            scheduler=spec.build(optimizer),
            scheduler_metadata=spec.to_metadata(),
        )
        engine.data_resume = DataResumeController(engine)
        runtime = StageTrainingCheckpoint(
            engine,
            cursor,
            optimizer,
            microbatches=layout.microbatches,
            scheduler_spec=spec,
        )
        initial_model = state_fingerprint(dict(stage.named_parameters()))

        def inspect_saved_state():
            if not args.checkpoint_state_probe:
                return
            value = training_fingerprint(
                stage, optimizer, steps=2, initial=initial_model
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            reference = args.output_dir / f"rank-{rank:05d}-checkpoint-baseline.json"
            target = args.output_dir / f"rank-{rank:05d}-checkpoint-{args.mode}.json"
            target.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            if args.mode == "resume" and json.loads(reference.read_text()) != value:
                raise AssertionError(
                    "Loaded model/optimizer differs before the next update"
                )

        seed = (812 if args.mode == "baseline" else 9999) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if args.mode == "resume":
            source = args.checkpoint_dir
            if args.validate_catalog:
                selected, skipped = coordinated_call(
                    lambda: latest_stage_checkpoint(args.checkpoint_dir),
                    phase="catalog discovery",
                    identity=lambda result: (str(result[0]), result[1]),
                )
                if skipped or selected.name != "step-00000002":
                    raise AssertionError("Catalog did not select update2")
                source = selected
            restored = trainer.load_checkpoint(source, optimizer, runtime=runtime)
            if restored != 2:
                raise AssertionError("Expected update2 checkpoint")
            coordinated_call(inspect_saved_state, phase="loaded state fingerprint")
        iterator = engine.data_resume.prepare_iterator(cursor)
        if args.mode == "baseline" and args.validate_catalog:
            trainer.save_checkpoint(
                args.checkpoint_dir / "step-00000000",
                optimizer,
                step=0,
                runtime=runtime,
                checkpoint_format=args.checkpoint_format,
            )
        ahead = None
        updates = []
        while engine.state.global_step < 3:
            parts = [next(iterator) if ahead is None else ahead, next(iterator)]
            ahead = None
            draws = [
                random.random(),
                float(np.random.random()),
                torch.rand(()).item(),
                torch.rand((), device=device).item(),
            ]
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.cat([part["input_ids"] for part in parts]).to(device)
            labels = torch.cat([part["labels"] for part in parts]).to(device)
            result = trainer.forward_backward(inputs, labels, return_outputs=False)
            optimizer.step()
            engine.scheduler.step()
            engine.state.global_step += 1
            engine.data_resume.record_consumption(
                {"gradient_accumulation_steps": 2, "batch_size": 2}
            )
            updates.append(
                {
                    "step": engine.state.global_step,
                    "loss": result.loss,
                    "blocks": [block for part in parts for block in part["block_ids"]],
                    "rng_draws": draws,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"rank={rank} mode={args.mode} step={engine.state.global_step} loss={result.loss}",
                flush=True,
            )
            if args.mode == "baseline" and engine.state.global_step == 2:
                ahead = next(iterator)
                trainer.save_checkpoint(
                    args.checkpoint_dir / "step-00000002"
                    if args.validate_catalog
                    else args.checkpoint_dir,
                    optimizer,
                    step=2,
                    runtime=runtime,
                    checkpoint_format=args.checkpoint_format,
                )
                coordinated_call(inspect_saved_state, phase="saved state fingerprint")
                if args.validate_catalog:

                    def prune_and_check():
                        if rank != 0:
                            return
                        report = prune_stage_checkpoints(
                            args.checkpoint_dir / "step-00000002", keep_last=2
                        )
                        if (
                            report["removed"] != [0]
                            or report["protected"] != [2, 1]
                            or report["skipped"]
                        ):
                            raise AssertionError(
                                f"Unexpected retention report: {report}"
                            )
                        args.output_dir.mkdir(parents=True, exist_ok=True)
                        (args.output_dir / "catalog-retention.json").write_text(
                            json.dumps(report, indent=2), encoding="utf-8"
                        )

                    coordinated_call(prune_and_check, phase="catalog retention")
            elif (
                args.mode == "baseline"
                and args.validate_catalog
                and engine.state.global_step == 1
            ):
                trainer.save_checkpoint(
                    args.checkpoint_dir / "step-00000001",
                    optimizer,
                    step=1,
                    runtime=runtime,
                    checkpoint_format=args.checkpoint_format,
                )

        def digest_tensor(value):
            local = value.to_local() if hasattr(value, "to_local") else value
            local = local.detach().cpu().contiguous()
            return {
                "shape": list(local.shape),
                "dtype": str(local.dtype),
                "sha256": hashlib.sha256(
                    local.reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }

        error = None
        try:
            fingerprints = training_fingerprint(
                stage, optimizer, steps=3, initial=initial_model
            )
        except Exception as exc:
            error = f"rank {rank}: {exc}"
        errors = [None] * layout.world_size
        dist.all_gather_object(errors, error)
        if any(errors):
            raise AssertionError(str(errors))
        result = {
            "update": updates[-1],
            "parameters_and_optimizer": fingerprints,
            "runtime": runtime.capture(),
            "cpu_rng": digest_tensor(torch.get_rng_state()),
            "cuda_rng": digest_tensor(torch.cuda.get_rng_state(device)),
        }
        serialized = json.dumps(result, sort_keys=True, allow_nan=False)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        reference = args.output_dir / f"rank-{rank:05d}-baseline.json"
        errors = [None] * layout.world_size
        error = None
        if args.mode == "resume":
            try:
                (args.output_dir / f"rank-{rank:05d}-resume.json").write_text(
                    serialized, encoding="utf-8"
                )
                if reference.read_text(encoding="utf-8") != serialized:
                    expected = json.loads(reference.read_text(encoding="utf-8"))
                    observed = json.loads(serialized)
                    changed = [
                        key for key in observed if observed[key] != expected.get(key)
                    ]
                    raise AssertionError(
                        f"Resumed update/state differs from uninterrupted reference: {changed}"
                    )
            except Exception as exc:
                error = f"rank {rank}: {exc}"
        dist.all_gather_object(errors, error)
        if any(errors):
            raise AssertionError(str(errors))
        (args.output_dir / f"rank-{rank:05d}-{args.mode}.json").write_text(
            serialized, encoding="utf-8"
        )
        if rank == 0:
            (args.output_dir / f"{args.mode}-summary.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "mode": args.mode,
                        "dp": args.dp,
                        "tp": args.tp,
                        "pp": args.pp,
                        "pipeline_schedule": args.pipeline_schedule,
                        "checkpoint_format": args.checkpoint_format,
                        "precision": "bf16",
                        "updates": updates,
                        "scope": "exact fresh-process update3 recovery"
                        if args.mode == "resume"
                        else "reference capture",
                        "torch": torch.__version__,
                        "model": context["model"],
                        "data": context["data"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
