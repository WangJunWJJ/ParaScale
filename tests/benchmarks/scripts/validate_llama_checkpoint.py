# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午1:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Exact same-topology Llama continuation across independent torchrun processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

import parascale
from parascale.parallel import CausalLMLoss, DistributedParallelConfig
from parascale.parallel.checkpoint_context import canonical_checkpoint_context
from parascale.parallel.execution import DistributedTrainer
from parascale.parallel.llama import LlamaStage
from tests.benchmarks.scripts.benchmark_llama_parallel import file_digest


def state_fingerprint(value):
    """Hash finite local tensor bytes and typed structure without gathering shards."""
    digest = hashlib.sha256()

    def emit(item):
        encoded = json.dumps(item, sort_keys=True, allow_nan=False).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def visit(item):
        if isinstance(item, torch.Tensor):
            placements = (
                [str(p) for p in item.placements] if hasattr(item, "placements") else []
            )
            local = item.to_local() if hasattr(item, "to_local") else item
            local = local.detach().cpu().contiguous()
            if local.is_floating_point() and not torch.isfinite(local).all():
                raise ValueError("Cannot accept nonfinite checkpoint state")
            emit(
                [
                    "tensor",
                    str(item.dtype),
                    list(item.shape),
                    list(local.shape),
                    placements,
                ]
            )
            digest.update(memoryview(local.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, dict):
            emit(["dict", len(item)])
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            emit([type(item).__name__, len(item)])
            for child in item:
                visit(child)
        elif item is None or type(item) in (str, bool, int, float):
            emit([type(item).__name__, item])
        else:
            raise ValueError(f"Unsupported fingerprint value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def model_configuration(config):
    """Use HF's complete JSON representation at the third-party boundary."""
    return json.loads(config.to_json_string(use_diff=False))


def build_context(
    config, assets, text, encoded, *, global_batch, sequence_length, precision="fp32"
):
    """Declare effective third-party configuration, code and packed input identity."""
    return {
        "model": {
            "config": model_configuration(config),
            "assets": assets,
            "attention": "sdpa",
            "adapter": "LlamaStage-v1",
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "implementation_sha256": {
                path.name: file_digest(path)
                for path in sorted(
                    (Path(parascale.__file__).parent / "parallel").glob("*.py")
                )
            },
        },
        "loss": {
            "kind": "CausalLMLoss",
            "vocab_size": config.vocab_size,
            "ignore_index": -100,
            "shift": 1,
            "normalization": "global-valid-tokens",
        },
        "data": {
            "sha256": hashlib.sha256(text).hexdigest(),
            "token_stream_sha256": hashlib.sha256(
                torch.tensor(encoded, dtype=torch.int64).numpy().tobytes()
            ).hexdigest(),
            "packing": "contiguous-v1",
            "add_special_tokens": False,
            "global_batch": global_batch,
            "sequence_length": sequence_length,
            "shuffle": False,
        },
        "tokenizer": {
            name: digest for name, digest in assets.items() if "token" in name
        },
        "precision": precision,
    }


def load_resume_checkpoint(trainer, directory, optimizer, *, expected_step, steps):
    """Reject replay-from-zero and wrong checkpoint cursors before continuation."""
    restored = trainer.load_checkpoint(directory, optimizer)
    if (
        type(restored) is not int
        or type(expected_step) is not int
        or restored != expected_step
        or not 0 < restored < steps
    ):
        raise AssertionError(
            "Restored checkpoint step does not match the saved reference step"
        )
    return restored


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument(
        "--pipeline-schedule", choices=("gpipe", "1f1b"), default="gpipe"
    )
    parser.add_argument("--global-batch", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--save-step", type=int, default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint-dir", type=Path)
    mode.add_argument("--resume-from", type=Path)
    parser.add_argument("--reference-result", type=Path)
    args = parser.parse_args(argv)
    try:
        config = DistributedParallelConfig(
            args.dp,
            args.tp,
            args.pp,
            args.microbatches,
            pipeline_schedule=args.pipeline_schedule,
        )
    except ValueError as error:
        parser.error(str(error))
    config.validate_world_size(int(os.environ["WORLD_SIZE"]))
    if args.global_batch <= 0 or args.global_batch % (args.dp * args.microbatches):
        parser.error("global-batch must be a positive multiple of DP * microbatches")
    if args.sequence_length < 2 or not 0 < args.save_step < args.steps:
        parser.error("require sequence-length >= 2 and 0 < save-step < steps")
    if bool(args.resume_from) != bool(args.reference_result):
        parser.error(
            "resume requires reference-result, and reference runs must omit it"
        )
    if args.output.exists():
        parser.error("output already exists; use a fresh result path")
    return args, config


def main():
    args, config = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    try:
        rank = dist.get_rank()
        replica, stage_index, _ = config.coordinates(rank)
        torch.manual_seed(719)
        torch.backends.cuda.matmul.allow_tf32 = False
        source = LlamaForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        ).train()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        text = args.data.read_bytes()
        encoded = tokenizer(text.decode("utf-8"), add_special_tokens=False)["input_ids"]
        needed = args.steps * args.global_batch * args.sequence_length
        if len(encoded) < needed:
            raise ValueError(
                "Not enough original packed tokens for the requested steps"
            )
        tokens = torch.tensor(encoded[:needed], dtype=torch.int64).reshape(
            args.steps, args.global_batch, args.sequence_length
        )
        assets = [
            {
                path.name: file_digest(path)
                for path in sorted(args.model.iterdir())
                if path.is_file()
            }
            if rank == 0
            else None
        ]
        dist.broadcast_object_list(assets, src=0)
        context = build_context(
            source.config,
            assets[0],
            text,
            encoded,
            global_batch=args.global_batch,
            sequence_length=args.sequence_length,
            precision=args.precision,
        )
        context_hash = hashlib.sha256(
            canonical_checkpoint_context(context).encode()
        ).hexdigest()
        stage = LlamaStage(source, stage_index, args.pp, tp_size=args.tp).to(device)
        del source
        shape = (
            args.global_batch // args.dp // args.microbatches,
            args.sequence_length,
        )
        example = torch.zeros(
            shape if stage_index == 0 else (*shape, stage.hidden_size),
            device=device,
            dtype=torch.int64 if stage_index == 0 else torch.float32,
        )
        trainer = DistributedTrainer(
            stage,
            config,
            example,
            summed_loss=CausalLMLoss(stage.vocab_size),
            tensor_plan=stage.tensor_parallel_plan() or None,
            checkpoint_context=context,
            precision=args.precision,
        )
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-5, foreach=False)
        torch.manual_seed((1900 if args.resume_from else 9000) + rank)
        reference = None
        if args.reference_result:
            reference = json.loads(args.reference_result.read_text(encoding="utf-8"))
            if (
                reference["context_sha256"] != context_hash
                or reference["topology"] != asdict(config)
                or reference["steps"] != args.steps
                or reference["start"] != 0
                or reference["save_step"] != args.save_step
                or reference["status"] != "reference"
                or reference["precision"] != args.precision
                or len(reference["records"]) != dist.get_world_size()
            ):
                raise AssertionError("Reference run identity mismatch")
        start = (
            load_resume_checkpoint(
                trainer,
                args.resume_from,
                optimizer,
                expected_step=reference["save_step"],
                steps=args.steps,
            )
            if reference is not None
            else 0
        )
        losses = []
        for step in range(start, args.steps):
            batch = tokens[step].chunk(args.dp)[replica].to(device)
            optimizer.zero_grad(set_to_none=True)
            result = trainer.forward_backward(batch, batch)
            if not math.isfinite(result.loss):
                raise ValueError("Nonfinite training loss")
            optimizer.step()
            losses.append(result.loss)
            if args.checkpoint_dir and step + 1 == args.save_step:
                trainer.save_checkpoint(args.checkpoint_dir, optimizer, step=step + 1)
        record = {
            "rank": rank,
            "losses": losses,
            "model": state_fingerprint(trainer.module.state_dict()),
            "optimizer": state_fingerprint(optimizer.state_dict()),
            "rng": state_fingerprint(
                {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
            ),
        }
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, record)
        if reference is not None:
            for actual, expected in zip(records, reference["records"]):
                for key in ("rank", "model", "optimizer", "rng"):
                    if actual[key] != expected[key]:
                        raise AssertionError(
                            f"Continuation mismatch rank={actual['rank']} field={key}"
                        )
                if actual["losses"] != expected["losses"][start:]:
                    raise AssertionError(
                        f"Continuation loss mismatch rank={actual['rank']}"
                    )
        if rank == 0:
            report = {
                "status": "passed" if args.resume_from else "reference",
                "scope": "Exact same-topology packed-token continuation with FP32 master parameters, not convergence or fault recovery",
                "precision": trainer.precision,
                "parameter_dtype": "float32",
                "context_sha256": context_hash,
                "topology": asdict(config),
                "pipeline_schedule": config.pipeline_schedule,
                "steps": args.steps,
                "start": start,
                "save_step": args.save_step,
                "records": records,
                "script_sha256": file_digest(Path(__file__)),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                json.dumps(
                    {"status": report["status"], "start": start, "steps": args.steps}
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
