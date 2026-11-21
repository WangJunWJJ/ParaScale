# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Compare complete Llama stage training with a native full-model reference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from parascale.parallel import CausalLMLoss, DistributedParallelConfig
from parascale.parallel.execution import DistributedTrainer
from parascale.parallel.llama import LlamaStage
from tests.benchmarks.scripts.benchmark_llama_parallel import native_tensor_plan
from tests.benchmarks.tools.parallel_numerics import GATES, compare_training_tensor


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_tensor(value):
    return value.full_tensor() if hasattr(value, "full_tensor") else value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument(
        "--pipeline-schedule", choices=("gpipe", "1f1b"), default="gpipe"
    )
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Matched AdamW epsilon for both paths; changing it requires quality validation",
    )
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--reference-layout", choices=("unsharded", "matched-tp"), default="unsharded"
    )
    parser.add_argument("--native-forward-control", action="store_true")
    parser.add_argument(
        "--output-policy", choices=("retain", "discard", "alternate"), default="retain"
    )
    parser.add_argument(
        "--logits-atol",
        type=float,
        default=3e-6,
        help="Explicit output tolerance; calibrate against native controls",
    )
    parser.add_argument(
        "--observe-errors",
        action="store_true",
        help="Record strict-gate failures without declaring validation passed",
    )
    parser.add_argument("--output", type=Path, required=True)
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
    if args.global_batch % (args.dp * args.microbatches):
        parser.error("global batch must be divisible by DP * microbatches")
    if args.steps < 2 or args.sequence_length < 2:
        parser.error("steps and sequence length must be at least two")
    if (args.model_path is None) != (args.data_file is None):
        parser.error("real validation requires both model-path and data-file")
    if args.native_forward_control and args.pp != 1:
        parser.error("native-forward-control requires a single pipeline stage")
    if not math.isfinite(args.logits_atol) or args.logits_atol < 0:
        parser.error("logits-atol must be finite and nonnegative")
    if not math.isfinite(args.adam_eps) or args.adam_eps <= 0:
        parser.error("adam-eps must be finite and positive")
    return args, config


def main():
    args, config = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    try:
        config.validate_world_size(dist.get_world_size())
        replica, stage_index, _ = config.coordinates(dist.get_rank())
        torch.manual_seed(719)
        torch.backends.cuda.matmul.allow_tf32 = False
        if args.model_path is None:
            model_config = LlamaConfig(
                vocab_size=31,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                tie_word_embeddings=False,
                attention_dropout=0.0,
            )
            model_config._attn_implementation = args.attention
            source = LlamaForCausalLM(model_config).float().train()
            tokens = torch.randint(
                0, 31, (args.steps, args.global_batch, args.sequence_length)
            )
        else:
            source = LlamaForCausalLM.from_pretrained(
                args.model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation=args.attention,
            ).train()
            model_config = source.config
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, local_files_only=True
            )
            encoded = tokenizer(
                args.data_file.read_text(encoding="utf-8"), add_special_tokens=False
            )["input_ids"]
            needed = args.steps * args.global_batch * args.sequence_length
            if len(encoded) < needed:
                raise ValueError("data file has insufficient unique packed tokens")
            tokens = torch.tensor(encoded[:needed], dtype=torch.int64).reshape(
                args.steps, args.global_batch, args.sequence_length
            )
        token_digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
        reference = copy.deepcopy(source).to(device)
        stage = LlamaStage(source, stage_index, args.pp, tp_size=args.tp).to(device)
        if args.native_forward_control:
            native_source = source.to(device)
            stage.forward = lambda inputs: native_source(inputs, use_cache=False).logits
        del source
        batch = args.global_batch // args.dp
        example_shape = (batch // args.microbatches, args.sequence_length)
        example = torch.zeros(
            example_shape
            if stage_index == 0
            else (*example_shape, model_config.hidden_size),
            device=device,
            dtype=torch.int64 if stage_index == 0 else torch.float32,
        )
        objective = CausalLMLoss(model_config.vocab_size)
        trainer = DistributedTrainer(
            stage,
            config,
            example,
            summed_loss=objective,
            tensor_plan=stage.tensor_parallel_plan() or None,
            precision=args.precision,
        )
        if args.reference_layout == "matched-tp" and args.tp > 1:
            from torch.distributed.tensor.parallel import parallelize_module

            parallelize_module(
                reference,
                trainer.mesh["tp"],
                native_tensor_plan(model_config.num_hidden_layers),
            )
        optimizer = torch.optim.AdamW(
            trainer.parameters(), lr=1e-5, eps=args.adam_eps, foreach=False
        )
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-5, eps=args.adam_eps, foreach=False
        )
        expected_parameters = dict(reference.named_parameters())
        errors = dict.fromkeys(GATES, 0.0)
        relative_errors = dict.fromkeys(errors, 0.0)
        mismatches = dict.fromkeys(errors, 0)
        worst = {}
        losses = []

        def compare(actual, expected, key, name=""):
            actual = full_tensor(actual).detach()
            expected = full_tensor(expected).detach()
            statistics = compare_training_tensor(
                actual, expected, key, logits_atol=args.logits_atol
            )
            if (
                key not in worst
                or statistics["relative_l2"] > worst[key]["relative_l2"]
            ):
                worst[key] = statistics | {"name": name, "step": step}
            errors[key] = max(errors[key], statistics["max_abs"])
            relative_errors[key] = max(relative_errors[key], statistics["relative_l2"])
            if not statistics["passed"]:
                mismatches[key] += 1
                if args.observe_errors:
                    return
                raise AssertionError(
                    f"rank={dist.get_rank()} step={step} {name} {key}: {statistics}"
                )

        for step, cpu_tokens in enumerate(tokens):
            inputs = cpu_tokens.to(device)
            targets = inputs.clone()
            # Alternate locally empty replicas while retaining a positive global loss.
            if step % 2 == 0:
                targets[: args.global_batch // 2] = -100
            else:
                targets[args.global_batch // 2 :] = -100
            targets[0, 2:] = -100
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            count = objective.normalization_count(targets)
            expected_outputs, reference_losses = [], []
            # Match GEMM batch shapes while retaining one unsharded global objective.
            microbatch_size = batch // args.microbatches
            for token_batch, target_batch in zip(
                inputs.split(microbatch_size), targets.split(microbatch_size)
            ):
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=args.precision == "bf16",
                    cache_enabled=False,
                ):
                    prediction = reference(token_batch, use_cache=False).logits
                    reference_loss = objective(prediction, target_batch) / count
                reference_loss.backward()
                expected_outputs.append(prediction.detach())
                reference_losses.append(reference_loss.detach())
            expected_output = torch.cat(expected_outputs)
            expected_loss = torch.stack(reference_losses).sum()
            retain_outputs = args.output_policy == "retain" or (
                args.output_policy == "alternate" and step % 2 == 1
            )
            result = trainer.forward_backward(
                inputs.chunk(args.dp)[replica],
                targets.chunk(args.dp)[replica],
                return_outputs=retain_outputs,
            )
            assert (result.output is not None) == (
                retain_outputs and stage_index == args.pp - 1
            )
            assert result.normalization_count == count.item()
            compare(torch.tensor(result.loss, device=device), expected_loss, "loss")
            if result.output is not None:
                assert result.output.dtype == (
                    torch.bfloat16 if args.precision == "bf16" else torch.float32
                )
                compare(
                    result.output, expected_output.chunk(args.dp)[replica], "output"
                )
            for name, parameter in trainer.named_parameters():
                assert parameter.dtype == parameter.grad.dtype == torch.float32
                compare(
                    parameter.grad, expected_parameters[name].grad, "gradient", name
                )
            before = {
                name: full_tensor(parameter).detach().clone()
                for name, parameter in trainer.named_parameters()
            }
            reference_before = {
                name: full_tensor(expected_parameters[name]).detach().clone()
                for name in before
            }
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in trainer.named_parameters():
                expected = expected_parameters[name]
                canonical_parameter = full_tensor(parameter).detach()
                compare(canonical_parameter, expected, "parameter", name)
                compare(
                    canonical_parameter - before[name],
                    full_tensor(expected).detach() - reference_before[name],
                    "update",
                    name,
                )
                assert (
                    optimizer.state[parameter].keys()
                    == reference_optimizer.state[expected].keys()
                )
                for key, value in optimizer.state[parameter].items():
                    assert value.dtype == torch.float32
                    compare(value, reference_optimizer.state[expected][key], key, name)
            del before, reference_before
            losses.append(result.loss)
        optimizer.zero_grad(set_to_none=True)
        invalid_inputs = inputs.chunk(args.dp)[replica].clone()
        if dist.get_rank() == 0:
            invalid_inputs[0, 0] = model_config.vocab_size
        try:
            trainer.forward_backward(invalid_inputs, targets.chunk(args.dp)[replica])
        except ValueError as error:
            assert "preflight" in str(error)
        else:
            raise AssertionError("Invalid input token was accepted")
        assert all(parameter.grad is None for parameter in trainer.parameters())
        dist.barrier()
        records = [None] * config.world_size
        dist.all_gather_object(
            records,
            {
                "rank": dist.get_rank(),
                "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "max_abs_error": errors,
                "max_tensor_relative_l2": relative_errors,
                "strict_gate_mismatches": mismatches,
                "worst_tensor_by_relative_l2": worst,
                "losses": losses,
            },
        )
        if dist.get_rank() == 0:
            import parascale

            source_root = Path(parascale.__file__).resolve().parent
            result = {
                "status": "diagnostic"
                if args.observe_errors or args.native_forward_control
                else "passed",
                "numerical_gates": GATES,
                "pipeline_schedule": config.pipeline_schedule,
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "input_sha256": token_digest,
                "asset_sha256": {
                    path.name: file_digest(path)
                    for path in sorted(args.model_path.iterdir())
                    if path.is_file()
                }
                if args.model_path
                else {},
                "data_sha256": file_digest(args.data_file) if args.data_file else None,
                "source_sha256": {
                    str(path.relative_to(source_root)): hashlib.sha256(
                        path.read_bytes().replace(b"\r\n", b"\n")
                    ).hexdigest()
                    for path in sorted(source_root.rglob("*.py"))
                },
                "validator_sha256": file_digest(Path(__file__)),
                "gates_sha256": file_digest(
                    Path(__file__).parents[1] / "tools" / "parallel_numerics.py"
                ),
                "reference_microbatch_size": microbatch_size,
                "reference_plan_sha256": file_digest(
                    Path(__file__).with_name("benchmark_llama_parallel.py")
                ),
                "model_config": model_config.to_dict(),
                "versions": {
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "ranks": records,
                "scope": "correctness comparison including full reference; not throughput or convergence evidence",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "dp": args.dp,
                        "tp": args.tp,
                        "pp": args.pp,
                        "ranks": records,
                    },
                    allow_nan=False,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
