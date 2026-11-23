# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CUDA token-loss/gradient contracts; an untied diagnostic, not a pretrained LM."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from parascale.parallel import CausalLMLoss, DistributedParallelConfig
from parascale.parallel.execution import DistributedTrainer


class Projection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up = nn.Linear(width, width * 4, bias=False)
        self.down = nn.Linear(width * 4, width, bias=False)

    def forward(self, values):
        return values + self.down(torch.nn.functional.gelu(self.up(values)))


def full_tensor(tensor):
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = DistributedParallelConfig(args.dp, args.tp, args.pp, args.microbatches)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    try:
        config.validate_world_size(dist.get_world_size())
        replica, stage_index, _ = config.coordinates(dist.get_rank())
        torch.manual_seed(71)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = nn.Sequential(
            OrderedDict(
                [
                    ("embedding", nn.Embedding(31, 16)),
                    ("block0", Projection(16)),
                    ("block1", Projection(16)),
                    ("head", nn.Linear(16, 31, bias=False)),
                ]
            )
        )
        reference = copy.deepcopy(model).to(device)
        names = list(model._modules)
        stage_names = names[
            len(names) * stage_index // args.pp : len(names)
            * (stage_index + 1)
            // args.pp
        ]
        stage = nn.Sequential(
            OrderedDict((name, model._modules[name]) for name in stage_names)
        ).to(device)
        tensor_plan = {}
        for name in stage_names:
            if name.startswith("block"):
                tensor_plan[f"{name}.up"] = ColwiseParallel()
                tensor_plan[f"{name}.down"] = RowwiseParallel()
        local_batch = 8 // args.dp
        assert local_batch % args.microbatches == 0
        example_shape = (local_batch // args.microbatches, 5)
        example = torch.zeros(
            example_shape if stage_index == 0 else (*example_shape, 16),
            dtype=torch.int64 if stage_index == 0 else torch.float32,
            device=device,
        )
        objective = CausalLMLoss(31)
        for invalid in ("dtype", "loss_mode", "microbatches"):
            bad_example = example
            bad_config = config
            loss_options = {"summed_loss": objective}
            if dist.get_rank() == 0:
                if invalid == "dtype":
                    bad_example = example.half()
                elif invalid == "loss_mode":
                    loss_options = {"loss_fn": nn.MSELoss()}
                else:
                    bad_config = DistributedParallelConfig(
                        args.dp, args.tp, args.pp, args.microbatches + 1
                    )
            if config.world_size == 1 and invalid != "dtype":
                continue
            try:
                DistributedTrainer(
                    copy.deepcopy(stage),
                    bad_config,
                    bad_example,
                    tensor_plan=tensor_plan if args.tp > 1 else None,
                    **loss_options,
                )
            except ValueError as error:
                assert "preflight" in str(error)
            else:
                raise AssertionError(f"Constructor accepted {invalid}")
            dist.barrier()
        trainer = DistributedTrainer(
            stage,
            config,
            example,
            summed_loss=objective,
            tensor_plan=tensor_plan if args.tp > 1 else None,
        )
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-3, foreach=False)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-3, foreach=False
        )
        errors = dict(loss=0.0, output=0.0, gradient=0.0, parameter=0.0, optimizer=0.0)

        def compare(actual, expected, key):
            actual = full_tensor(actual).detach()
            expected = expected.detach()
            error = (actual - expected).abs().max().item()
            errors[key] = max(errors[key], error)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)

        for step in range(4):
            generator = torch.Generator(device=device).manual_seed(91 + step)
            inputs = torch.randint(0, 31, (8, 5), generator=generator, device=device)
            targets = inputs.clone()
            if step % 2 == 0:
                targets[:4] = -100
                targets[4, 2:] = -100
            else:
                targets[4:] = -100
                targets[0, 2:] = -100
            reference_optimizer.zero_grad(set_to_none=True)
            expected_output = reference(inputs)
            count = objective.normalization_count(targets)
            expected_loss = objective(expected_output, targets) / count
            expected_loss.backward()
            optimizer.zero_grad(set_to_none=True)
            local_inputs = inputs.chunk(args.dp)[replica]
            local_targets = targets.chunk(args.dp)[replica]
            result = trainer.forward_backward(
                local_inputs, local_targets, return_outputs=True
            )
            assert result.normalization_count == count.item()
            compare(torch.tensor(result.loss, device=device), expected_loss, "loss")
            if result.output is not None:
                compare(
                    result.output, expected_output.chunk(args.dp)[replica], "output"
                )
            expected_parameters = dict(reference.named_parameters())
            for name, parameter in trainer.named_parameters():
                compare(parameter.grad, expected_parameters[name].grad, "gradient")
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in trainer.named_parameters():
                expected_parameter = expected_parameters[name]
                compare(parameter, expected_parameter, "parameter")
                for key, value in optimizer.state[parameter].items():
                    compare(
                        value,
                        reference_optimizer.state[expected_parameter][key],
                        "optimizer",
                    )

        invalid_cases = ["empty", "rank_label", "rank_input"]
        if args.tp > 1:
            invalid_cases.extend(["tp_empty", "tp_count"])
        for invalid in invalid_cases:
            optimizer.zero_grad(set_to_none=True)
            bad_targets = local_targets.clone()
            bad_inputs = local_inputs
            if invalid == "empty":
                bad_targets.fill_(-100)
            if invalid == "rank_label" and dist.get_rank() == config.world_size - 1:
                bad_targets[0, 1] = 31
            if invalid == "rank_input" and dist.get_rank() == 0:
                bad_inputs = bad_inputs.float()
            if invalid.startswith("tp_"):
                bad_targets = local_inputs.clone()
                if trainer.tp_rank == 0:
                    if invalid == "tp_empty":
                        bad_targets.fill_(-100)
                    else:
                        bad_targets[0, 1] = -100
            try:
                trainer.forward_backward(bad_inputs, bad_targets)
            except ValueError as error:
                assert "normalization_count" in str(error) or "preflight" in str(error)
            else:
                raise AssertionError(f"Rank {dist.get_rank()} accepted {invalid}")
            assert all(parameter.grad is None for parameter in trainer.parameters())
            dist.barrier()
        records = [None] * config.world_size
        dist.all_gather_object(
            records, {"rank": dist.get_rank(), "max_abs_error": errors}
        )
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "status": "passed",
                "topology": vars(args) | {"output": str(args.output)},
                "ranks": records,
                "scope": "untied token projection fixture; not full pretrained TP/PP or a throughput claim",
            }
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
            print(json.dumps(result, allow_nan=False))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
