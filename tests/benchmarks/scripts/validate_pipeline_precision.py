# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CUDA PP boundary failures and BF16 forwarding against a native full model."""

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from parascale.parallel import DistributedParallelConfig
from parascale.parallel.execution import DistributedTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = DistributedParallelConfig(dp_size=args.dp, pp_size=2, microbatches=2)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    try:
        config.validate_world_size(dist.get_world_size())
        replica, pp, _ = config.coordinates(dist.get_rank())
        torch.manual_seed(77)
        reference = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).to(device)
        microbatch = 8 // args.dp // config.microbatches
        rejected = []
        for invalid in ("dtype", "shape"):
            # Linear accepts either batch size, so both local inferences succeed.
            example = torch.zeros(
                microbatch + int(pp == 1 and invalid == "shape"),
                8,
                device=device,
                dtype=torch.float32
                if pp == 0 or invalid == "dtype"
                else torch.bfloat16,
            )
            try:
                DistributedTrainer(
                    copy.deepcopy(reference[pp]),
                    config,
                    example,
                    torch.nn.MSELoss(),
                    precision="bf16",
                )
            except ValueError as error:
                assert "pipeline boundary" in str(error)
                rejected.append(invalid)
            else:
                raise AssertionError(f"Mismatched {invalid} was accepted")
        trainer = DistributedTrainer(
            copy.deepcopy(reference[pp]),
            config,
            torch.zeros(
                microbatch,
                8,
                device=device,
                dtype=torch.float32 if pp == 0 else torch.bfloat16,
            ),
            torch.nn.MSELoss(),
            precision="bf16",
        )
        optimizer = torch.optim.SGD(trainer.parameters(), lr=0.01)
        baseline = torch.optim.SGD(reference.parameters(), lr=0.01)
        losses = []
        for _ in range(3):
            inputs, targets = (
                torch.randn(8, 8, device=device),
                torch.randn(8, 8, device=device),
            )
            optimizer.zero_grad()
            baseline.zero_grad()
            expected_outputs, expected_losses = [], []
            for x, y in zip(inputs.split(microbatch), targets.split(microbatch)):
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    prediction = reference(x)
                    loss = torch.nn.functional.mse_loss(prediction, y) / (
                        8 // microbatch
                    )
                loss.backward()
                expected_outputs.append(prediction.detach())
                expected_losses.append(loss.detach())
            result = trainer.forward_backward(
                inputs.chunk(args.dp)[replica],
                targets.chunk(args.dp)[replica],
                return_outputs=True,
            )
            torch.testing.assert_close(
                torch.tensor(result.loss, device=device),
                torch.stack(expected_losses).sum(),
            )
            if pp == 1:
                assert result.output.dtype == torch.bfloat16
                torch.testing.assert_close(
                    result.output,
                    torch.cat(expected_outputs).chunk(args.dp)[replica],
                    rtol=0,
                    atol=0,
                )
            for actual, expected in zip(
                trainer.parameters(), reference[pp].parameters()
            ):
                assert actual.dtype == actual.grad.dtype == torch.float32
                torch.testing.assert_close(
                    actual.grad, expected.grad, rtol=1e-5, atol=1e-7
                )
            optimizer.step()
            baseline.step()
            for actual, expected in zip(
                trainer.parameters(), reference[pp].parameters()
            ):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            losses.append(result.loss)
        records = [None] * config.world_size
        dist.all_gather_object(
            records, {"rank": rank, "rejected": rejected, "losses": losses}
        )
        if rank == 0:
            result = {
                "status": "passed",
                "dp": args.dp,
                "pp": 2,
                "precision": "bf16",
                "records": records,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
