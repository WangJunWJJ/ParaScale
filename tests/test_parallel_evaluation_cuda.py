# -*- coding: utf-8 -*-
# @Time : 2026/11/26 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real CUDA Llama evaluation and subsequent training across admitted layouts."""

import copy
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
dist = torch.distributed


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def worker(rank, axes, schedule, precision, policy, folder):
    from transformers import LlamaConfig, LlamaForCausalLM

    from parascale.parallel.distributed import DistributedParallelConfig
    from parascale.parallel.execution import DistributedTrainer
    from parascale.parallel.llama import LlamaStage
    from parascale.parallel.loss import CausalLMLoss
    from parascale.parallel.precision import autocast_context

    config = DistributedParallelConfig(
        *axes, microbatches=2, pipeline_schedule=schedule
    )
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        init_method=(Path(folder) / "rendezvous").as_uri(),
        rank=rank,
        world_size=config.world_size,
        device_id=device,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(491)
        model_config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
        model_config._attn_implementation = "eager"
        source = LlamaForCausalLM(model_config).float()
        replica, stage_index, _ = config.coordinates(rank)
        all_inputs = torch.randint(0, 32, (8, 8), device=device)
        all_targets = all_inputs.clone()
        all_targets[0, 1:] = -100
        all_targets[2, 2:] = -100
        inputs = all_inputs.chunk(config.dp_size)[replica]
        targets = all_targets.chunk(config.dp_size)[replica]
        shape = (inputs.shape[0] // config.microbatches, 8)
        example = torch.zeros(
            shape if stage_index == 0 else (*shape, 32),
            device=device,
            dtype=torch.int64 if stage_index == 0 else torch.float32,
        )
        objective = CausalLMLoss(32)
        trainers = []
        for _ in range(2):
            stage = LlamaStage(
                copy.deepcopy(source),
                stage_index,
                config.pp_size,
                tp_size=config.tp_size,
            ).to(device)
            trainers.append(
                DistributedTrainer(
                    stage,
                    config,
                    example,
                    summed_loss=objective,
                    tensor_plan=stage.tensor_parallel_plan() or None,
                    precision=precision,
                    ddp_comm_hook=policy,
                )
            )
        actual, control = trainers
        optimizers = [
            torch.optim.AdamW(item.parameters(), lr=1e-3, foreach=False)
            for item in trainers
        ]
        source = source.to(device).eval()
        with torch.no_grad(), autocast_context(precision, device):
            full_output = source(all_inputs, use_cache=False).logits
            expected_loss = objective(
                full_output, all_targets
            ) / objective.normalization_count(all_targets)
        for update in range(2):
            old_gradients = [
                None if p.grad is None else local(p.grad).clone()
                for p in actual.parameters()
            ]
            rng = torch.cuda.get_rng_state(device).clone()
            cpu_rng = torch.get_rng_state().clone()
            results = [
                actual.evaluate(inputs, targets, return_outputs=update == 0)
                for _ in range(2)
            ]
            assert results[0].loss == results[1].loss
            assert (
                results[0].normalization_count
                == objective.normalization_count(all_targets).item()
            )
            torch.testing.assert_close(
                torch.cuda.get_rng_state(device), rng, rtol=0, atol=0
            )
            torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
            for parameter, gradient in zip(actual.parameters(), old_gradients):
                if gradient is None:
                    assert parameter.grad is None
                else:
                    torch.testing.assert_close(
                        local(parameter.grad), gradient, rtol=0, atol=0
                    )
            if update == 0:
                assert results[0].loss == pytest.approx(
                    expected_loss.item(), rel=0.005 if precision == "bf16" else 1e-5
                )
                if stage_index == config.pp_size - 1:
                    torch.testing.assert_close(
                        results[0].output,
                        full_output.chunk(config.dp_size)[replica],
                        atol=0.005 if precision == "bf16" else 3e-6,
                        rtol=0.005 if precision == "bf16" else 1e-5,
                    )
                else:
                    assert results[0].output is None
            if actual.schedule is not None:
                assert not actual.schedule._stage.fwd_cache
                assert not actual.schedule._stage.output_chunks
            losses = []
            for trainer, optimizer in zip(trainers, optimizers):
                optimizer.zero_grad(set_to_none=True)
                losses.append(trainer.forward_backward(inputs, targets).loss)
            assert losses[0] == losses[1]
            for left, right in zip(actual.parameters(), control.parameters()):
                torch.testing.assert_close(
                    local(left.grad), local(right.grad), rtol=0, atol=0
                )
            for optimizer in optimizers:
                optimizer.step()
            for left, right in zip(actual.parameters(), control.parameters()):
                torch.testing.assert_close(local(left), local(right), rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


CASES = [
    ((1, 1, 1), "gpipe", "none"),
    ((2, 1, 1), "gpipe", "none"),
    ((4, 1, 1), "gpipe", "none"),
    ((1, 2, 1), "gpipe", "none"),
    ((2, 2, 1), "gpipe", "none"),
    ((2, 1, 1), "gpipe", "bf16_compress"),
    ((4, 1, 1), "gpipe", "bf16_compress"),
] + [
    (axes, schedule, "none")
    for axes in ((1, 1, 2), (2, 1, 2), (1, 2, 2))
    for schedule in ("gpipe", "1f1b")
]


@pytest.mark.parametrize("axes,schedule,policy", CASES)
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cuda_evaluation_preserves_training(
    tmp_path, axes, schedule, policy, precision
):
    transformers = pytest.importorskip("transformers")
    world = axes[0] * axes[1] * axes[2]
    if (
        torch.__version__.split("+")[0] != "2.9.1"
        or transformers.__version__ != "4.51.3"
    ):
        pytest.skip("Requires pinned Torch2.9.1 / Transformers4.51.3")
    if not dist.is_nccl_available() or torch.cuda.device_count() < world:
        pytest.skip(f"Requires {world} CUDA devices and NCCL")
    torch.multiprocessing.spawn(
        worker,
        args=(axes, schedule, precision, policy, str(tmp_path)),
        nprocs=world,
        join=True,
    )
