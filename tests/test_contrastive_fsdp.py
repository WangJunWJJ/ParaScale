# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午11:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real FSDP owner-scope and product scalar-layout regression."""

import time
from copy import deepcopy
from datetime import timedelta
from functools import partial
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _worker(rank, rendezvous, family, data_dir):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    from parascale.workloads.clip import build_clip_contrastive_components
    from parascale.workloads.specs.clip import ClipContrastiveSpec
    from tests.test_hf_contrastive_objectives import _model

    torch.set_num_threads(1)
    # HF 4.51 constructs its causal mask without an explicit dtype. Keep the
    # default consistent with this FP64 numerical oracle to avoid overflow.
    torch.set_default_dtype(torch.float64)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(719)
        template = _model(family)
        config = deepcopy(template.config)
        for item in (config, config.text_config, config.vision_config):
            item._attn_implementation = "eager"
        reference = type(template)(config).double().eval()
        del template
        actual = deepcopy(reference)
        for name in ("logit_scale", "logit_bias"):
            parameter = getattr(reference, name, None)
            if parameter is not None:
                assert parameter.numel() == 1
                setattr(
                    reference,
                    name,
                    torch.nn.Parameter(parameter.detach().reshape(1).clone()),
                )
        original_names = set(dict(reference.named_parameters()))
        assert "logit_scale" in original_names
        if family == "siglip":
            assert "logit_bias" in original_names

        # Only asset loading is substituted: the product builder registers the
        # real owner hook before size-based FSDP wrapping changes ownership.
        with (
            patch.object(
                transformers.AutoModel, "from_pretrained", return_value=actual
            ),
            patch.object(
                transformers.AutoProcessor, "from_pretrained", return_value=object()
            ),
        ):
            wrapper, _, _, objective = build_clip_contrastive_components(
                ClipContrastiveSpec(
                    model_type=f"hf_{family}",
                    pretrained_model_name_or_path="unused-local",
                    data_type="webdataset",
                    data_dir=data_dir,
                    num_batches=1,
                )
            )
        model = FSDP(
            wrapper.to(device).eval(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1),
            use_orig_params=True,
            device_id=device,
        )
        assert isinstance(model.module.wrapped_model, FSDP), (
            "HF scalar owner was not wrapped"
        )
        assert model.module.wrapped_model.module is actual
        objective.bind_data_parallel(dist.group.WORLD)
        reference.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        oracle_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.01, foreach=False
        )

        sizes = (1, 2)
        total = sum(sizes)
        start = sum(sizes[:rank])
        batch = {
            "pixel_values": torch.randn(total, 3, 8, 8, dtype=torch.float64).to(device),
            "input_ids": torch.randint(3, 17, (total, 4)).to(device),
            "attention_mask": torch.ones(total, 4, dtype=torch.long, device=device),
        }
        local = {
            name: value[start : start + sizes[rank]] for name, value in batch.items()
        }
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            oracle_optimizer.zero_grad(set_to_none=True)
            output = model(**local)
            assert output["logit_scale"].numel() == 1
            if family == "siglip":
                assert output["logit_bias"].numel() == 1
            loss = objective(output, local) * (2 * sizes[rank] / total)
            loss.backward()
            expected = reference(**batch, return_loss=True).loss
            expected.backward()
            reported = loss.detach().clone()
            dist.all_reduce(reported)
            torch.testing.assert_close(reported / 2, expected, atol=1e-9, rtol=1e-7)
            with FSDP.summon_full_params(model, with_grads=True, writeback=False):
                actual_parameters = {
                    name.removeprefix("wrapped_model."): parameter
                    for name, parameter in model.named_parameters()
                }
                assert set(actual_parameters) == original_names
                for name, target in reference.named_parameters():
                    parameter = actual_parameters[name]
                    assert parameter.grad is not None, name
                    assert target.grad is not None, name
                    torch.testing.assert_close(
                        parameter.grad,
                        target.grad,
                        atol=1e-9,
                        rtol=1e-7,
                        msg=lambda message, name=name: f"{name}: {message}",
                    )
            optimizer.step()
            oracle_optimizer.step()
            with FSDP.summon_full_params(model, writeback=False):
                for name, parameter in model.named_parameters():
                    target = reference.get_parameter(
                        name.removeprefix("wrapped_model.")
                    )
                    torch.testing.assert_close(parameter, target, atol=1e-9, rtol=1e-7)
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available()
    or torch.cuda.device_count() < 2,
    reason="Requires two CUDA devices and NCCL",
)
@pytest.mark.parametrize("family", ["clip", "siglip"])
def test_hf_owner_features_survive_nested_full_shard(tmp_path, family):
    deadline = time.monotonic() + 75
    context = torch.multiprocessing.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(), family, str(tmp_path)),
        nprocs=2,
        join=False,
    )
    try:
        # Avoid ProcessContext.join's own lengthy failure-cleanup timeout.
        while any(process.is_alive() for process in context.processes):
            failed = [p for p in context.processes if p.exitcode not in (None, 0)]
            if failed:
                pytest.fail(
                    f"FSDP worker failed: {[(p.pid, p.exitcode) for p in failed]}"
                )
            if time.monotonic() >= deadline:
                pytest.fail("FSDP ownership regression exceeded 75 seconds")
            time.sleep(0.05)
        context.join()  # All workers exited; propagate any saved child traceback.
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
        for process in context.processes:
            if process.is_alive():
                process.kill()
        for process in context.processes:
            process.join(timeout=2)
