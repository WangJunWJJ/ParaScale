# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午7:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Contrastive objectives must preserve global-batch loss and gradients."""

from datetime import timedelta

import pytest

from parascale.workloads.contrastive_loss import ContrastiveLoss

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
F = torch.nn.functional


def _reference(features, family):
    logits = features["logit_scale"] * (
        features["image_features"] @ features["text_features"].T
    )
    labels = torch.arange(logits.shape[0], device=logits.device)
    if family == "clip":
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    logits = logits + features["logit_bias"]
    signs = 2 * torch.eye(len(labels), device=logits.device) - 1
    return -F.logsigmoid(signs * logits).sum() / len(labels)


class _PairModel(torch.nn.Module):
    def __init__(self, family):
        super().__init__()
        self.image = torch.nn.Linear(3, 4)
        self.text = torch.nn.Linear(3, 4)
        self.scale = torch.nn.Parameter(torch.tensor(0.3))
        if family == "siglip":
            self.bias = torch.nn.Parameter(torch.tensor(-0.8))

    def forward(self, images, texts, **kwargs):
        result = {
            "image_features": F.normalize(self.image(images), dim=-1),
            "text_features": F.normalize(self.text(texts), dim=-1),
            "logit_scale": self.scale.exp(),
        }
        if hasattr(self, "bias"):
            result["logit_bias"] = self.bias.clone()
        return result


@pytest.mark.parametrize("family", ["clip", "siglip"])
@pytest.mark.parametrize("size", [1, 4])
def test_single_rank_matches_native_formula_and_gradients(family, size):
    torch.manual_seed(17)
    model = _PairModel(family).double()
    images, texts = torch.randn(2, size, 3, dtype=torch.float64)
    output = model(images, texts)
    loss = ContrastiveLoss(family)(output, {})
    expected = _reference(output, family)
    torch.testing.assert_close(loss, expected, atol=1e-9, rtol=1e-7)
    actual_grads = torch.autograd.grad(
        loss, tuple(model.parameters()), retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, atol=1e-9, rtol=1e-7)
    if family == "siglip":
        assert loss > 0  # A positive-only batch is not zero-loss softmax.


@pytest.mark.parametrize("shape", [(0, 4), (2, 5), (2, 4, 1)])
def test_invalid_feature_pairs_reject(shape):
    output = {
        "image_features": torch.ones(2, 4),
        "text_features": torch.ones(shape),
        "logit_scale": torch.tensor(1.0),
    }
    with pytest.raises(ValueError, match="feature"):
        ContrastiveLoss("clip")(output, {})


def test_objective_family_is_explicit():
    with pytest.raises(ValueError, match="family"):
        ContrastiveLoss("unknown")


def test_fit_binds_local_loss_and_does_not_infer_world_group():
    from parascale.config import ParaScaleConfig
    from parascale.runtime.training import TrainEngine

    class TrackedLoss(ContrastiveLoss):
        def bind_data_parallel(self, group):
            calls.append(group)
            super().bind_data_parallel(group)

    calls = []
    model = _PairModel("clip")
    objective = TrackedLoss("clip")
    engine = TrainEngine(ParaScaleConfig(training_backend="native"))
    engine.fit(
        [{"images": torch.randn(2, 3), "texts": torch.randn(2, 3), "num_pairs": 2}],
        max_steps=1,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=objective,
    )
    assert calls == [None]


@pytest.mark.parametrize("backend", ["native_ddp", "fsdp", "deepspeed"])
def test_runtime_binds_only_an_initialized_dp_group(monkeypatch, backend):
    from types import SimpleNamespace

    from parascale.config import ParaScaleConfig
    from parascale.runtime.training import TrainEngine

    engine = TrainEngine(ParaScaleConfig(training_backend=backend))
    engine.training_backend = SimpleNamespace(name=backend)
    calls = []
    objective = SimpleNamespace(bind_data_parallel=calls.append)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="process group"):
        engine._bind_loss_data_parallel(objective)
    group = object()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=group))
    engine._bind_loss_data_parallel(objective)
    assert calls == [group]


def _distributed_worker(rank, rendezvous, result_dir, family, sizes):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=len(sizes),
        timeout=timedelta(seconds=35),
    )
    try:
        torch.manual_seed(17)
        model = _PairModel(family).double()
        reference = _PairModel(family).double()
        reference.load_state_dict(model.state_dict())
        images, texts = torch.randn(2, sum(sizes), 3, dtype=torch.float64)
        start, end = sum(sizes[:rank]), sum(sizes[: rank + 1])
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        loss_fn = ContrastiveLoss(family)
        loss_fn.bind_data_parallel(dist.group.WORLD)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.07, momentum=0.9)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.07, momentum=0.9
        )
        for _ in range(2):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            output = ddp(images[start:end], texts[start:end])
            if min(sizes) == 0:
                with pytest.raises(ValueError, match="feature"):
                    loss_fn(output, {})
                break
            loss = loss_fn(output, {})
            # Mirrors TrainEngine's global sample-weight normalization.
            scaled = loss * (len(sizes) * sizes[rank] / sum(sizes))
            scaled.backward()
            expected = _reference(reference(images, texts), family)
            expected.backward()
            mean_loss = scaled.detach().clone()
            dist.all_reduce(mean_loss)
            torch.testing.assert_close(
                mean_loss / len(sizes), expected, atol=1e-9, rtol=1e-7
            )
            for actual, target in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(
                    actual.grad, target.grad, atol=1e-9, rtol=1e-7
                )
            optimizer.step()
            reference_optimizer.step()
            for actual, target in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, target, atol=1e-9, rtol=1e-7)
        torch.save({"passed": True}, f"{result_dir}/{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("family", ["clip", "siglip"])
@pytest.mark.parametrize("sizes", [(1, 1), (1, 3), (0, 2)])
def test_distributed_matches_global_batch_or_rejects_empty(tmp_path, family, sizes):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    context = torch.multiprocessing.spawn(
        _distributed_worker,
        args=(rendezvous, str(tmp_path), family, sizes),
        nprocs=len(sizes),
        join=False,
    )
    try:
        import time

        deadline = time.monotonic() + 50
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("contrastive collective did not finish within 50 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(len(sizes)):
        assert torch.load(tmp_path / f"{rank}.pt", weights_only=True)["passed"]
