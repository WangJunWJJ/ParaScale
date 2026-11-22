# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午7:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Workload objectives must agree with real Hugging Face model formulas."""

from copy import deepcopy

import pytest

from parascale.workloads.clip import build_clip_contrastive_components
from parascale.workloads.specs.clip import ClipContrastiveSpec

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def fp64_default():
    previous = torch.get_default_dtype()
    # Transformers 4.51 creates causal masks without specifying their dtype.
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _model(family):
    text = dict(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = dict(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    if family == "clip":
        config = transformers.CLIPConfig(
            text_config=text, vision_config=vision, projection_dim=4
        )
        return transformers.CLIPModel(config).double().eval()
    config = transformers.SiglipConfig(text_config=text, vision_config=vision)
    return transformers.SiglipModel(config).double().eval()


@pytest.mark.parametrize("family", ["clip", "siglip"])
@pytest.mark.parametrize("size", [1, 3])
def test_hf_workload_preserves_native_loss_and_gradients(
    monkeypatch, tmp_path, family, size, fp64_default
):
    torch.manual_seed(19)
    reference = _model(family)
    actual = deepcopy(reference)
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: actual
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: object()
    )
    spec = ClipContrastiveSpec(
        model_type=f"hf_{family}",
        pretrained_model_name_or_path="unused-local",
        data_type="webdataset",
        data_dir=str(tmp_path),
        num_batches=1,
    )
    model, _, _, loss_fn = build_clip_contrastive_components(spec)
    assert all(parameter.ndim > 0 for parameter in model.parameters())
    batch = {
        "pixel_values": torch.randn(size, 3, 8, 8, dtype=torch.float64),
        "input_ids": torch.randint(3, 17, (size, 4)),
        "attention_mask": torch.ones(size, 4, dtype=torch.long),
    }
    expected = reference(**batch, return_loss=True).loss
    output = model(**batch)
    loss = loss_fn(output, {**batch, "labels": torch.arange(size)})
    torch.testing.assert_close(loss, expected, atol=1e-9, rtol=1e-7)
    loss.backward()
    expected.backward()
    for (name, parameter), (_, target) in zip(
        actual.named_parameters(), reference.named_parameters()
    ):
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad.reshape_as(target.grad), target.grad, atol=1e-9, rtol=1e-7
        )


def test_hf_family_mismatch_rejects_before_processor_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: _model("clip")
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *a, **k: pytest.fail("processor loaded"),
    )
    with pytest.raises(ValueError, match="family"):
        build_clip_contrastive_components(
            ClipContrastiveSpec(
                model_type="hf_siglip",
                pretrained_model_name_or_path="unused-local",
                data_type="webdataset",
                data_dir=str(tmp_path),
            )
        )


def test_synthetic_temperature_is_not_clamped_at_initialization():
    spec = ClipContrastiveSpec(
        num_samples=2, batch_size=2, num_batches=1, temperature=0.07
    )
    model, _, batches, _ = build_clip_contrastive_components(spec)
    assert model.logit_scale.shape == (1,)
    output = model(**next(iter(batches)))
    torch.testing.assert_close(
        output["logit_scale"], torch.tensor(1 / spec.temperature)
    )
    output["logit_scale"].backward()
    assert model.logit_scale.grad.abs() > 0


def test_hf_scalars_materialize_before_owner_forward_returns(monkeypatch, tmp_path):
    actual = _model("siglip")
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: actual
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: object()
    )
    model, _, _, loss_fn = build_clip_contrastive_components(
        ClipContrastiveSpec(
            model_type="hf_siglip",
            pretrained_model_name_or_path="unused-local",
            data_type="webdataset",
            data_dir=str(tmp_path),
        )
    )

    class ReshardAfterReturn(torch.nn.Module):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def forward(self, **kwargs):
            return self.owner(**kwargs)

    # An FSDP wrapper exposes neither scalar as its own parameter. Access must
    # occur inside the owner, not by unwrapping it after its forward returns.
    model.wrapped_model = ReshardAfterReturn(actual)
    batch = {
        "pixel_values": torch.randn(1, 3, 8, 8, dtype=torch.float64),
        "input_ids": torch.ones(1, 4, dtype=torch.long),
    }
    output = model(**batch)
    assert output["logit_scale"].numel() == output["logit_bias"].numel() == 1
    loss_fn(output, {}).backward()
    assert actual.logit_scale.grad is not None
    assert actual.logit_bias.grad is not None
