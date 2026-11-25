# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午7:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Native scalar promotion must survive mixed-precision feature extraction."""

from copy import deepcopy

import pytest

from parascale.workloads.clip import build_clip_contrastive_components
from parascale.workloads.specs.clip import ClipContrastiveSpec

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tests.test_hf_contrastive_objectives import _model  # noqa: E402


@pytest.mark.parametrize("family", ["clip", "siglip"])
def test_bf16_autocast_matches_native_loss_and_parameter_gradients(
    monkeypatch, tmp_path, family
):
    torch.manual_seed(19)
    reference = _model(family).float()
    actual = deepcopy(reference)
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: actual
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: object()
    )
    model, _, _, objective = build_clip_contrastive_components(
        ClipContrastiveSpec(
            model_type=f"hf_{family}",
            pretrained_model_name_or_path="unused-local",
            data_type="webdataset",
            data_dir=str(tmp_path),
        )
    )
    batch = {
        "pixel_values": torch.randn(3, 3, 8, 8),
        "input_ids": torch.randint(3, 17, (3, 4)),
        "attention_mask": torch.ones(3, 4, dtype=torch.long),
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference(**batch, return_loss=True).loss
        output = model(**batch)
        if family == "siglip":
            logits = output["logit_scale"] * (
                output["image_features"] @ output["text_features"].T
            )
            assert logits.dtype == torch.float32
        loss = objective(output, batch)
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)
    loss.backward()
    expected.backward()
    for (name, value), (_, target) in zip(
        actual.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(
            value.grad.reshape_as(target.grad),
            target.grad,
            atol=1e-5,
            rtol=1e-4,
            msg=name,
        )
