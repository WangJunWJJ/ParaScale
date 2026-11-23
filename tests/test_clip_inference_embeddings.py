# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real CLIP inference must expose usable, bounded, native-equivalent features."""

import json

import pytest

from parascale.runtime.inference.runner import InferenceRunner
from parascale.workloads.inference import HFClipInferenceModel

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_real_clip_returns_native_features_for_non_square_batches():
    torch.manual_seed(43)
    model = transformers.CLIPModel(
        transformers.CLIPConfig(
            projection_dim=4,
            text_config={
                "vocab_size": 17,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 6,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 4,
            },
        )
    ).eval()
    inputs = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "input_ids": torch.tensor([[1, 3, 4, 2], [1, 5, 6, 2], [1, 7, 8, 2]]),
        "attention_mask": torch.ones(3, 4, dtype=torch.long),
    }
    calls = []

    def processor(**kwargs):
        calls.append(kwargs)
        return inputs

    adapter = HFClipInferenceModel("unused-local")
    adapter._model, adapter._processor = model, processor
    with torch.inference_mode():
        expected = model(**inputs)
    captured = []
    result = InferenceRunner(adapter, "multimodal_embedding").run(
        [{"images": [None, None], "texts": ["a", "b", "c"]}],
        warmup_steps=0,
        max_outputs=0,
        on_output=captured.append,
    )
    assert result["outputs"] == []
    assert len(captured) == 1
    output = captured[0]
    for name, target in (
        ("image_embeddings", expected.image_embeds),
        ("text_embeddings", expected.text_embeds),
    ):
        actual = torch.tensor(output[name])
        torch.testing.assert_close(actual, target, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(len(actual)))
    assert output["logits_per_image_shape"] == [2, 3]
    json.dumps(output, allow_nan=False)
    assert calls[0]["truncation"] is True
    assert calls[0]["max_length"] == model.config.text_config.max_position_embeddings
    assert all(parameter.grad is None for parameter in model.parameters())
