# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Offline pretrained input contracts, without downloading model weights."""

import io
import sys
import tarfile
from types import SimpleNamespace

import pytest

from parascale.workloads.clip import build_clip_contrastive_components
from parascale.workloads.datacomp import _datacomp_entry_to_sample
from parascale.workloads.specs.clip import ClipContrastiveSpec

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def hf(monkeypatch):
    calls = SimpleNamespace(
        model_paths=[], processor_paths=[], processed=[], forwarded=[]
    )
    encoded = {
        "input_ids": torch.tensor([[49406, 812, 49407]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "pixel_values": torch.tensor([[[[-1.25, 0.75]]]]),
    }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        @classmethod
        def from_pretrained(cls, path):
            calls.model_paths.append(path)
            return cls()

        def forward(self, **kwargs):
            calls.forwarded.append(kwargs)
            return SimpleNamespace(logits_per_image=self.weight.reshape(1, 1))

    class Processor:
        @classmethod
        def from_pretrained(cls, path):
            calls.processor_paths.append(path)
            return cls()

        def __call__(self, **kwargs):
            calls.processed.append(kwargs)
            if getattr(calls, "fail", False):
                raise ValueError("processor failed")
            return dict(encoded)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=Model,
            AutoProcessor=Processor,
        ),
    )
    calls.encoded = encoded
    return calls


def make_shard(tmp_path, payload=None):
    if payload is None:
        buffer = io.BytesIO()
        Image.new("RGB", (19, 11), (17, 91, 243)).save(buffer, format="PNG")
        payload = buffer.getvalue()
    with tarfile.open(tmp_path / "samples.tar", "w") as archive:
        for name, content in [
            ("sample.png", payload),
            ("sample.txt", b"Original caption!"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return payload


@pytest.mark.parametrize(
    "model_type", ["hf_clip", "hf_siglip", "openai_clip", "siglip"]
)
@pytest.mark.parametrize("image_mode", ["tensor", "bytes"])
@pytest.mark.parametrize("streaming", [False, True])
def test_checkpoint_processor_receives_original_images_and_captions(
    tmp_path,
    hf,
    model_type,
    image_mode,
    streaming,
):
    make_shard(tmp_path)
    spec = ClipContrastiveSpec(
        model_type=model_type,
        pretrained_model_name_or_path="local/checkpoint",
        data_type="datacomp_wds",
        data_dir=str(tmp_path),
        wds_image_mode=image_mode,
        streaming=streaming,
        num_samples=1,
        batch_size=1,
        num_batches=1,
        image_size=32,
    )
    model, _, batches, _ = build_clip_contrastive_components(spec)
    batch = next(iter(batches))
    model(**batch)
    assert hf.model_paths == hf.processor_paths == ["local/checkpoint"]
    assert len(hf.processed) == 1
    request = hf.processed[0]
    assert request["text"] == ["Original caption!"]
    assert request["images"][0].size == (19, 11)
    assert request["images"][0].getpixel((0, 0)) == (17, 91, 243)
    assert request["return_tensors"] == "pt"
    assert request["padding"] == "max_length"
    assert request["truncation"] is True
    assert "max_length" not in request  # Checkpoint tokenizer owns its context limit.
    for key, expected in hf.encoded.items():
        assert torch.equal(batch[key], expected)
        assert torch.equal(hf.forwarded[0][key], expected)
    assert batch["labels"].tolist() == [0]


@pytest.mark.parametrize("data_type", ["synthetic_image_text", "datacomp_metadata"])
def test_pretrained_rejects_sources_without_original_images(hf, data_type):
    spec = ClipContrastiveSpec(
        model_type="hf_clip",
        pretrained_model_name_or_path="local/checkpoint",
        data_type=data_type,
    )
    with pytest.raises(ValueError, match="original images"):
        build_clip_contrastive_components(spec)


@pytest.mark.parametrize("streaming", [False, True])
def test_pretrained_cache_and_batch_budget_are_preserved(tmp_path, hf, streaming):
    make_shard(tmp_path)
    cache = tmp_path / "cache"
    spec = ClipContrastiveSpec(
        model_type="hf_clip",
        pretrained_model_name_or_path="local/checkpoint",
        data_type="datacomp_wds",
        data_dir=str(tmp_path),
        dataset_local_cache_dir=str(cache),
        streaming=streaming,
        num_samples=10,
        batch_size=1,
        num_batches=1,
    )
    _, _, batches, _ = build_clip_contrastive_components(spec)
    assert len(list(batches)) == 1
    assert len(list(cache.glob("*.tar"))) == 1
    if streaming:
        assert all(path.parent == cache for path, _ in batches.samples)


@pytest.mark.parametrize("failure", ["decode", "processor"])
def test_pretrained_input_failure_is_not_replaced_with_fake_inputs(
    tmp_path, hf, failure
):
    make_shard(tmp_path, b"\x89PNG\r\n\x1a\ninvalid" if failure == "decode" else None)
    hf.fail = failure == "processor"
    spec = ClipContrastiveSpec(
        model_type="hf_clip",
        pretrained_model_name_or_path="local/checkpoint",
        data_type="datacomp_wds",
        data_dir=str(tmp_path),
        num_samples=1,
    )
    _, _, batches, _ = build_clip_contrastive_components(spec)
    with pytest.raises((OSError, ValueError)):
        next(batches)
    assert not hf.forwarded


def test_nonpretrained_tensor_preprocessing_is_unchanged(tmp_path):
    payload = make_shard(tmp_path)
    spec = ClipContrastiveSpec(image_size=8)
    sample = _datacomp_entry_to_sample(
        torch,
        "sample",
        {"image_bytes": payload, "text": "Original caption!"},
        spec,
    )
    assert sample["pixel_values"].shape == (3, 8, 8)
    assert torch.allclose(
        sample["pixel_values"][:, 0, 0], torch.tensor([17, 91, 243]) / 255
    )
    assert len(sample["input_ids"]) == 2


def test_synthetic_pipeline_still_runs():
    spec = ClipContrastiveSpec(num_samples=2, batch_size=2, num_batches=1)
    model, _, batches, loss_fn = build_clip_contrastive_components(spec)
    batch = next(batches)
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert torch.isfinite(loss_fn(model(**batch), batch))
