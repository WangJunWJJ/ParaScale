# -*- coding: utf-8 -*-
# @Time : 2026/6/11 上午9:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from parascale.cli import run_benchmark_from_config, run_train_from_config
from parascale.data import estimate_sample_tokens
from parascale.workloads.clip import (
    ClipContrastiveSpec,
    build_clip_contrastive_components,
)
from parascale.workloads.datacomp import (
    _DataCompWdsIterableDataset,
    _iter_datacomp_tar_entries,
    _looks_like_supported_image,
)


def test_datacomp_streaming_repeats_assigned_shard_to_requested_sample_count(
    tmp_path,
):
    import io
    import tarfile

    wds_dir = tmp_path / "repeating-wds"
    wds_dir.mkdir()
    tar_path = wds_dir / "datacomp-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        image = Image.new("RGB", (16, 16), color=(20, 40, 80))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        for name, payload in (
            ("000000000.jpg", image_bytes),
            ("000000000.txt", b"one reusable sample"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    spec = ClipContrastiveSpec(
        data_type="datacomp_wds",
        data_dir=str(wds_dir),
        streaming=True,
        image_size=16,
        patch_size=8,
        num_samples=3,
    )

    samples = list(_DataCompWdsIterableDataset(torch, spec))

    assert len(samples) == 3
    assert {sample["metadata"]["sample_id"] for sample in samples} == {
        "000000000"
    }


def _clip_config(tmp_path):
    return {
        "parascale": {
            "task_type": "multimodal",
            "model_family": "clip",
            "training_backend": "native",
            "batch_size": 4,
            "max_patch_tokens_per_batch": 1024,
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 1,
        },
        "task": {
            "type": "multimodal",
            "workload": "clip_contrastive",
            "modalities": ["text", "image"],
            "objective": "image_text_contrastive",
            "temperature": 0.07,
        },
        "model": {
            "type": "tiny_clip",
            "image_size": 32,
            "patch_size": 16,
            "vocab_size": 128,
            "text_length": 12,
            "embed_dim": 16,
        },
        "data": {
            "type": "synthetic_image_text",
            "num_samples": 12,
            "batch_size": 4,
            "image_size": 32,
            "text_length": 12,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 0.01,
        },
        "model_profile": {
            "total_params": 100_000,
            "total_memory": 400_000,
            "num_layers": 4,
            "model_type": "clip",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpus_per_node": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
        },
        "training": {
            "workload": "clip_contrastive",
            "max_steps": 2,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 1,
        },
    }


def test_clip_contrastive_factory_runs_one_forward_step():
    spec = ClipContrastiveSpec(
        image_size=32,
        patch_size=16,
        embed_dim=16,
        batch_size=4,
        num_batches=1,
        num_samples=4,
        device="cpu",
    )
    model, optimizer, dataloader, loss_fn = build_clip_contrastive_components(spec)
    batch = next(iter(dataloader))
    logits = model(**batch)
    loss = loss_fn(logits, batch)

    assert logits.shape == (4, 4)
    assert batch["num_pairs"] == 4
    assert batch["tokens"] > 0
    assert loss.item() > 0
    assert optimizer is not None


def test_clip_contrastive_dataloader_emits_cpu_batches_for_backend_placement():
    spec = ClipContrastiveSpec(
        image_size=32,
        patch_size=16,
        embed_dim=16,
        batch_size=2,
        num_batches=1,
        num_samples=2,
        device="cpu",
    )
    _model, _optimizer, dataloader, _loss_fn = build_clip_contrastive_components(spec)

    batch = next(iter(dataloader))

    assert batch["pixel_values"].device.type == "cpu"
    assert batch["input_ids"].device.type == "cpu"
    assert batch["attention_mask"].device.type == "cpu"
    assert batch["labels"].device.type == "cpu"


def test_token_estimator_handles_torch_pixel_tensors():
    estimate = estimate_sample_tokens(
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "pixel_values": torch.zeros(3, 32, 32),
        },
        image_patch_size=16,
    )

    assert estimate.text_tokens == 3
    assert estimate.image_tokens == 4


def test_clip_contrastive_factory_reads_datacomp_wds(tmp_path):
    import io
    import json
    import tarfile

    wds_dir = tmp_path / "wds"
    wds_dir.mkdir()
    tar_path = wds_dir / "datacomp-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        for index in range(4):
            stem = f"{index:09d}"
            image = Image.new("RGB", (32, 32), color=(index * 30, 20, 120))
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="JPEG")
            image_bytes = image_buffer.getvalue()
            image_info = tarfile.TarInfo(f"{stem}.jpg")
            image_info.size = len(image_bytes)
            archive.addfile(image_info, io.BytesIO(image_bytes))

            text_bytes = f"sample caption {index}".encode("utf-8")
            text_info = tarfile.TarInfo(f"{stem}.txt")
            text_info.size = len(text_bytes)
            archive.addfile(text_info, io.BytesIO(text_bytes))

            meta_bytes = json.dumps({"uid": stem}).encode("utf-8")
            meta_info = tarfile.TarInfo(f"{stem}.json")
            meta_info.size = len(meta_bytes)
            archive.addfile(meta_info, io.BytesIO(meta_bytes))

    spec = ClipContrastiveSpec(
        data_type="datacomp_wds",
        data_dir=str(wds_dir),
        dataset_local_cache_dir=str(tmp_path / "local-shard-cache"),
        image_size=32,
        patch_size=16,
        embed_dim=16,
        batch_size=4,
        num_batches=1,
        num_samples=4,
        device="cpu",
    )
    model, _optimizer, dataloader, loss_fn = build_clip_contrastive_components(spec)
    batch = next(iter(dataloader))
    logits = model(**batch)
    loss = loss_fn(logits, batch)

    assert batch["pixel_values"].shape == (4, 3, 32, 32)
    assert batch["num_pairs"] == 4
    assert batch["metadata"][0]["data_source"] == "datacomp_wds"
    assert list((tmp_path / "local-shard-cache").glob("*.tar"))
    assert batch["pipeline_profile"]["shard_read_ms"] >= 0.0
    assert batch["pipeline_profile"]["sample_decode_ms"] > 0.0
    assert batch["pipeline_profile"]["collate_ms"] > 0.0
    assert loss.item() > 0


def test_clip_contrastive_streaming_wds_medium_model_runs(tmp_path):
    import io
    import json
    import tarfile

    wds_dir = tmp_path / "streaming-wds"
    wds_dir.mkdir()
    tar_path = wds_dir / "datacomp-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        for index in range(6):
            stem = f"{index:09d}"
            image = Image.new("RGB", (40, 40), color=(index * 20, 40, 90))
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="JPEG")
            image_bytes = image_buffer.getvalue()
            image_info = tarfile.TarInfo(f"{stem}.jpg")
            image_info.size = len(image_bytes)
            archive.addfile(image_info, io.BytesIO(image_bytes))

            text_bytes = f"streaming medium caption {index}".encode("utf-8")
            text_info = tarfile.TarInfo(f"{stem}.txt")
            text_info.size = len(text_bytes)
            archive.addfile(text_info, io.BytesIO(text_bytes))

            meta_bytes = json.dumps({"uid": stem}).encode("utf-8")
            meta_info = tarfile.TarInfo(f"{stem}.json")
            meta_info.size = len(meta_bytes)
            archive.addfile(meta_info, io.BytesIO(meta_bytes))

    spec = ClipContrastiveSpec(
        model_type="clip_medium",
        data_type="datacomp_wds",
        data_dir=str(wds_dir),
        streaming=True,
        num_workers=0,
        image_size=32,
        patch_size=16,
        vocab_size=256,
        text_length=8,
        embed_dim=32,
        vision_layers=1,
        text_layers=1,
        num_heads=4,
        batch_size=2,
        num_batches=2,
        num_samples=4,
        device="cpu",
    )
    model, _optimizer, dataloader, loss_fn = build_clip_contrastive_components(spec)
    batches = list(dataloader)

    assert len(batches) == 2
    logits = model(**batches[0])
    loss = loss_fn(logits, batches[0])
    assert logits.shape == (2, 2)
    assert batches[0]["num_pairs"] == 2
    assert loss.item() > 0


def test_datacomp_wds_streaming_rejects_remote_lazy_cold_corruption(tmp_path):
    import io
    import tarfile

    wds_dir = tmp_path / "broken-wds"
    wds_dir.mkdir()
    tar_path = wds_dir / "datacomp-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in [
            ("000000001.jpg", b"not an image"),
            ("000000001.txt", b"bad image sample"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    stem, entry = next(_iter_datacomp_tar_entries(tar_path))

    assert stem == "000000001"
    assert _looks_like_supported_image(entry["image_bytes"]) is False


def test_datacomp_wds_streaming_rejects_non_contiguous_duplicate_stems(tmp_path):
    import io
    import tarfile

    tar_path = tmp_path / "datacomp-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in [
            ("000000001.jpg", b"\xff\xd8\xfffake"),
            ("000000002.jpg", b"\xff\xd8\xfffake"),
            ("000000001.txt", b"late text"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    iterator = _iter_datacomp_tar_entries(tar_path)
    next(iterator)
    next(iterator)
    with pytest.raises(RuntimeError, match="not contiguous"):
        next(iterator)


def test_datacomp_wds_streaming_skips_invalid_image_bytes(tmp_path):
    import io
    import tarfile

    wds_dir = tmp_path / "skip-wds"
    wds_dir.mkdir()
    tar_path = wds_dir / "datacomp-000000.tar"
    with tarfile.open(tar_path, "w") as archive:
        for stem, image_bytes, text in [
            ("000000001", b"not an image", "bad"),
            ("000000002", b"\xff\xd8\xffgood", "good"),
        ]:
            image_info = tarfile.TarInfo(f"{stem}.jpg")
            image_info.size = len(image_bytes)
            archive.addfile(image_info, io.BytesIO(image_bytes))

            text_bytes = text.encode("utf-8")
            text_info = tarfile.TarInfo(f"{stem}.txt")
            text_info.size = len(text_bytes)
            archive.addfile(text_info, io.BytesIO(text_bytes))

    spec = ClipContrastiveSpec(
        data_type="datacomp_wds",
        data_dir=str(wds_dir),
        image_size=32,
        patch_size=16,
        embed_dim=16,
        batch_size=1,
        num_batches=1,
        num_samples=1,
        device="cpu",
    )
    model, _optimizer, dataloader, _loss_fn = build_clip_contrastive_components(spec)
    batch = next(iter(dataloader))

    assert batch["metadata"][0]["text"] == "good"
    assert batch["num_pairs"] == 1


def test_cli_clip_contrastive_train_outputs_multimodal_metrics(tmp_path):
    payload = run_train_from_config(_clip_config(tmp_path))
    metrics = payload["last_metrics"]

    assert payload["clip_contrastive"] is True
    assert payload["capability_level"] == "local_native_clip_contrastive_synthetic"
    assert payload["global_step"] == 2
    assert metrics["image_text_pairs"] == 4
    assert metrics["image_text_pairs_per_second"] > 0
    assert metrics["tokens_per_second"] > 0
    assert metrics["padding_ratio"] >= 0


def test_cli_clip_contrastive_benchmark_writes_comparison_metrics(tmp_path):
    config = _clip_config(tmp_path)
    config["training"]["benchmark_steps"] = 2
    payload = run_benchmark_from_config(config)

    assert payload["mode"] == "benchmark"
    assert payload["benchmark_type"] == "clip_contrastive_train"
    assert payload["metrics"]["image_text_pairs_per_second"] > 0
    assert payload["comparison_contract"]["workload"] == "clip_contrastive"
