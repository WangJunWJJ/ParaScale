# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from parascale import (
    ContrastivePairSpec,
    DistributedTokenBudgetBatchSampler,
    MultiModalBatchSchema,
    MultiModalCollator,
    MultiModalDataPipeline,
    ParaScaleConfig,
    PatchTokenBatchSampler,
    ResolutionBucketSampler,
    VisionCollator,
    VisionMetadataCache,
    VlmLoraSpec,
    build_dataloader_plan,
    estimate_multimodal_token_cost,
    estimate_patch_tokens,
    normalize_multimodal_sample,
)
from parascale.data.vision import (
    VisionPreprocessor,
    VisionSample,
    VisionTransformConfig,
)


def test_readme_and_reports_are_utf8_encoded():
    from pathlib import Path

    paths = [Path("README.md"), *Path("doc").glob("*.md")]

    assert paths
    for path in paths:
        path.read_text(encoding="utf-8")


def test_multimodal_schema_normalizes_common_aliases():
    schema = MultiModalBatchSchema()
    sample = normalize_multimodal_sample(
        {
            "input_ids": [1, 2, 3],
            "images": "image-tensor",
            "video": "video-tensor",
            "input_features": "audio-tensor",
        },
        schema,
    )

    assert sample[schema.pixel_values] == "image-tensor"
    assert sample[schema.video_values] == "video-tensor"
    assert sample[schema.audio_features] == "audio-tensor"
    assert sample[schema.modality_mask]["image"] is True


def test_multimodal_collator_pads_text_and_keeps_schema_keys():
    collator = MultiModalCollator(max_length=4)

    batch = collator(
        [
            {"input_ids": [1, 2], "labels": [1, 2], "images": "a"},
            {"input_ids": [3, 4, 5, 6, 7], "labels": [3, 4, 5, 6, 7], "images": "b"},
        ]
    )

    assert batch["input_ids"] == [[1, 2, 0, 0], [3, 4, 5, 6]]
    assert batch["labels"] == [[1, 2, -100, -100], [3, 4, 5, 6]]
    assert batch["pixel_values"] == ["a", "b"]


def test_distributed_token_budget_sampler_shards_batches_by_rank():
    dataset = [{"input_ids": list(range(length))} for length in [3, 4, 5, 6, 7, 8]]
    rank0 = list(
        DistributedTokenBudgetBatchSampler(
            dataset,
            max_tokens=8,
            rank=0,
            world_size=2,
            shuffle=False,
        )
    )
    rank1 = list(
        DistributedTokenBudgetBatchSampler(
            dataset,
            max_tokens=8,
            rank=1,
            world_size=2,
            shuffle=False,
        )
    )

    assert rank0
    assert rank1
    assert rank0 != rank1


def test_build_dataloader_plan_uses_deepspeed_like_defaults():
    config = ParaScaleConfig(
        batch_size=32,
        batching_strategy="token_budget",
        max_tokens_per_batch=4096,
        dataloader_num_workers=6,
    )

    plan = build_dataloader_plan(config, world_size=8)

    assert plan.batch_sampler == "token_budget"
    assert plan.max_tokens_per_batch == 4096
    assert plan.num_workers == 6
    assert plan.pin_memory is True


def test_vision_and_multimodal_package_entries_are_available():
    dataset = [
        {"height": 224, "width": 224, "image": "a"},
        {"height": 384, "width": 384, "image": "b"},
    ]
    resolution_batches = list(
        ResolutionBucketSampler(dataset, buckets=[(224, 224), (384, 384)], batch_size=1)
    )
    patch_batches = list(
        PatchTokenBatchSampler(dataset, max_patch_tokens=1024, patch_size=16)
    )
    collated = VisionCollator()(dataset)
    pipeline = MultiModalDataPipeline(tokenizer=lambda text: [1, 2, 3])
    processed = pipeline.process({"text": "hello", "images": "image"})

    assert estimate_patch_tokens(224, 224, patch_size=16) == 196
    assert resolution_batches
    assert patch_batches
    assert collated["image"] == ["a", "b"]
    assert processed["input_ids"] == [1, 2, 3]
    assert processed["pixel_values"] == "image"


def test_patch_token_sampler_respects_optional_sample_limit():
    dataset = [{"height": 32, "width": 32} for _ in range(5)]
    batches = list(
        PatchTokenBatchSampler(
            dataset, max_patch_tokens=64, patch_size=16, max_samples=2
        )
    )

    assert batches == [[0, 1], [2, 3], [4]]


def test_vision_metadata_cache_and_normalized_collator_fields():
    cache = VisionMetadataCache()
    cache.put("img-1", {"height": 512, "width": 768})
    collated = VisionCollator()([{"image": "pixels", **cache.get("img-1")}])

    assert collated["pixel_values"] == ["pixels"]
    assert collated["height"] == [512]
    assert collated["width"] == [768]


def test_multimodal_pipeline_cache_and_profile():
    calls = {"tokenizer": 0}

    def tokenizer(text):
        calls["tokenizer"] += 1
        return [ord(ch) for ch in text]

    pipeline = MultiModalDataPipeline(
        tokenizer=tokenizer, image_processor=lambda image: f"processed:{image}"
    )
    first = pipeline.process_cached("sample-1", {"text": "hi", "images": "img"})
    second = pipeline.process_cached(
        "sample-1", {"text": "ignored", "images": "ignored"}
    )
    profile = pipeline.profile_sample(first)

    assert calls["tokenizer"] == 1
    assert second["input_ids"] == first["input_ids"]
    assert first["pixel_values"] == "processed:img"
    assert profile["tokens"] == 2
    assert profile["total_tokens"] >= 2
    assert profile["has_image"] is True


def test_vlm_lora_and_clip_specs_are_first_multimodal_targets():
    vlm = VlmLoraSpec(lora_rank=8, target_modules=("q_proj", "k_proj", "v_proj"))
    clip = ContrastivePairSpec(temperature=0.05)

    assert vlm.to_dict()["adapter_policy"] == "lora"
    assert vlm.to_dict()["lora_rank"] == 8
    assert clip.to_dict()["objective"] == "image_text_contrastive"
    assert clip.to_dict()["symmetric_loss"] is True


def test_multimodal_token_cost_estimator_counts_text_and_image_tokens():
    sample = {
        "input_ids": [1, 2, 3, 4],
        "pixel_values": type("FakeImage", (), {"shape": (3, 224, 224)})(),
    }

    estimate = estimate_multimodal_token_cost(sample, image_patch_size=16)

    assert estimate.text_tokens == 4
    assert estimate.image_tokens == 196
    assert estimate.total_tokens == 200


def test_vision_preprocessor_cache_key_distinguishes_bytes_images():
    preprocessor = VisionPreprocessor(
        transform=VisionTransformConfig(image_size=32),
        tensor_cache=False,
    )

    first = preprocessor._cache_key(VisionSample(image=b"image-a"))
    same = preprocessor._cache_key(VisionSample(image=b"image-a"))
    second = preprocessor._cache_key(VisionSample(image=b"image-b"))

    assert first == same
    assert first != second


def test_vision_preprocessor_wraps_bytes_for_pil_open():
    captured = {}
    sentinel = object()

    class FakePilImage:
        def open(self, image):
            captured["image"] = image
            return sentinel

    opened = VisionPreprocessor._open_image(FakePilImage(), b"raw-image-bytes")

    assert opened is sentinel
    assert captured["image"].getvalue() == b"raw-image-bytes"


def test_python_files_have_utf8_creation_time_authorship_headers():
    import re
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    header_time = re.compile(
        r"^# @Time : \d{4}/\d{1,2}/\d{1,2} (?:\u4e0a\u5348|\u4e0b\u5348)\d{1,2}:\d{2}$"
    )
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        text=True,
        encoding="utf-8",
        cwd=repo_root,
    ).splitlines()
    failures = []
    for relative_path in paths:
        file_path = repo_root / relative_path
        if not file_path.is_file():
            continue
        lines = file_path.read_text(encoding="utf-8").splitlines()
        start = 1 if lines and lines[0].startswith("#!") else 0
        header = lines[start : start + 4]
        invalid = len(header) != 4 or header[0] != "# -*- coding: utf-8 -*-"
        if not invalid:
            invalid = not header_time.fullmatch(header[1])
        if not invalid:
            invalid = header[2] != "# @Author : Wang Jun"
        if not invalid:
            invalid = header[3] != "# @Email: wj_xd@foxmail.com"
        if invalid:
            failures.append(relative_path)
    assert not failures, f"invalid Python headers: {sorted(set(failures))}"
