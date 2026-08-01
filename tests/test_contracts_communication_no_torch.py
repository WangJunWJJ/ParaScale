# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from parascale.communication import build_communication_plan, recommend_ddp_hook
from parascale.contracts import BatchContract, BatchMetadata, MetricContract


def test_batch_contract_extracts_common_metadata_without_torch():
    batch = {
        "pixel_values": object(),
        "num_images": 2,
        "num_pairs": 2,
        "tokens": 64,
        "patch_tokens": 392,
        "padding_ratio": 0.125,
        "pipeline_profile": {"pipeline_cache_hit": 1.0},
    }

    metadata = BatchMetadata.from_batch(batch)
    warnings = BatchContract().validate_lightweight(batch)

    assert metadata.num_images == 2
    assert metadata.patch_tokens == 392
    assert warnings == []


def test_metric_contract_filters_stable_metrics_without_torch():
    stable = MetricContract().filter_stable(
        {
            "end_to_end_images_per_second": 123.4,
            "pipeline_cache_hit": 1.0,
            "debug_text": "ignored",
        }
    )

    assert stable["end_to_end_images_per_second"] == 123.4
    assert stable["pipeline_cache_hit"] == 1.0
    assert "debug_text" not in stable


def test_communication_plan_prefers_bf16_hook_for_multimodal_without_torch():
    hook = recommend_ddp_hook(
        precision="bf16",
        task_type="multimodal",
        model_family="clip",
    )

    assert hook.hook == "bf16_compress"


def test_communication_plan_prefers_fp16_hook_for_clip_without_torch():
    hook = recommend_ddp_hook(
        precision="fp16",
        task_type="multimodal",
        model_family="clip",
    )

    assert hook.hook == "fp16_compress"


def test_communication_plan_carries_bucket_cap_without_torch():
    plan = build_communication_plan(
        backend="native_ddp",
        precision="bf16",
        task_type="multimodal",
        model_family="clip",
        bucket_cap_mb=100,
    )

    assert plan.ddp_hook == "bf16_compress"
    assert plan.bucket_cap_mb == 100
    assert plan.evidence["bucket_cap_mb"] == 100


def test_communication_plan_detects_adapter_only_sync_without_torch():
    plan = build_communication_plan(
        backend="native_ddp",
        precision="bf16",
        task_type="multimodal",
        model_family="vlm",
        gradient_accumulation_steps=4,
        trainable_ratio=0.01,
        dataloader_wait_ms=8.0,
    )

    assert plan.use_no_sync is True
    assert plan.adapter_only_sync is True
    assert plan.overlap_h2d is True
    assert plan.ddp_hook == "none"
