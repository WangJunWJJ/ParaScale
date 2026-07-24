# -*- coding: utf-8 -*-
# @Time : 2026/7/24 下午4:20
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Runtime evidence tests that avoid importing torch."""

from __future__ import annotations


def test_runtime_evidence_summarizes_matrix_recommendations_and_tuner_decisions():
    from parascale.runtime.evidence import build_runtime_evidence

    evidence = build_runtime_evidence(
        {
            "mode": "benchmark_matrix",
            "dry_run": False,
            "runtime_status": "real_matrix",
            "capability_level": "backend_matrix",
            "report": {
                "recommendations": [
                    {"selected_backend": "native_ddp"},
                    {"selected_backend": "fsdp"},
                ],
                "tuner_explanations": [
                    {
                        "runtime_tuning": {
                            "decisions": [
                                {"action": "prefetch_to_device"},
                                {"action": "cache_processed_images"},
                            ]
                        }
                    }
                ],
            },
        }
    )

    assert evidence["benchmark_matrix"]["recommendation_count"] == 2
    assert evidence["benchmark_matrix"]["selected_backends"] == [
        "native_ddp",
        "fsdp",
    ]
    assert evidence["tuner"]["available"] is True
    assert evidence["tuner"]["explanation_count"] == 1
    assert evidence["tuner"]["decision_count"] == 2


def test_runtime_evidence_summarizes_device_backend_capabilities():
    from parascale.runtime.evidence import build_runtime_evidence

    evidence = build_runtime_evidence(
        {
            "mode": "doctor",
            "runtime_status": "diagnostic",
            "device_backends": [
                {
                    "name": "cpu",
                    "accelerator": "cpu",
                    "available": True,
                    "device_count": 1,
                    "memory": {"peak_memory_allocated_bytes": 0},
                },
                {
                    "name": "nvidia",
                    "accelerator": "cuda",
                    "available": False,
                    "device_count": 0,
                    "memory": {"peak_memory_allocated_bytes": 0},
                },
                {
                    "name": "ascend",
                    "accelerator": "npu",
                    "available": True,
                    "device_count": 8,
                    "memory": {"peak_memory_allocated_bytes": 1024},
                },
            ],
        }
    )

    assert evidence["devices"]["accelerators"] == ["cpu", "cuda", "npu"]
    assert evidence["devices"]["available_accelerators"] == ["cpu", "npu"]
    assert evidence["devices"]["device_counts"]["npu"] == 8
    assert evidence["devices"]["peak_memory_allocated_bytes"]["npu"] == 1024
