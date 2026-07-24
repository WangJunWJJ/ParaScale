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
