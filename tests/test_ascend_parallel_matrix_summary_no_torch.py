# -*- coding: utf-8 -*-
# @Time : 2026/7/20 下午1:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json

from tests.benchmarks.tools.summarize_ascend_parallel_matrix import build_report


def _write_run(path, run_id, throughput, loss=1.0):
    path.write_text(
        json.dumps(
            {
                "mode": "benchmark",
                "runtime_status": "executed",
                "metrics": {
                    "stable_end_to_end_image_text_pairs_per_second": throughput,
                    "stable_loss": loss,
                    "stable_peak_memory_bytes": 2 * 1024**3,
                    "stable_dataloader_wait_ms": 3.0,
                },
                "train": {
                    "backend": "native_ddp",
                    "global_step": 80,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_parallel_matrix_sums_concurrent_container_throughput(tmp_path):
    _write_run(tmp_path / "single_docker_2card.json", "single_docker_2card", 100.0)
    _write_run(tmp_path / "two_docker_1card_a.json", "two_docker_1card_a", 60.0)
    _write_run(tmp_path / "two_docker_1card_b.json", "two_docker_1card_b", 62.0)
    _write_run(tmp_path / "two_docker_2card_a.json", "two_docker_2card_a", 105.0)
    _write_run(tmp_path / "two_docker_2card_b.json", "two_docker_2card_b", 107.0)

    report = build_report(
        tmp_path,
        suite_id="unit",
        hardware="Ascend 910B4",
        image="unit:image",
        steps=80,
        warmup_steps=10,
        batch_size=8,
    )

    by_scenario = {item["scenario"]: item for item in report["scenarios"]}
    assert report["passed"] is True
    assert by_scenario["single_docker_2card"]["aggregate_throughput"] == 100.0
    assert by_scenario["two_docker_1card"]["aggregate_throughput"] == 122.0
    assert by_scenario["two_docker_1card"]["throughput_per_card"] == 61.0
    assert by_scenario["two_docker_2card"]["aggregate_throughput"] == 212.0
    assert by_scenario["two_docker_2card"]["throughput_per_card"] == 53.0


def test_parallel_matrix_marks_missing_component_failed(tmp_path):
    _write_run(tmp_path / "single_docker_2card.json", "single_docker_2card", 100.0)

    report = build_report(
        tmp_path,
        suite_id="unit",
        hardware="Ascend 910B4",
        image="unit:image",
        steps=80,
        warmup_steps=10,
        batch_size=8,
    )

    by_scenario = {item["scenario"]: item for item in report["scenarios"]}
    assert report["passed"] is False
    assert by_scenario["two_docker_1card"]["ok"] is False
    assert by_scenario["two_docker_1card"]["missing"] == [
        "two_docker_1card_a",
        "two_docker_1card_b",
    ]
