# -*- coding: utf-8 -*-

import json

from tests.benchmarks.tools.summarize_cross_hardware_clip_datacomp import build_report


def _write_run(path, throughput):
    path.write_text(
        json.dumps(
            {
                "mode": "benchmark",
                "runtime_status": "real_local",
                "metrics": {
                    "stable_end_to_end_image_text_pairs_per_second": throughput,
                    "stable_loss": 2.1,
                    "stable_peak_memory_bytes": 2 * 1024**3,
                    "stable_dataloader_wait_ms": 3.0,
                },
                "train": {
                    "backend": "native_ddp",
                    "global_step": 80,
                },
            }
        ),
        encoding="utf-8",
    )


def test_cross_hardware_summary_uses_4090_baseline(tmp_path):
    rtx = tmp_path / "rtx4090.json"
    ascend = tmp_path / "ascend.json"
    _write_run(rtx, 100.0)
    _write_run(ascend, 75.0)

    report = build_report(
        [
            {
                "label": "ascend",
                "hardware": "Ascend 910B4",
                "image": "ascend:image",
                "path": str(ascend),
            },
            {
                "label": "rtx4090",
                "hardware": "dual RTX 4090D",
                "image": "cuda:image",
                "path": str(rtx),
            },
        ],
        suite_id="unit",
        dataset="datacomp_10k_wds",
        model="clip_medium",
        precision="fp32",
        steps=80,
        warmup_steps=10,
        batch_size=8,
    )

    assert report["passed"] is True
    comparisons = {item["label"]: item for item in report["comparisons"]}
    assert comparisons["rtx4090"]["relative_to_baseline"] == 1.0
    assert comparisons["ascend"]["relative_to_baseline"] == 0.75
