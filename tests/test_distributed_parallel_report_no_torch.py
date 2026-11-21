# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:36
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Distributed parallel report integration needs only summary artifacts."""

import builtins
import json

import pytest

from tests.benchmarks.tools.build_benchmark_report import (
    build_report_markdown,
    load_summaries,
)


@pytest.fixture(autouse=True)
def forbid_torch_import(monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Report generation must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)


def test_distributed_parallel_summary_renders_diagnostic_evidence(tmp_path):
    summary = {
        "suite": "A6000 distributed parallel capability",
        "status": "recorded",
        "hardware": "host156 max4 A6000",
        "image": "validation-image:torch",
        "scope": "Model-specific experimental execution parity.",
        "runs": [
            {
                "name": "tp2-pp2",
                "model": "tiny_transformer",
                "dp": 1,
                "tp": 2,
                "pp": 2,
                "status": "passed",
                "max_absolute_errors": {
                    "loss": 1e-7,
                    "output": 2e-7,
                    "gradient": 3e-7,
                    "parameter": 4e-9,
                },
                "forward_backward_samples_per_second": 123.456,
                "evidence": "distributed_parallel/tp2-pp2.json",
            },
        ],
        "limitations": ["Synthetic inputs only.", "At most four GPUs tested."],
    }
    path = tmp_path / "distributed_parallel" / "summary.json"
    path.parent.mkdir()
    path.write_text(json.dumps(summary), encoding="utf-8")

    summaries = load_summaries(tmp_path)
    assert summaries["distributed_parallel"]["runs"] == summary["runs"]
    text = build_report_markdown(summaries, report_root=tmp_path)

    assert (
        "A6000 distributed parallel capability | recorded | host156 max4 A6000" in text
    )
    assert "`validation-image:torch`" in text
    assert (
        "[distributed_parallel/summary.json](distributed_parallel/summary.json)" in text
    )
    assert "## A6000 Distributed Parallel Capability" in text
    assert summary["scope"] in text
    assert (
        "| tp2-pp2 | tiny_transformer | DP=1 TP=2 PP=2 | passed | 4.000000e-09 | 123.456 |"
        in text
    )
    assert (
        "[distributed_parallel/tp2-pp2.json](distributed_parallel/tp2-pp2.json)" in text
    )
    assert "Max parameter absolute error" in text
    assert "Diagnostic forward/backward samples/s" in text
    assert "not production throughput" in text
    assert "no full-3D" in text
    assert all(limitation in text for limitation in summary["limitations"])


def test_missing_distributed_parallel_summary_is_optional(tmp_path):
    summaries = load_summaries(tmp_path)
    assert summaries["distributed_parallel"]["missing"] is True
    text = build_report_markdown(summaries, report_root=tmp_path)
    assert "A6000 distributed parallel capability | missing |" in text
    assert "## A6000 Distributed Parallel Capability" not in text
    assert "## Dual RTX 4090 Validation" in text


def test_existing_callers_can_omit_distributed_parallel_key(tmp_path):
    summaries = load_summaries(tmp_path)
    summaries.pop("distributed_parallel", None)
    text = build_report_markdown(summaries, report_root=tmp_path)
    assert "## A6000 Distributed Parallel Capability" not in text
    assert "## Evidence Files" in text
