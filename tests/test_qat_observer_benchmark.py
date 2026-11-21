# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午11:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Exercise the reproducible observer A/B driver without requiring a GPU."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from parascale.quantization.observers import BaseObserver  # noqa: E402
from tests.benchmarks.scripts.benchmark_qat_observer import run  # noqa: E402


def test_observer_benchmark_checks_parity_and_restores_implementation(tmp_path):
    args = SimpleNamespace(
        baseline_source=Path(__file__).parents[1],
        output=tmp_path / "result.json",
        device="cpu",
        widths=[8],
        steps=1,
        warmup=0,
        repeats=1,
    )
    original = BaseObserver._range
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        payload = run(args)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cuda.matmul.allow_tf32 = tf32
    assert BaseObserver._range is original
    assert json.loads(args.output.read_text(encoding="utf-8")) == payload
    assert len(payload["rows"]) == 2
    for key in ("state_sha256", "gradient_sha256", "input_sha256"):
        assert payload["rows"][0][key] == payload["rows"][1][key]
    assert len(payload["summary"]) == 1
    assert payload["summary"][0]["speedup"] > 0
    with pytest.raises(FileExistsError):
        run(args)
