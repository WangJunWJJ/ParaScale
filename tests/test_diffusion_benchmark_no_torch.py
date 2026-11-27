# -*- coding: utf-8 -*-
# @Time : 2026/11/24 下午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Reject incomparable diffusion measurements before calculating any gain."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("custom", ["external_library", "local_module"])
def test_native_rejects_custom_components_before_loading(tmp_path, monkeypatch, custom):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL.Image")
    from tests.benchmarks.scripts import benchmark_diffusion_inference as benchmark

    model = tmp_path / "model"
    model.mkdir()
    library = "untrusted_library" if custom == "external_library" else "diffusers"
    (model / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "unet": [library, "UNet2DConditionModel"],
            }
        )
    )
    if custom == "local_module":
        (model / "unet").mkdir()
        (model / "unet" / "diffusers.py").write_text(
            "raise RuntimeError('must not import')"
        )

    def forbidden(*args, **kwargs):
        pytest.fail("custom component reached native pipeline loader")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            StableDiffusionPipeline=SimpleNamespace(from_pretrained=forbidden)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16"))
    args = SimpleNamespace(
        model=model,
        prompts=tmp_path / "prompts.jsonl",
        batch_size=1,
        height=32,
        width=32,
        steps=2,
        requests=1,
    )
    config = benchmark.configuration(args, "native-eager", tmp_path / "output")
    with pytest.raises(ValueError, match="custom|component"):
        benchmark.native_run(config)


def reports():
    modes = ("native-eager", "parascale-eager", "native-compiled", "parascale-compiled")
    return [
        {
            "mode": mode,
            "repeat": repeat,
            "model_assets_sha256": {"unet": "fixed"},
            "input_sha256": "fixed-input",
            "settings": {"steps": 30},
            "images": 8,
            "images_per_second": rate,
            "manifest": "unused",
            "wall_seconds": 10,
        }
        for repeat in range(3)
        for mode, rate in zip(modes, (1.0, 0.99, 1.4, 1.38))
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_assets_sha256", {"unet": "different"}),
        ("input_sha256", "different"),
        ("settings", {"steps": 20}),
        ("images", 7),
    ],
)
def test_comparison_rejects_mismatched_work(field, value):
    from tests.benchmarks.scripts.benchmark_diffusion_inference import validate_reports

    rows = reports()
    rows[-1][field] = value
    with pytest.raises(ValueError, match=field):
        validate_reports(rows)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_comparison_rejects_invalid_throughput(value):
    from tests.benchmarks.scripts.benchmark_diffusion_inference import validate_reports

    rows = reports()
    rows[0]["images_per_second"] = value
    with pytest.raises(ValueError, match="throughput"):
        validate_reports(rows)


def test_comparison_requires_complete_unique_repeat_groups():
    from tests.benchmarks.scripts.benchmark_diffusion_inference import validate_reports

    for rows in (reports()[:-1], reports() + [copy.deepcopy(reports()[0])]):
        with pytest.raises(ValueError, match="repeat|duplicate"):
            validate_reports(rows)


def test_ratios_include_optimized_native_control():
    from tests.benchmarks.scripts.benchmark_diffusion_inference import throughput_ratios

    result = throughput_ratios(reports())
    assert result["parascale_compiled_vs_eager"]["median"] == pytest.approx(1.38 / 0.99)
    assert result["parascale_compiled_vs_native_compiled"]["median"] == pytest.approx(
        1.38 / 1.4
    )
    assert result["native_compiled_vs_eager"]["median"] == pytest.approx(1.4)
    assert len(result["parascale_eager_vs_native_eager"]["samples"]) == 3


@pytest.mark.parametrize("level,passed", [(0, True), (1, True), (255, False)])
def test_pixel_gate_preserves_failed_candidates(tmp_path, level, passed):
    image = pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    from tests.benchmarks.scripts.benchmark_diffusion_inference import (
        image_pair_metrics,
    )

    left, right = tmp_path / "left.png", tmp_path / "right.png"
    image.new("RGB", (8, 8)).save(left)
    image.new("RGB", (8, 8), (level,) * 3).save(right)
    result = image_pair_metrics(left, right)
    assert result["passed"] is passed
    assert result["mean_absolute_pixel_error"] == level
    assert result["exact"] is (level == 0)
    if not level:
        assert result["psnr_db"] is None


def test_pixel_gate_rejects_different_shapes(tmp_path):
    image = pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    from tests.benchmarks.scripts.benchmark_diffusion_inference import (
        image_pair_metrics,
    )

    left, right = tmp_path / "left.png", tmp_path / "right.png"
    image.new("RGB", (8, 8)).save(left)
    image.new("RGB", (8, 16)).save(right)
    with pytest.raises(ValueError, match="shape"):
        image_pair_metrics(left, right)


@pytest.mark.parametrize("repeats", [1, 3])
def test_failed_pilot_stops_before_more_timed_repeats(tmp_path, monkeypatch, repeats):
    from tests.benchmarks.scripts import benchmark_diffusion_inference as benchmark

    calls = []

    def run(command, **kwargs):
        mode = command[command.index("--worker") + 1]
        repeat = int(command[command.index("--repeat-index") + 1])
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        row = copy.deepcopy(
            next(
                row
                for row in reports()
                if row["mode"] == mode and row["repeat"] == repeat
            )
        )
        (destination / "measurement.json").write_text(json.dumps(row))
        calls.append((repeat, mode))

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    monkeypatch.setattr(
        benchmark, "compare_images", lambda *args, **kwargs: {"passed": False}
    )
    destination = tmp_path / "results"
    assert (
        benchmark.main(
            [
                "--model",
                str(tmp_path / "model"),
                "--prompts",
                str(tmp_path / "prompts.jsonl"),
                "--output",
                str(destination),
                "--repeats",
                str(repeats),
            ]
        )
        == 1
    )
    assert len(calls) == 4
    result = json.loads((destination / "comparison.json").read_text())
    assert result["status"] == "fidelity_failed"
    assert len(result["runs"]) == 4
