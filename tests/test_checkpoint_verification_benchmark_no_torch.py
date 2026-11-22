# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:44
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Opaque unit fixtures only: these timings are not trained-model evidence."""

import copy
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_stage_checkpoint_catalog_no_torch import checkpoint


def benchmark():
    from tests.benchmarks.scripts import benchmark_checkpoint_verification

    return benchmark_checkpoint_verification


@pytest.fixture
def baseline(tmp_path):
    from parascale.parallel import checkpoint_catalog

    path = tmp_path / "baseline_catalog.py"
    shutil.copyfile(checkpoint_catalog.__file__, path)
    return path


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_matched_immutable_unit_files_and_report(tmp_path, baseline, ranks):
    module = benchmark()
    directory = checkpoint(tmp_path / "opaque-unit-only", 3, ranks=ranks)
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = module.run_benchmark(directory, baseline)
    assert result["repeats"] == 3
    assert result["warmup_calls_per_implementation"] == 1
    assert result["order"] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
    ]
    assert result["total_payload_bytes"] == sum(
        len(value) for name, value in original.items() if name.endswith(".pt")
    )
    assert result["canonical_metadata"]["step"] == 3
    assert result["median_verification_speedup"] > 0
    for variant in ("baseline", "candidate"):
        report = result["results"][variant]
        assert len(report["seconds"]) == 3
        assert all(value > 0 for value in report["seconds"])
        assert report["median_seconds"] == statistics.median(report["seconds"])
        assert report["mean_seconds"] == statistics.mean(report["seconds"])
        assert report["stdev_seconds"] == statistics.stdev(report["seconds"])
        source = Path(report["source_path"])
        assert (
            report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        )
    assert result["environment"]["python"]
    assert result["environment"]["platform"]
    assert "cpu_affinity" in result["environment"]
    assert result["scope"]["cache"] == "warm cache; no cache eviction"
    assert result["scope"]["end_to_end_claim"] is False
    assert {"training", "save", "fsync", "retention"} <= set(
        result["scope"]["excluded"]
    )
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_warmup_measurement_order_counts_and_clock(tmp_path, baseline, monkeypatch):
    module = benchmark()
    directory = checkpoint(tmp_path / "unit", 1, ranks=2)
    calls = []
    original_load = module._load_baseline
    candidate = module.checkpoint_catalog.inspect_stage_checkpoint

    def wrap(implementation, name):
        def inspect(path):
            assert path == directory
            calls.append(name)
            return implementation(path)

        return inspect

    def load(path):
        loaded = original_load(path)
        assert loaded.__name__ == "parascale.parallel._catalog_baseline"
        assert loaded._digest is module.checkpoint_catalog._digest
        loaded.inspect_stage_checkpoint = wrap(
            loaded.inspect_stage_checkpoint, "baseline"
        )
        return loaded

    monkeypatch.setattr(module, "_load_baseline", load)
    monkeypatch.setattr(
        module.checkpoint_catalog,
        "inspect_stage_checkpoint",
        wrap(candidate, "candidate"),
    )
    times = iter([0, 2, 2, 3, 3, 4, 4, 8, 8, 14, 14, 17])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    result = module.run_benchmark(directory, baseline)
    assert calls == [
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert result["results"]["baseline"]["seconds"] == [2, 4, 6]
    assert result["results"]["candidate"]["seconds"] == [1, 1, 3]
    assert result["median_verification_speedup"] == 4
    assert [row["implementation"] for row in result["measurements"]] == calls[2:]


@pytest.mark.parametrize("call_number", [1, 3])
def test_all_metadata_must_match_canonically(
    tmp_path, baseline, monkeypatch, call_number
):
    module = benchmark()
    directory = checkpoint(tmp_path / "unit", 1)
    original = module.checkpoint_catalog.inspect_stage_checkpoint
    count = 0

    def inspect(path):
        nonlocal count
        count += 1
        report = copy.deepcopy(original(path))
        if count == call_number:
            report["runtime"]["version"] = True  # Python equality treats True as 1.
        return report

    monkeypatch.setattr(module.checkpoint_catalog, "inspect_stage_checkpoint", inspect)
    with pytest.raises(ValueError, match="metadata"):
        module.run_benchmark(directory, baseline)


@pytest.mark.parametrize("repeats", [True, 0, 2, 3.0, "3", None])
def test_invalid_repeats_rejected_before_loading(tmp_path, repeats):
    with pytest.raises(ValueError, match="repeats"):
        benchmark().run_benchmark(
            tmp_path / "missing", tmp_path / "missing.py", repeats
        )


@pytest.mark.parametrize("elapsed", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_clock_durations_are_rejected(tmp_path, baseline, monkeypatch, elapsed):
    module = benchmark()
    directory = checkpoint(tmp_path / "unit", 1)
    clock = iter([0.0, elapsed])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    with pytest.raises(ValueError, match="positive|finite"):
        module.run_benchmark(directory, baseline)


def test_small_positive_durations_are_not_clamped(tmp_path, baseline, monkeypatch):
    module = benchmark()
    directory = checkpoint(tmp_path / "unit", 1)
    clock = iter([0.0, 1e-12] * 6)
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    result = module.run_benchmark(directory, baseline)
    assert result["results"]["candidate"]["seconds"] == [1e-12] * 3


def test_corrupt_payload_is_not_benchmarked(tmp_path, baseline):
    directory = checkpoint(tmp_path / "unit", 1, ranks=4)
    (directory / "rank-00002.pt").write_bytes(b"corrupt opaque unit bytes")
    with pytest.raises(ValueError, match="checksum"):
        benchmark().run_benchmark(directory, baseline)


def test_blocked_torch_cli_and_required_output(tmp_path, baseline):
    directory = checkpoint(tmp_path / "unit", 1, ranks=2)
    output = tmp_path / "runtime-artifacts" / "benchmark.json"
    script = (
        Path(__file__).resolve().parent
        / "benchmarks/scripts/benchmark_checkpoint_verification.py"
    )
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("Benchmark must not import Torch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    args = [
        sys.executable,
        "-c",
        code,
        str(script),
        "--checkpoint",
        str(directory),
        "--baseline-catalog",
        str(baseline),
    ]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    missing = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
    assert missing.returncode != 0 and "--output" in missing.stderr
    result = subprocess.run(
        [*args, "--output", str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["repeats"] == 3 and len(report["measurements"]) == 6
    assert report["scope"]["end_to_end_claim"] is False


def test_input_mutation_aborts_report(tmp_path, baseline, monkeypatch):
    module = benchmark()
    directory = checkpoint(tmp_path / "unit", 1)
    original = module.checkpoint_catalog.inspect_stage_checkpoint

    def inspect(path):
        metadata = original(path)
        (path / "concurrent-writer.txt").write_text(
            "unit failure injection", encoding="utf-8"
        )
        return metadata

    monkeypatch.setattr(module.checkpoint_catalog, "inspect_stage_checkpoint", inspect)
    with pytest.raises(ValueError, match="immutable"):
        module.run_benchmark(directory, baseline)


def test_source_mutation_aborts_report(tmp_path, baseline, monkeypatch):
    module = benchmark()
    directory = checkpoint(tmp_path / "unit", 1)
    original = module.checkpoint_catalog.inspect_stage_checkpoint

    def inspect(path):
        report = original(path)
        with baseline.open("a", encoding="utf-8") as stream:
            stream.write("\n# unit mutation\n")
        return report

    monkeypatch.setattr(module.checkpoint_catalog, "inspect_stage_checkpoint", inspect)
    with pytest.raises(ValueError, match="source changed"):
        module.run_benchmark(directory, baseline)


@pytest.mark.parametrize("destination", ["repository", "checkpoint", "existing"])
def test_cli_protects_inputs_and_requires_external_new_output(
    tmp_path, baseline, monkeypatch, destination
):
    module = benchmark()
    directory = checkpoint(tmp_path / "unit", 1)
    output = module.ROOT / "benchmark-unit-forbidden.json"
    if destination == "checkpoint":
        output = directory / "result.json"
    elif destination == "existing":
        output = tmp_path / "existing.json"
        output.write_text("preserve", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject output before running benchmark")

    monkeypatch.setattr(module, "run_benchmark", forbidden)
    with pytest.raises(FileExistsError if destination == "existing" else SystemExit):
        module.main(
            [
                "--checkpoint",
                str(directory),
                "--baseline-catalog",
                str(baseline),
                "--output",
                str(output),
            ]
        )
    if destination == "existing":
        assert output.read_text(encoding="utf-8") == "preserve"
    else:
        assert not output.exists()


def test_baseline_loader_preserves_existing_module_and_shared_helpers(
    tmp_path, baseline, monkeypatch
):
    from types import ModuleType

    from parascale.parallel.distributed_checkpoint import _digest

    name = "parascale.parallel._catalog_baseline"
    previous = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, previous)
    loaded = benchmark()._load_baseline(baseline)
    assert loaded._digest is _digest
    assert sys.modules[name] is previous


def test_cpu_affinity_is_recorded_when_available(tmp_path, baseline, monkeypatch):
    module = benchmark()
    monkeypatch.setattr(
        module.os, "sched_getaffinity", lambda pid: {3, 1}, raising=False
    )
    directory = checkpoint(tmp_path / "unit", 1)
    assert module.run_benchmark(directory, baseline)["environment"]["cpu_affinity"] == [
        1,
        3,
    ]
