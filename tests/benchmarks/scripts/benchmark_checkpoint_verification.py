# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Warm-cache integrity-inspection A/B on one immutable trusted stage checkpoint.

The baseline is trusted Python source executed with the candidate's shared
helpers. No payload is deserialized. This is not a training, save, recovery,
fsync, retention, cold-cache, or end-to-end performance benchmark.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from parascale.parallel import checkpoint_catalog  # noqa: E402
from parascale.parallel.distributed_checkpoint import _digest  # noqa: E402


def _load_baseline(path: Path) -> ModuleType:
    if path.suffix != ".py" or not path.is_file():
        raise ValueError("baseline_catalog must be a trusted .py source file")
    name = "parascale.parallel._catalog_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load baseline catalog source")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    if not callable(getattr(module, "inspect_stage_checkpoint", None)):
        raise ValueError("Baseline must expose inspect_stage_checkpoint")
    return module


def _canonical(metadata: dict) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _snapshot(directory: Path) -> list[tuple]:
    result = []
    for path in sorted(directory.iterdir()):
        info = path.lstat()
        result.append(
            (
                path.name,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ino,
                info.st_dev,
            )
        )
    return result


def _environment() -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError, NotImplementedError):
        affinity = None
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
    }


def run_benchmark(checkpoint: Path, baseline_catalog: Path, repeats: int = 3) -> dict:
    """Inspect immutable files in alternating AB/BA order without deserialization.

    Args:
        checkpoint: Trusted stage checkpoint directory with committed metadata.
        baseline_catalog: Trusted Python catalog source, executed on import.
        repeats: Number of measured pairs after equal warmup, at least three.

    Returns:
        JSON-compatible timings, source hashes, environment and scope details.

    Raises:
        ValueError: Invalid arguments, corrupt or changing inputs, mismatched
            metadata, or nonpositive/nonfinite timing samples.
        OSError: Source or checkpoint files cannot be read.
    """
    if type(repeats) is not int or repeats < 3:
        raise ValueError("repeats must be an integer >= 3")
    checkpoint, baseline_catalog = Path(checkpoint), Path(baseline_catalog)
    sources = {
        "baseline": baseline_catalog,
        "candidate": Path(checkpoint_catalog.__file__),
    }
    source_hashes = {name: _digest(path) for name, path in sources.items()}
    baseline = _load_baseline(baseline_catalog)
    implementations = {
        "baseline": baseline.inspect_stage_checkpoint,
        "candidate": checkpoint_catalog.inspect_stage_checkpoint,
    }
    initial = _snapshot(checkpoint)
    expected = None

    def validate(metadata: dict) -> None:
        nonlocal expected
        canonical = _canonical(metadata)
        if expected is None:
            expected = canonical
        elif canonical != expected:
            raise ValueError("Checkpoint metadata mismatch between inspections")
        if _snapshot(checkpoint) != initial:
            raise ValueError("Checkpoint files changed; immutable inputs required")

    for inspect in implementations.values():
        validate(inspect(checkpoint))

    durations: dict[str, list[float]] = {name: [] for name in implementations}
    order, measurements = [], []
    for repeat in range(repeats):
        pair = (
            ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
        )
        order.append(pair)
        for name in pair:
            started = time.perf_counter()
            metadata = implementations[name](checkpoint)
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(
                    "Inspection duration must be finite and strictly positive"
                )
            validate(metadata)
            durations[name].append(elapsed)
            measurements.append(
                {"repeat": repeat, "implementation": name, "seconds": elapsed}
            )

    # Hashing source, checking stable inputs, and building reports are not timed.
    for name, path in sources.items():
        if _digest(path) != source_hashes[name]:
            raise ValueError(f"{name} catalog source changed during benchmark")
    results = {
        name: {
            "source_path": str(sources[name].resolve()),
            "source_sha256": source_hashes[name],
            "seconds": values,
            "median_seconds": statistics.median(values),
            "mean_seconds": statistics.mean(values),
            "stdev_seconds": statistics.stdev(values),
        }
        for name, values in durations.items()
    }
    metadata = json.loads(expected)
    return {
        "schema_version": 1,
        "benchmark": "stage_checkpoint_integrity_inspection",
        "checkpoint": str(checkpoint.absolute()),
        "repeats": repeats,
        "warmup_calls_per_implementation": 1,
        "order": order,
        "measurements": measurements,
        "results": results,
        "canonical_metadata": metadata,
        "total_payload_bytes": sum(
            (checkpoint / entry["path"]).stat().st_size for entry in metadata["files"]
        ),
        "median_verification_speedup": (
            results["baseline"]["median_seconds"]
            / results["candidate"]["median_seconds"]
        ),
        "environment": _environment(),
        "scope": {
            "operation": "inspect_stage_checkpoint integrity inspection only",
            "cache": "warm cache; no cache eviction",
            "inputs": "same immutable trusted checkpoint files; payloads remain opaque",
            "baseline": "trusted catalog source with identical candidate shared helpers",
            "timing": "perf_counter around inspection only; metadata comparison and source hashing excluded",
            "excluded": [
                "training",
                "save",
                "fsync",
                "retention",
                "deserialization",
                "recovery",
                "cold cache",
            ],
            "end_to_end_claim": False,
            "qualification": "Input provenance must be established separately; unit fixtures are not trained-model performance evidence.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--baseline-catalog",
        required=True,
        type=Path,
        help="Trusted Python catalog source; executes code with current shared helpers.",
    )
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New runtime JSON artifact outside the repository and checkpoint.",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.is_relative_to(args.checkpoint.resolve()):
        parser.error("--output must be outside the repository and checkpoint directory")
    if output.exists():
        raise FileExistsError(output)
    report = run_benchmark(args.checkpoint, args.baseline_catalog, args.repeats)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
