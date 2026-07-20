# -*- coding: utf-8 -*-
# @Time : 2026/7/20 下午1:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Summarize Ascend multi-container parallel benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tests.benchmarks.tools.common import (
    IMAGE_TEXT_THROUGHPUT_KEYS,
    first_metric,
    loss_value,
    read_json_payload,
    run_id_from_path,
    train_section,
)

SCENARIOS = {
    "single_docker_2card": {
        "containers": 1,
        "cards": 2,
        "components": ("single_docker_2card",),
    },
    "two_docker_1card": {
        "containers": 2,
        "cards": 2,
        "components": ("two_docker_1card_a", "two_docker_1card_b"),
    },
    "two_docker_2card": {
        "containers": 2,
        "cards": 4,
        "components": ("two_docker_2card_a", "two_docker_2card_b"),
    },
}


def _record(path: Path) -> Dict[str, Any]:
    payload = read_json_payload(path)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    train = train_section(payload)
    throughput, throughput_metric = first_metric(metrics, IMAGE_TEXT_THROUGHPUT_KEYS)
    peak_memory, _ = first_metric(
        metrics,
        ("stable_peak_memory_bytes", "peak_memory_bytes"),
    )
    dataloader_wait, _ = first_metric(
        metrics,
        ("stable_dataloader_wait_ms", "dataloader_wait_ms"),
    )
    is_error = path.name.endswith(".error.json") or payload.get("status") == "error"
    return {
        "run_id": run_id_from_path(path),
        "ok": not is_error
        and payload.get("ok", True) is not False
        and payload.get("runtime_status") != "plan_only",
        "status": payload.get("status", "error" if is_error else "ok"),
        "runtime_status": payload.get("runtime_status"),
        "backend": train.get("backend") or payload.get("backend"),
        "global_step": train.get("global_step", payload.get("global_step")),
        "throughput": throughput,
        "throughput_metric": throughput_metric,
        "loss": loss_value(payload, keys=("stable_loss", "loss", "last_loss")),
        "peak_memory_bytes": peak_memory,
        "dataloader_wait_ms": dataloader_wait,
        "returncode": payload.get("returncode"),
        "error": payload.get("error"),
        "path": str(path),
    }


def _collect(input_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json"}:
            continue
        runs.append(_record(path))
    return runs


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _scenario_summary(
    scenario: str,
    spec: Dict[str, Any],
    run_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    components = [run_by_id.get(run_id) for run_id in spec["components"]]
    missing = [
        run_id for run_id, run in zip(spec["components"], components) if run is None
    ]
    present = [run for run in components if run is not None]
    ok = bool(present) and not missing and all(run.get("ok") for run in present)
    aggregate = sum(float(run.get("throughput") or 0.0) for run in present)
    memory_values = [float(run.get("peak_memory_bytes") or 0.0) for run in present]
    return {
        "scenario": scenario,
        "containers": spec["containers"],
        "cards": spec["cards"],
        "components": list(spec["components"]),
        "missing": missing,
        "ok": ok,
        "aggregate_throughput": aggregate,
        "throughput_per_card": aggregate / spec["cards"] if spec["cards"] else 0.0,
        "mean_loss": _mean(run.get("loss") for run in present),
        "peak_memory_bytes_sum": sum(memory_values),
        "peak_memory_bytes_max": max(memory_values, default=0.0),
        "mean_dataloader_wait_ms": _mean(
            run.get("dataloader_wait_ms") for run in present
        ),
    }


def build_report(
    input_dir: Path,
    *,
    suite_id: str,
    hardware: str,
    image: str,
    steps: int,
    warmup_steps: int,
    batch_size: int,
) -> Dict[str, Any]:
    runs = _collect(input_dir)
    run_by_id = {run["run_id"]: run for run in runs}
    scenarios = [
        _scenario_summary(name, spec, run_by_id) for name, spec in SCENARIOS.items()
    ]
    return {
        "mode": "ascend_parallel_matrix_summary",
        "suite_id": suite_id,
        "hardware": hardware,
        "image": image,
        "input_dir": str(input_dir),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "passed": bool(runs) and all(item["ok"] for item in scenarios),
        "scenarios": scenarios,
        "runs": runs,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Ascend Parallel Training Matrix",
        "",
        f"- Suite: `{report['suite_id']}`",
        f"- Hardware: {report['hardware']}",
        f"- Image: `{report['image']}`",
        f"- Steps: {report['steps']}",
        f"- Warmup steps: {report['warmup_steps']}",
        f"- Batch size: {report['batch_size']}",
        f"- Passed: {report['passed']}",
        f"- Input directory: `{report['input_dir']}`",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Containers | NPUs | OK | Aggregate pairs/s | Pairs/s/NPU | Loss | Peak memory GB sum | Peak memory GB max | Dataloader wait ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["scenarios"]:
        lines.append(
            "| {scenario} | {containers} | {cards} | {ok} | {throughput:.3f} | {per_card:.3f} | {loss} | {memory_sum:.3f} | {memory_max:.3f} | {wait} |".format(
                scenario=item["scenario"],
                containers=item["containers"],
                cards=item["cards"],
                ok=item["ok"],
                throughput=float(item.get("aggregate_throughput") or 0.0),
                per_card=float(item.get("throughput_per_card") or 0.0),
                loss=(
                    f"{float(item['mean_loss']):.6f}"
                    if item.get("mean_loss") is not None
                    else "n/a"
                ),
                memory_sum=float(item.get("peak_memory_bytes_sum") or 0.0) / 1024**3,
                memory_max=float(item.get("peak_memory_bytes_max") or 0.0) / 1024**3,
                wait=(
                    f"{float(item['mean_dataloader_wait_ms']):.3f}"
                    if item.get("mean_dataloader_wait_ms") is not None
                    else "n/a"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Component Runs",
            "",
            "| Run | Backend | OK | Runtime | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms | Return code |",
            "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in report["runs"]:
        lines.append(
            "| {run_id} | {backend} | {ok} | {runtime} | {step} | {throughput:.3f} | {metric} | {loss} | {memory:.3f} | {wait:.3f} | {returncode} |".format(
                run_id=run["run_id"],
                backend=run.get("backend") or "n/a",
                ok=run["ok"],
                runtime=run.get("runtime_status") or "n/a",
                step=run.get("global_step") or 0,
                throughput=float(run.get("throughput") or 0.0),
                metric=run.get("throughput_metric") or "n/a",
                loss=(
                    f"{float(run['loss']):.6f}"
                    if run.get("loss") is not None
                    else "n/a"
                ),
                memory=float(run.get("peak_memory_bytes") or 0.0) / 1024**3,
                wait=float(run.get("dataloader_wait_ms") or 0.0),
                returncode=(
                    run.get("returncode")
                    if run.get("returncode") is not None
                    else "n/a"
                ),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--hardware", default="Ascend 910B4")
    parser.add_argument("--image", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(
        Path(args.input_dir),
        suite_id=args.suite_id,
        hardware=args.hardware,
        image=args.image,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, Path(args.markdown))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
