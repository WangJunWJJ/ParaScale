# -*- coding: utf-8 -*-
# @Time : 2026/7/20 下午1:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Summarize strict CLIP DataComp runs across hardware targets."""

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
    train_section,
)


def _run_record(label: str, hardware: str, image: str, path: Path) -> Dict[str, Any]:
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
        "label": label,
        "hardware": hardware,
        "image": image,
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
        "path": str(path),
        "error": payload.get("error"),
        "returncode": payload.get("returncode"),
    }


def build_report(
    runs: List[Dict[str, str]],
    *,
    suite_id: str,
    dataset: str,
    model: str,
    precision: str,
    steps: int,
    warmup_steps: int,
    batch_size: int,
) -> Dict[str, Any]:
    records = [
        _run_record(
            item["label"],
            item["hardware"],
            item["image"],
            Path(item["path"]),
        )
        for item in runs
    ]
    baseline = next(
        (record for record in records if record["label"] == "rtx4090"),
        records[0] if records else None,
    )
    comparisons: List[Dict[str, Any]] = []
    if baseline:
        baseline_throughput = float(baseline.get("throughput") or 0.0)
        for record in records:
            throughput = float(record.get("throughput") or 0.0)
            comparisons.append(
                {
                    "label": record["label"],
                    "baseline": baseline["label"],
                    "throughput": throughput,
                    "baseline_throughput": baseline_throughput,
                    "relative_to_baseline": (
                        throughput / baseline_throughput
                        if baseline_throughput > 0
                        else 0.0
                    ),
                }
            )
    return {
        "mode": "cross_hardware_clip_datacomp_summary",
        "suite_id": suite_id,
        "dataset": dataset,
        "model": model,
        "precision": precision,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "passed": bool(records) and all(record["ok"] for record in records),
        "runs": records,
        "comparisons": comparisons,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Cross-Hardware CLIP DataComp Comparison",
        "",
        f"- Suite: `{report['suite_id']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Model: `{report['model']}`",
        f"- Precision: `{report['precision']}`",
        f"- Steps: {report['steps']}",
        f"- Warmup steps: {report['warmup_steps']}",
        f"- Batch size: {report['batch_size']}",
        f"- Passed: {report['passed']}",
        "",
        "## Runs",
        "",
        "| Label | Hardware | Image | Backend | OK | Runtime | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        lines.append(
            "| {label} | {hardware} | `{image}` | {backend} | {ok} | {runtime} | {step} | {throughput:.3f} | {metric} | {loss} | {memory:.3f} | {wait:.3f} |".format(
                label=run["label"],
                hardware=run["hardware"],
                image=run["image"],
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
            )
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| Label | Baseline | Throughput | Baseline throughput | Relative |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in report["comparisons"]:
        lines.append(
            "| {label} | {baseline} | {throughput:.3f} | {baseline_throughput:.3f} | {relative:.3f} |".format(
                label=item["label"],
                baseline=item["baseline"],
                throughput=float(item.get("throughput") or 0.0),
                baseline_throughput=float(item.get("baseline_throughput") or 0.0),
                relative=float(item.get("relative_to_baseline") or 0.0),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_run_spec(value: str) -> Dict[str, str]:
    parts = value.split("=", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--run must use label=hardware=image=path format"
        )
    label, hardware, image, path = parts
    return {
        "label": label,
        "hardware": hardware,
        "image": image,
        "path": path,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run_spec, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="clip_medium")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(
        args.run,
        suite_id=args.suite_id,
        dataset=args.dataset,
        model=args.model,
        precision=args.precision,
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
