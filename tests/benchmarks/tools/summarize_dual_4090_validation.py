# -*- coding: utf-8 -*-
# @Time : 2026/7/12 下午1:22
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Summarize dual-4090 functional and performance validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

THROUGHPUT_KEYS = (
    "stable_end_to_end_image_text_pairs_per_second",
    "end_to_end_image_text_pairs_per_second",
    "stable_end_to_end_images_per_second",
    "end_to_end_images_per_second",
    "stable_samples_per_second",
    "samples_per_second",
    "steps_per_second",
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": str(exc),
        }
    payload.setdefault("path", str(path))
    return payload


def _model_from_name(path: Path) -> str:
    name = path.stem
    if name.endswith(".error"):
        name = name[: -len(".error")]
    return name.split("_", 1)[0]


def _backend_from_payload(path: Path, payload: Dict[str, Any]) -> str:
    train = payload.get("train", {})
    if isinstance(train, dict) and train.get("backend"):
        return str(train["backend"])
    if payload.get("backend"):
        return str(payload["backend"])
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 1 else path.stem


def _metric(metrics: Dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        try:
            value = float(metrics.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 0.0


def _loss(payload: Dict[str, Any]) -> float | None:
    metrics = payload.get("metrics", {})
    train = payload.get("train", {})
    last_metrics = train.get("last_metrics", {}) if isinstance(train, dict) else {}
    for source in (metrics, last_metrics):
        for key in ("loss", "stable_loss", "last_loss"):
            if key in source:
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    return None
    return None


def _run_record(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics", {})
    train = payload.get("train", {})
    if not isinstance(train, dict):
        train = {}
    status = "error" if path.name.endswith(".error.json") else payload.get("status", "ok")
    throughput = _metric(metrics, THROUGHPUT_KEYS)
    peak_memory = _metric(metrics, ("peak_memory_bytes",))
    return {
        "run_id": path.stem.removesuffix(".error"),
        "model": _model_from_name(path),
        "backend": _backend_from_payload(path, payload),
        "status": status,
        "ok": status == "ok" and payload.get("runtime_status") != "plan_only",
        "runtime_status": payload.get("runtime_status"),
        "capability_level": payload.get("capability_level"),
        "global_step": train.get("global_step", payload.get("global_step")),
        "loss": _loss(payload),
        "throughput": throughput,
        "throughput_metric": next(
            (key for key in THROUGHPUT_KEYS if float(metrics.get(key, 0.0) or 0.0) > 0),
            None,
        ),
        "peak_memory_bytes": peak_memory,
        "dataloader_wait_ms": _metric(metrics, ("dataloader_wait_ms",)),
        "path": str(path),
        "error": payload.get("error"),
    }


def _collect_runs(input_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json", "comparison.json", "report.json"}:
            continue
        runs.append(_run_record(path, _read_json(path)))
    return runs


def _summaries(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    models = sorted({str(run["model"]) for run in runs})
    summaries: List[Dict[str, Any]] = []
    for model in models:
        model_runs = [run for run in runs if run["model"] == model]
        ok_runs = [run for run in model_runs if run.get("ok")]
        best = max(ok_runs, key=lambda run: float(run.get("throughput") or 0.0), default=None)
        summaries.append(
            {
                "model": model,
                "ok_runs": len(ok_runs),
                "total_runs": len(model_runs),
                "best_backend": best.get("backend") if best else None,
                "best_throughput": best.get("throughput") if best else 0.0,
                "best_loss": best.get("loss") if best else None,
                "best_peak_memory_bytes": best.get("peak_memory_bytes") if best else 0.0,
            }
        )
    return summaries


def build_report(
    input_dir: Path,
    *,
    suite_id: str,
    hardware: str,
    image: str,
) -> Dict[str, Any]:
    runs = _collect_runs(input_dir)
    summaries = _summaries(runs)
    return {
        "mode": "dual_4090_validation_summary",
        "suite_id": suite_id,
        "hardware": hardware,
        "image": image,
        "input_dir": str(input_dir),
        "passed": bool(runs) and all(item["ok_runs"] > 0 for item in summaries),
        "summaries": summaries,
        "runs": runs,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Dual RTX 4090 Functional and Performance Validation",
        "",
        f"- Suite: `{report['suite_id']}`",
        f"- Hardware: {report['hardware']}",
        f"- Image: `{report['image']}`",
        f"- Passed: {report['passed']}",
        f"- Input directory: `{report['input_dir']}`",
        "",
        "## Model Summary",
        "",
        "| Model | OK runs | Total runs | Best backend | Best throughput | Loss | Peak memory GB |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in report["summaries"]:
        lines.append(
            "| {model} | {ok_runs} | {total_runs} | {backend} | {throughput:.3f} | {loss} | {memory:.3f} |".format(
                model=item["model"],
                ok_runs=item["ok_runs"],
                total_runs=item["total_runs"],
                backend=item.get("best_backend") or "n/a",
                throughput=float(item.get("best_throughput") or 0.0),
                loss=(
                    f"{float(item['best_loss']):.6f}"
                    if item.get("best_loss") is not None
                    else "n/a"
                ),
                memory=float(item.get("best_peak_memory_bytes") or 0.0) / 1024**3,
            )
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Model | Backend | OK | Runtime | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for run in report["runs"]:
        lines.append(
            "| {run_id} | {model} | {backend} | {ok} | {runtime} | {step} | {throughput:.3f} | {metric} | {loss} | {memory:.3f} | {wait:.3f} |".format(
                run_id=run["run_id"],
                model=run["model"],
                backend=run["backend"],
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
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--hardware", default="dual RTX 4090D")
    parser.add_argument("--image", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(
        Path(args.input_dir),
        suite_id=args.suite_id,
        hardware=args.hardware,
        image=args.image,
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
