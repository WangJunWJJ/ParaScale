# -*- coding: utf-8 -*-
# @Time : 2026/6/11 下午5:35
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Aggregate ParaScale backend benchmark JSON files into a comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

DEFAULT_PRIMARY_METRIC = "end_to_end_image_text_pairs_per_second"
MEMORY_METRIC = "peak_memory_bytes"


def load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    train = payload.get("train", {})
    return {
        "path": str(path),
        "run_id": path.stem,
        "backend": train.get("backend", payload.get("backend", path.stem)),
        "status": "ok",
        "capability_level": payload.get("capability_level"),
        "benchmark_type": payload.get("benchmark_type"),
        "metrics": metrics,
        "last_metrics": train.get("last_metrics", {}),
        "runtime_status": payload.get("runtime_status"),
    }


def load_error(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path),
            "run_id": path.stem,
            "backend": path.stem,
            "status": "error",
            "error": str(exc),
        }
    return {
        "path": str(path),
        "run_id": path.stem,
        "backend": payload.get("backend", path.stem),
        "status": payload.get("status", "error"),
        "error": payload.get("error"),
        "command": payload.get("command"),
        "returncode": payload.get("returncode"),
    }


def metric(result: Dict[str, Any], name: str) -> float:
    try:
        return float(result.get("metrics", {}).get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compare(
    results: List[Dict[str, Any]],
    target: str,
    baseline: str,
    primary_metric: str,
) -> Dict[str, Any]:
    def best_result(backend: str) -> Dict[str, Any] | None:
        candidates = [
            result
            for result in results
            if result.get("status") == "ok" and str(result.get("backend")) == backend
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda result: metric(result, primary_metric))

    target_result = best_result(target)
    baseline_result = best_result(baseline)
    if target_result is None or baseline_result is None:
        return {
            "target_backend": target,
            "baseline_backend": baseline,
            "primary_metric": primary_metric,
            "status": "missing_result",
            "passed": False,
        }
    target_value = metric(target_result, primary_metric)
    baseline_value = metric(baseline_result, primary_metric)
    speedup = target_value / baseline_value if baseline_value > 0 else 0.0
    target_memory = metric(target_result, MEMORY_METRIC)
    baseline_memory = metric(baseline_result, MEMORY_METRIC)
    memory_ratio = target_memory / baseline_memory if baseline_memory > 0 else 0.0
    return {
        "target_backend": target,
        "baseline_backend": baseline,
        "primary_metric": primary_metric,
        "target_value": target_value,
        "baseline_value": baseline_value,
        "target_run_id": target_result.get("run_id"),
        "baseline_run_id": baseline_result.get("run_id"),
        "speedup": speedup,
        "target_peak_memory_bytes": target_memory,
        "baseline_peak_memory_bytes": baseline_memory,
        "memory_ratio": memory_ratio,
        "passed": speedup > 1.0,
    }


def collect(input_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"comparison.json", "report.json"}:
            continue
        if path.name.endswith(".error.json"):
            results.append(load_error(path))
            continue
        results.append(load_payload(path))
    return results


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    primary_metric = report["primary_metric"]
    lines = [
        f"# {report['title']}",
        "",
        "## Scope",
        "",
        f"- Workload: {report['workload_label']}",
        f"- Primary metric: {primary_metric}",
        "- Memory metric: CUDA peak allocated memory bytes",
        "- Result rule: the ParaScale target backend is considered faster only when measured speedup is greater than 1.0",
        f"- Target backend: {report['target_backend']}",
        "",
        "## Results",
        "",
        "| Run | Backend | Status | E2E pairs/s | Compute pairs/s | Tokens/s | Patch tokens/s | Peak memory GB | Dataloader wait ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        metrics = result.get("metrics", {})
        peak_gb = float(metrics.get(MEMORY_METRIC, 0.0) or 0.0) / 1024**3
        lines.append(
            "| {run_id} | {backend} | {status} | {pairs:.3f} | {compute:.3f} | {tokens:.3f} | {patch:.3f} | {memory:.3f} | {wait:.3f} |".format(
                run_id=result.get("run_id"),
                backend=result.get("backend"),
                status=result.get("status"),
                pairs=float(metrics.get(primary_metric, 0.0) or 0.0),
                compute=float(
                    metrics.get("compute_image_text_pairs_per_second", 0.0) or 0.0
                ),
                tokens=float(metrics.get("end_to_end_tokens_per_second", 0.0) or 0.0),
                patch=float(
                    metrics.get("end_to_end_patch_tokens_per_second", 0.0) or 0.0
                ),
                memory=peak_gb,
                wait=float(metrics.get("dataloader_wait_ms", 0.0) or 0.0),
            )
        )
    lines.extend(["", "## Comparisons", ""])
    for item in report["comparisons"]:
        lines.append(
            "- {target} ({target_run}) vs {baseline} ({baseline_run}): speedup={speedup:.4f}, memory_ratio={memory_ratio:.4f}, passed={passed}".format(
                target=item.get("target_backend"),
                baseline=item.get("baseline_backend"),
                target_run=item.get("target_run_id", "n/a"),
                baseline_run=item.get("baseline_run_id", "n/a"),
                speedup=float(item.get("speedup", 0.0) or 0.0),
                memory_ratio=float(item.get("memory_ratio", 0.0) or 0.0),
                passed=item.get("passed"),
            )
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(
    input_dir: Path,
    *,
    benchmark_id: str,
    title: str,
    workload_label: str,
    primary_metric: str,
    target_backend: str | None = None,
) -> Dict[str, Any]:
    results = collect(input_dir)
    backends = {str(result.get("backend")) for result in results}
    if target_backend is None:
        target_backend = "native_ddp" if "native_ddp" in backends else "native"
    comparisons = [
        compare(results, target_backend, "fsdp", primary_metric),
        compare(results, target_backend, "deepspeed", primary_metric),
    ]
    ok_comparisons = [
        item for item in comparisons if item.get("status") != "missing_result"
    ]
    passed = bool(ok_comparisons) and all(item.get("passed") for item in ok_comparisons)
    if passed:
        conclusion = (
            "Measured native ParaScale path is faster than all available "
            f"FSDP/DeepSpeed baselines for {workload_label}."
        )
    else:
        conclusion = "The benchmark does not prove ParaScale is faster than all FSDP/DeepSpeed baselines under the available measurements."
    return {
        "benchmark_id": benchmark_id,
        "title": title,
        "workload_label": workload_label,
        "primary_metric": primary_metric,
        "target_backend": target_backend,
        "memory_metric": MEMORY_METRIC,
        "results": results,
        "comparisons": comparisons,
        "passed": passed,
        "conclusion": conclusion,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Directory containing backend JSON files."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write comparison JSON."
    )
    parser.add_argument("--markdown", help="Optional path to write Markdown report.")
    parser.add_argument(
        "--benchmark-id",
        default="datacomp_wds_clip_contrastive_backend_matrix",
        help="Stable benchmark identifier.",
    )
    parser.add_argument(
        "--title",
        default="DataComp Backend Benchmark Report",
        help="Markdown report title.",
    )
    parser.add_argument(
        "--workload-label",
        default="DataComp WDS CLIP-style contrastive smoke",
        help="Human-readable workload label.",
    )
    parser.add_argument(
        "--primary-metric",
        default=DEFAULT_PRIMARY_METRIC,
        help="Metric used for speedup comparison.",
    )
    parser.add_argument(
        "--target-backend",
        default=None,
        help="Backend to compare against FSDP/DeepSpeed. Defaults to native_ddp when present.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(
        Path(args.input),
        benchmark_id=args.benchmark_id,
        title=args.title,
        workload_label=args.workload_label,
        primary_metric=args.primary_metric,
        target_backend=args.target_backend,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.markdown:
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
