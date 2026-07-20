# -*- coding: utf-8 -*-
# @Time : 2026/7/20 下午1:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Summarize ParaScale Ascend validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tests.benchmarks.tools.common import (
    SAMPLE_THROUGHPUT_KEYS,
    loss_value,
    merged_train_metrics,
    metric_value,
    read_json_payload,
    run_id_from_path,
    train_section,
)


def _record(path: Path) -> Dict[str, Any]:
    payload = read_json_payload(path)
    train = train_section(payload)
    metrics = merged_train_metrics(payload)
    is_error = path.name.endswith(".error.json") or payload.get("status") == "error"
    return {
        "run_id": run_id_from_path(path),
        "ok": not is_error
        and payload.get("ok", True) is not False
        and payload.get("diagnostics", {}).get("ok", True) is not False,
        "status": payload.get("status", "error" if is_error else "ok"),
        "mode": payload.get("mode"),
        "runtime_status": payload.get("runtime_status"),
        "backend": train.get("backend") or payload.get("backend"),
        "global_step": train.get("global_step", payload.get("global_step")),
        "throughput": metric_value(metrics, SAMPLE_THROUGHPUT_KEYS),
        "loss": loss_value(payload, include_top_level=True),
        "npu_available": payload.get("ascend_runtime", {}).get("available"),
        "npu_device_count": payload.get("ascend_runtime", {}).get("device_count"),
        "torch_npu": payload.get("dependencies", {}).get("torch_npu"),
        "returncode": payload.get("returncode"),
        "path": str(path),
    }


def _collect(input_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"summary.json"}:
            continue
        runs.append(_record(path))
    return runs


def build_report(
    input_dir: Path,
    *,
    suite_id: str,
    hardware: str,
    image: str,
    blocked_reason: str | None = None,
) -> Dict[str, Any]:
    runs = _collect(input_dir)
    required = {"doctor", "tiny_single", "tiny_hccl"}
    present = {run["run_id"] for run in runs}
    required_runs = [run for run in runs if run["run_id"] in required]
    missing = sorted(required - present)
    passed = bool(required_runs) and not missing and all(
        run["ok"] for run in required_runs
    )
    if blocked_reason:
        passed = False
    return {
        "mode": "ascend_validation_summary",
        "suite_id": suite_id,
        "hardware": hardware,
        "image": image,
        "input_dir": str(input_dir),
        "passed": passed,
        "blocked": bool(blocked_reason),
        "blocked_reason": blocked_reason,
        "missing": missing,
        "runs": runs,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Ascend Functional Validation",
        "",
        f"- Suite: `{report['suite_id']}`",
        f"- Hardware: {report['hardware']}",
        f"- Image: `{report['image']}`",
        f"- Passed: {report['passed']}",
        f"- Blocked: {report['blocked']}",
        f"- Input directory: `{report['input_dir']}`",
    ]
    if report.get("blocked_reason"):
        lines.append(f"- Blocked reason: {report['blocked_reason']}")
    if report.get("missing"):
        lines.append(f"- Missing required runs: `{', '.join(report['missing'])}`")
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | OK | Status | Mode | Backend | Step | Throughput | Loss | torch_npu | NPU count | Return code |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for run in report["runs"]:
        lines.append(
            "| {run_id} | {ok} | {status} | {mode} | {backend} | {step} | {throughput:.3f} | {loss} | {torch_npu} | {npu_count} | {returncode} |".format(
                run_id=run["run_id"],
                ok=run["ok"],
                status=run.get("status") or "n/a",
                mode=run.get("mode") or "n/a",
                backend=run.get("backend") or "n/a",
                step=run.get("global_step") or 0,
                throughput=float(run.get("throughput") or 0.0),
                loss=(
                    f"{float(run['loss']):.6f}"
                    if run.get("loss") is not None
                    else "n/a"
                ),
                torch_npu=run.get("torch_npu"),
                npu_count=run.get("npu_device_count") or 0,
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
    parser.add_argument("--blocked-reason")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(
        Path(args.input_dir),
        suite_id=args.suite_id,
        hardware=args.hardware,
        image=args.image,
        blocked_reason=args.blocked_reason,
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
