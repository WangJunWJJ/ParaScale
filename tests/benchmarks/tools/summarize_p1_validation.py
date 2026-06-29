# -*- coding: utf-8 -*-
# @Time : 2026/6/12 下午3:55
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Summarize P1 functional validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "ok": False, "error": "missing file"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "ok": False, "error": str(exc)}
    payload.setdefault("path", str(path))
    return payload


def _smoke_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    steps = payload.get("steps", {})
    required = ["doctor", "plan", "train", "checkpoint_validate", "resume", "serve"]
    missing = [name for name in required if name not in steps]
    failed = [
        name
        for name in required
        if name in steps and not bool(steps.get(name, {}).get("ok", False))
    ]
    return {
        "name": "server_tiny_smoke",
        "kind": "smoke",
        "ok": not missing and not failed,
        "missing": missing,
        "failed": failed,
        "path": payload.get("path"),
    }


def _benchmark_status(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    validation = payload.get("validation", {})
    checkpoint = validation.get("checkpoint", {})
    resume = validation.get("resume", {})
    metrics = payload.get("metrics", {})
    return {
        "name": name,
        "kind": "benchmark",
        "ok": bool(checkpoint.get("ok")) and bool(resume.get("ok")),
        "backend": payload.get("train", {}).get("backend"),
        "global_step": payload.get("train", {}).get("global_step"),
        "checkpoint_ok": checkpoint.get("ok"),
        "resume_ok": resume.get("ok"),
        "backend_state_loaded": resume.get("backend_state_loaded"),
        "samples_per_second": metrics.get("samples_per_second"),
        "images_per_second": metrics.get("end_to_end_images_per_second"),
        "pairs_per_second": metrics.get("end_to_end_image_text_pairs_per_second"),
        "peak_memory_bytes": metrics.get("peak_memory_bytes"),
        "path": payload.get("path"),
        "error": payload.get("error"),
    }


def build_report(input_dir: Path) -> Dict[str, Any]:
    smoke = _read_json(input_dir / "server_tiny_smoke.json")
    clip = _read_json(input_dir / "clip_native_ddp_resume_benchmark.json")
    yolo = _read_json(input_dir / "yolo_native_ddp_resume_benchmark.json")
    checks = [
        _smoke_status(smoke),
        _benchmark_status("clip_native_ddp_resume", clip),
        _benchmark_status("yolo_native_ddp_resume", yolo),
    ]
    return {
        "mode": "p1_validation_summary",
        "input_dir": str(input_dir),
        "passed": all(item.get("ok") for item in checks),
        "checks": checks,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = [
        "# P1 Functional Validation Report",
        "",
        f"- Passed: {report['passed']}",
        f"- Input directory: `{report['input_dir']}`",
        "",
        "## Checks",
        "",
        "| Check | Kind | OK | Backend | Checkpoint | Resume | Throughput | Peak memory GB |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in report["checks"]:
        throughput = (
            item.get("pairs_per_second")
            or item.get("images_per_second")
            or item.get("samples_per_second")
            or 0.0
        )
        memory_gb = float(item.get("peak_memory_bytes") or 0.0) / 1024**3
        lines.append(
            "| {name} | {kind} | {ok} | {backend} | {checkpoint} | {resume} | {throughput:.3f} | {memory:.3f} |".format(
                name=item.get("name"),
                kind=item.get("kind"),
                ok=item.get("ok"),
                backend=item.get("backend", "n/a"),
                checkpoint=item.get("checkpoint_ok", "n/a"),
                resume=item.get("resume_ok", "n/a"),
                throughput=float(throughput or 0.0),
                memory=memory_gb,
            )
        )
    lines.extend(["", "## Notes", ""])
    if report["passed"]:
        lines.append(
            "P1 train/checkpoint/validate/resume/serve and native-DDP workload validation passed."
        )
    else:
        lines.append("One or more P1 validation checks failed; inspect the JSON files.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(Path(args.input_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(report, Path(args.markdown))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
