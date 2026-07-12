# -*- coding: utf-8 -*-

"""Compare ParaScale CLIP benchmark results with direct PyTorch baselines."""

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
    "image_text_pairs_per_second",
    "images_per_second",
)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = str(path)
    return payload


def _metric(metrics: Dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        try:
            value = float(metrics.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 0.0


def _metric_name(metrics: Dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        try:
            if float(metrics.get(key, 0.0) or 0.0) > 0:
                return key
        except (TypeError, ValueError):
            continue
    return None


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


def _row(path: Path, label: str, stack: str) -> Dict[str, Any]:
    payload = _read_json(path)
    metrics = payload.get("metrics", {})
    train = payload.get("train", {})
    if not isinstance(train, dict):
        train = {}
    backend = train.get("backend") or payload.get("backend") or label
    throughput = _metric(metrics, THROUGHPUT_KEYS)
    return {
        "label": label,
        "stack": stack,
        "backend": str(backend),
        "throughput": throughput,
        "throughput_metric": _metric_name(metrics, THROUGHPUT_KEYS),
        "loss": _loss(payload),
        "global_step": train.get("global_step", payload.get("global_step")),
        "peak_memory_bytes": _metric(metrics, ("peak_memory_bytes",)),
        "dataloader_wait_ms": _metric(metrics, ("dataloader_wait_ms",)),
        "path": str(path),
    }


def build_report(
    input_dir: Path,
    *,
    suite_id: str,
    hardware: str,
    image: str,
) -> Dict[str, Any]:
    mapping = [
        ("parascale_native_ddp", "ParaScale", input_dir / "parascale_native_ddp.json"),
        ("parascale_fsdp", "ParaScale", input_dir / "parascale_fsdp.json"),
        ("torch_ddp", "Direct PyTorch", input_dir / "torch_ddp.json"),
        ("torch_fsdp", "Direct PyTorch", input_dir / "torch_fsdp.json"),
    ]
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for label, stack, path in mapping:
        if path.exists():
            rows.append(_row(path, label, stack))
        else:
            missing.append(str(path))
    by_label = {row["label"]: row for row in rows}
    comparisons = []
    for direct, parascale in (
        ("torch_ddp", "parascale_native_ddp"),
        ("torch_fsdp", "parascale_fsdp"),
    ):
        direct_row = by_label.get(direct)
        parascale_row = by_label.get(parascale)
        if not direct_row or not parascale_row:
            continue
        direct_t = float(direct_row.get("throughput") or 0.0)
        parascale_t = float(parascale_row.get("throughput") or 0.0)
        comparisons.append(
            {
                "direct": direct,
                "parascale": parascale,
                "direct_throughput": direct_t,
                "parascale_throughput": parascale_t,
                "parascale_vs_direct": (
                    parascale_t / direct_t if direct_t > 0 else None
                ),
            }
        )
    return {
        "mode": "direct_pytorch_comparison",
        "suite_id": suite_id,
        "hardware": hardware,
        "image": image,
        "input_dir": str(input_dir),
        "passed": not missing and all(row["throughput"] > 0 for row in rows),
        "missing": missing,
        "runs": rows,
        "comparisons": comparisons,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Direct PyTorch vs ParaScale CLIP Comparison",
        "",
        f"- Suite: `{report['suite_id']}`",
        f"- Hardware: {report['hardware']}",
        f"- Image: `{report['image']}`",
        f"- Passed: {report['passed']}",
        f"- Input directory: `{report['input_dir']}`",
        "",
        "## Runs",
        "",
        "| Label | Stack | Backend | Step | Throughput | Metric | Loss | Peak memory GB | Dataloader wait ms |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in report["runs"]:
        lines.append(
            "| {label} | {stack} | {backend} | {step} | {throughput:.3f} | {metric} | {loss} | {memory:.3f} | {wait:.3f} |".format(
                label=row["label"],
                stack=row["stack"],
                backend=row["backend"],
                step=row.get("global_step") or 0,
                throughput=float(row.get("throughput") or 0.0),
                metric=row.get("throughput_metric") or "n/a",
                loss=(
                    f"{float(row['loss']):.6f}"
                    if row.get("loss") is not None
                    else "n/a"
                ),
                memory=float(row.get("peak_memory_bytes") or 0.0) / 1024**3,
                wait=float(row.get("dataloader_wait_ms") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| ParaScale | Direct PyTorch | ParaScale throughput | Direct throughput | Ratio |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in report["comparisons"]:
        ratio = item.get("parascale_vs_direct")
        lines.append(
            "| {parascale} | {direct} | {parascale_t:.3f} | {direct_t:.3f} | {ratio} |".format(
                parascale=item["parascale"],
                direct=item["direct"],
                parascale_t=float(item.get("parascale_throughput") or 0.0),
                direct_t=float(item.get("direct_throughput") or 0.0),
                ratio=f"{float(ratio):.4f}x" if ratio is not None else "n/a",
            )
        )
    if report.get("missing"):
        lines.extend(["", "## Missing", ""])
        lines.extend(f"- `{item}`" for item in report["missing"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--hardware", required=True)
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
