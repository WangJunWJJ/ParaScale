# -*- coding: utf-8 -*-
# @Time : 2026/7/24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Summarize A6000 native-DDP precision, hook, and dataloader sweeps."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.benchmarks.tools.common import (  # noqa: E402
    IMAGE_TEXT_THROUGHPUT_KEYS,
    first_metric,
    loss_value,
    read_json_payload,
    train_section,
)

RUN_RE = re.compile(
    r"(?P<group>scale|hook|data)_"
    r"(?P<gpus>\d+)gpu_"
    r"(?P<precision>fp32|fp16|bf16)"
    r"(?:_(?P<hook>none|fp16_compress|bf16_compress))?"
    r"(?:_bucket(?P<bucket>\d+))?"
    r"(?:_cuda(?P<topology>\d+))?"
    r"_b(?P<batch>\d+)_w(?P<workers>\d+)"
)

BUCKET_RE = re.compile(
    r"bucket_(?P<gpus>\d+)gpu_(?P<precision>fp32|fp16|bf16)_"
    r"(?P<hook>none|fp16_compress|bf16_compress)_bucket(?P<bucket>\d+)_"
    r"b(?P<batch>\d+)_w(?P<workers>\d+)"
)

TOPOLOGY_RE = re.compile(
    r"topo_(?P<gpus>\d+)gpu_(?P<precision>fp32|fp16|bf16)_"
    r"(?P<hook>none|fp16_compress|bf16_compress)_bucket(?P<bucket>\d+)_"
    r"cuda(?P<topology>\d+)_b(?P<batch>\d+)_w(?P<workers>\d+)"
)


def _parse_run_id(path: Path) -> Dict[str, Any]:
    run_id = path.stem
    if run_id.endswith(".error"):
        run_id = run_id[: -len(".error")]
    group = "unknown"
    match = RUN_RE.match(run_id)
    if match:
        group = match.group("group")
    else:
        match = BUCKET_RE.match(run_id)
        if match:
            group = "bucket"
        else:
            match = TOPOLOGY_RE.match(run_id)
            if match:
                group = "topo"
    if not match:
        return {
            "run_id": run_id,
            "group": "unknown",
            "gpus": 0,
            "precision": "n/a",
            "ddp_comm_hook": "n/a",
            "ddp_bucket_cap_mb": None,
            "visible_devices": "",
            "batch_per_gpu": 0,
            "num_workers": 0,
        }
    hook = match.group("hook")
    bucket = match.groupdict().get("bucket")
    topology = match.groupdict().get("topology") or ""
    return {
        "run_id": run_id,
        "group": group,
        "gpus": int(match.group("gpus")),
        "precision": match.group("precision"),
        "ddp_comm_hook": hook or "none",
        "ddp_bucket_cap_mb": int(bucket) if bucket else None,
        "visible_devices": ",".join(topology) if topology else "",
        "batch_per_gpu": int(match.group("batch")),
        "num_workers": int(match.group("workers")),
    }


def _record(path: Path) -> Dict[str, Any]:
    payload = read_json_payload(path)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    train = train_section(payload)
    parsed = _parse_run_id(path)
    throughput, throughput_metric = first_metric(metrics, IMAGE_TEXT_THROUGHPUT_KEYS)
    peak_memory, _ = first_metric(
        metrics,
        ("stable_peak_memory_bytes", "peak_memory_bytes"),
    )
    dataloader_wait, _ = first_metric(
        metrics,
        ("stable_dataloader_wait_ms", "dataloader_wait_ms"),
    )
    step_time, _ = first_metric(
        metrics,
        ("stable_step_time_ms", "step_time_ms"),
    )
    is_error = path.name.endswith(".error.json") or payload.get("status") == "error"
    parsed.update(
        {
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
            "step_time_ms": step_time,
            "path": str(path),
            "returncode": payload.get("returncode"),
            "error": payload.get("error"),
        }
    )
    return parsed


def _successful(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("ok")]


def _find(records: Iterable[Dict[str, Any]], **criteria: Any) -> Dict[str, Any] | None:
    for record in records:
        if all(record.get(key) == value for key, value in criteria.items()):
            return record
    return None


def _scaling(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ok = _successful(records)
    for precision in ("fp32", "fp16", "bf16"):
        one = _find(ok, group="scale", gpus=1, precision=precision, ddp_comm_hook="none")
        two = _find(ok, group="scale", gpus=2, precision=precision, ddp_comm_hook="none")
        four = _find(ok, group="scale", gpus=4, precision=precision, ddp_comm_hook="none")
        one_t = float(one.get("throughput") or 0.0) if one else 0.0
        two_t = float(two.get("throughput") or 0.0) if two else 0.0
        four_t = float(four.get("throughput") or 0.0) if four else 0.0
        rows.append(
            {
                "precision": precision,
                "one_gpu_throughput": one_t,
                "two_gpu_throughput": two_t,
                "four_gpu_throughput": four_t,
                "scale_1_to_2": two_t / one_t if one_t > 0 else 0.0,
                "scale_2_to_4": four_t / two_t if two_t > 0 else 0.0,
                "scale_1_to_4": four_t / one_t if one_t > 0 else 0.0,
                "efficiency_1_to_4": four_t / (one_t * 4.0) if one_t > 0 else 0.0,
            }
        )
    return rows


def _hook_comparisons(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ok = _successful(records)
    for gpus, precision, hook in (
        (2, "bf16", "bf16_compress"),
        (4, "bf16", "bf16_compress"),
        (2, "fp16", "fp16_compress"),
        (4, "fp16", "fp16_compress"),
    ):
        baseline = _find(ok, group="scale", gpus=gpus, precision=precision, ddp_comm_hook="none")
        tuned = _find(ok, group="hook", gpus=gpus, precision=precision, ddp_comm_hook=hook)
        base_t = float(baseline.get("throughput") or 0.0) if baseline else 0.0
        tuned_t = float(tuned.get("throughput") or 0.0) if tuned else 0.0
        rows.append(
            {
                "gpus": gpus,
                "precision": precision,
                "hook": hook,
                "baseline_throughput": base_t,
                "hook_throughput": tuned_t,
                "relative_to_none": tuned_t / base_t if base_t > 0 else 0.0,
            }
        )
    return rows


def _bucket_comparisons(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ok = _successful(records)
    baseline = _find(
        ok,
        group="hook",
        gpus=4,
        precision="bf16",
        ddp_comm_hook="bf16_compress",
    )
    baseline_t = float(baseline.get("throughput") or 0.0) if baseline else 0.0
    for record in sorted(
        (
            item
            for item in ok
            if item.get("group") == "bucket"
            and item.get("precision") == "bf16"
            and item.get("ddp_comm_hook") == "bf16_compress"
        ),
        key=lambda item: int(item.get("ddp_bucket_cap_mb") or 0),
    ):
        throughput = float(record.get("throughput") or 0.0)
        rows.append(
            {
                "bucket_cap_mb": record.get("ddp_bucket_cap_mb"),
                "throughput": throughput,
                "baseline_throughput": baseline_t,
                "relative_to_default": (
                    throughput / baseline_t if baseline_t > 0 else 0.0
                ),
                "dataloader_wait_ms": record.get("dataloader_wait_ms"),
            }
        )
    return rows


def _topology_comparisons(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ok = _successful(records)
    for record in sorted(
        (item for item in ok if item.get("group") == "topo"),
        key=lambda item: str(item.get("visible_devices") or ""),
    ):
        rows.append(
            {
                "visible_devices": record.get("visible_devices"),
                "bucket_cap_mb": record.get("ddp_bucket_cap_mb"),
                "throughput": record.get("throughput"),
                "dataloader_wait_ms": record.get("dataloader_wait_ms"),
                "peak_memory_bytes": record.get("peak_memory_bytes"),
            }
        )
    return rows


def _best_data_loader(records: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    data_records = [
        record
        for record in _successful(records)
        if record.get("group") == "data" and record.get("throughput")
    ]
    if not data_records:
        return None
    return max(data_records, key=lambda item: float(item.get("throughput") or 0.0))


def build_report(
    input_dir: Path,
    *,
    hardware: str,
    image: str,
    dataset: str,
    model: str,
    steps: int,
    warmup_steps: int,
    batch_per_gpu: int,
) -> Dict[str, Any]:
    records = [
        _record(path)
        for path in sorted(input_dir.glob("*.json"))
        if path.name != "summary.json"
    ]
    return {
        "mode": "a6000_native_ddp_scaling_summary",
        "suite_id": "a6000_native_ddp_scaling",
        "hardware": hardware,
        "image": image,
        "dataset": dataset,
        "model": model,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "batch_per_gpu": batch_per_gpu,
        "passed": bool(records) and any(record.get("ok") for record in records),
        "runs": records,
        "scaling": _scaling(records),
        "hook_comparisons": _hook_comparisons(records),
        "bucket_comparisons": _bucket_comparisons(records),
        "topology_comparisons": _topology_comparisons(records),
        "best_dataloader": _best_data_loader(records),
        "notes": [
            "Bucket sweep uses bf16_compress on 4 visible GPUs.",
            "Topology sweep constrains CUDA_VISIBLE_DEVICES before torchrun.",
        ],
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# A6000 Native-DDP Scaling",
        "",
        f"- Hardware: `{report['hardware']}`",
        f"- Image: `{report['image']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Model: `{report['model']}`",
        f"- Steps: {report['steps']}",
        f"- Warmup steps: {report['warmup_steps']}",
        f"- Batch per GPU: {report['batch_per_gpu']}",
        "",
        "## Scaling",
        "",
        "| Precision | 1 GPU | 2 GPU | 4 GPU | 1->2 | 2->4 | 1->4 | 4 GPU efficiency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["scaling"]:
        lines.append(
            "| {precision} | {one:.3f} | {two:.3f} | {four:.3f} | {s12:.3f}x | {s24:.3f}x | {s14:.3f}x | {eff:.3f} |".format(
                precision=item["precision"],
                one=float(item["one_gpu_throughput"] or 0.0),
                two=float(item["two_gpu_throughput"] or 0.0),
                four=float(item["four_gpu_throughput"] or 0.0),
                s12=float(item["scale_1_to_2"] or 0.0),
                s24=float(item["scale_2_to_4"] or 0.0),
                s14=float(item["scale_1_to_4"] or 0.0),
                eff=float(item["efficiency_1_to_4"] or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Communication Hooks",
            "",
            "| GPUs | Precision | Hook | Baseline pairs/s | Hook pairs/s | Relative |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in report["hook_comparisons"]:
        lines.append(
            "| {gpus} | {precision} | {hook} | {base:.3f} | {hook_t:.3f} | {relative:.3f}x |".format(
                gpus=item["gpus"],
                precision=item["precision"],
                hook=item["hook"],
                base=float(item["baseline_throughput"] or 0.0),
                hook_t=float(item["hook_throughput"] or 0.0),
                relative=float(item["relative_to_none"] or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Bucket Cap",
            "",
            "| Bucket cap MB | Throughput | Relative to default | Dataloader wait ms |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["bucket_comparisons"]:
        lines.append(
            "| {bucket} | {throughput:.3f} | {relative:.3f}x | {wait:.3f} |".format(
                bucket=item.get("bucket_cap_mb") or "default",
                throughput=float(item.get("throughput") or 0.0),
                relative=float(item.get("relative_to_default") or 0.0),
                wait=float(item.get("dataloader_wait_ms") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Topology",
            "",
            "| CUDA_VISIBLE_DEVICES | Bucket cap MB | Throughput | Dataloader wait ms | Peak GB |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["topology_comparisons"]:
        lines.append(
            "| `{visible}` | {bucket} | {throughput:.3f} | {wait:.3f} | {memory:.3f} |".format(
                visible=item.get("visible_devices") or "all",
                bucket=item.get("bucket_cap_mb") or "default",
                throughput=float(item.get("throughput") or 0.0),
                wait=float(item.get("dataloader_wait_ms") or 0.0),
                memory=float(item.get("peak_memory_bytes") or 0.0) / 1024**3,
            )
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Group | GPUs | Precision | Hook | Bucket MB | Visible devices | Workers | OK | Throughput | Loss | Peak GB | Wait ms |",
            "| --- | --- | ---: | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in report["runs"]:
        lines.append(
            "| {run_id} | {group} | {gpus} | {precision} | {hook} | {bucket} | {visible} | {workers} | {ok} | {throughput:.3f} | {loss} | {memory:.3f} | {wait:.3f} |".format(
                run_id=run["run_id"],
                group=run["group"],
                gpus=run["gpus"],
                precision=run["precision"],
                hook=run["ddp_comm_hook"],
                bucket=run.get("ddp_bucket_cap_mb") or 0,
                visible=run.get("visible_devices") or "all",
                workers=run["num_workers"],
                ok=run["ok"],
                throughput=float(run.get("throughput") or 0.0),
                loss=(f"{float(run['loss']):.6f}" if run.get("loss") is not None else "n/a"),
                memory=float(run.get("peak_memory_bytes") or 0.0) / 1024**3,
                wait=float(run.get("dataloader_wait_ms") or 0.0),
            )
        )
    if report.get("best_dataloader"):
        best = report["best_dataloader"]
        lines.extend(
            [
                "",
                "## Best Dataloader Candidate",
                "",
                "- Run: `{run_id}`",
                "- Throughput: {throughput:.3f}",
                "- Dataloader wait ms: {wait:.3f}",
            ]
        )
        lines = [
            line.format(
                run_id=best["run_id"],
                throughput=float(best.get("throughput") or 0.0),
                wait=float(best.get("dataloader_wait_ms") or 0.0),
            )
            if "{run_id}" in line or "{throughput" in line or "{wait" in line
            else line
            for line in lines
        ]
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="clip_medium")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--batch-per-gpu", type=int, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(
        Path(args.input),
        hardware=args.hardware,
        image=args.image,
        dataset=args.dataset,
        model=args.model,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_per_gpu=args.batch_per_gpu,
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
