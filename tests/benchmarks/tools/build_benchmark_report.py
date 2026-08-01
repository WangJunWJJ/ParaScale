# -*- coding: utf-8 -*-
# @Time : 2026/7/20 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Build the unified benchmark report from compact summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

SUMMARY_FILES = {
    "dual_4090": Path("dual_4090_full_validation/summary.json"),
    "direct_pytorch": Path("direct_pytorch_clip_comparison/summary.json"),
    "ascend_validation": Path("ascend_validation/summary.json"),
    "ascend_matrix": Path("ascend_parallel_matrix/summary.json"),
    "cross_hardware": Path("cross_hardware_clip_datacomp/summary.json"),
    "rtx4090_precision": Path("rtx4090_clip_precision_datacomp/summary.json"),
    "a6000_native_ddp_scaling": Path("a6000_native_ddp_scaling/summary.json"),
}

CONFIG_FILES = {
    "cross_hardware": (
        Path(
            "cross_hardware_clip_datacomp/ascend/ascend_clip_datacomp_native_ddp_fp32.config.json"
        ),
        Path(
            "cross_hardware_clip_datacomp/rtx4090/rtx4090_clip_datacomp_native_ddp_fp32.config.json"
        ),
    ),
    "rtx4090_precision": (
        Path(
            "rtx4090_clip_precision_datacomp/fp16/rtx4090_clip_datacomp_native_ddp_fp16.config.json"
        ),
    ),
}


def _read_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "missing": True,
            "path": str(path),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_summary_path"] = str(path)
    return payload


def load_summaries(report_root: Path) -> Dict[str, Dict[str, Any]]:
    """Load every known benchmark summary from the report root."""

    return {
        name: _read_summary(report_root / relative_path)
        for name, relative_path in SUMMARY_FILES.items()
    }


def build_report_markdown(
    summaries: Dict[str, Dict[str, Any]],
    *,
    report_root: Path,
) -> str:
    """Render the unified benchmark report markdown."""

    lines: List[str] = [
        "# ParaScale Benchmark Report",
        "",
        "This report is the review entrypoint for ParaScale benchmark evidence. "
        "Each section is generated from compact `summary.json` artifacts; config "
        "snapshots remain linked where they are needed for auditability.",
        "",
        "## How to Update",
        "",
        "1. Run or refresh a benchmark suite so its `summary.json` is current.",
        "2. Regenerate this file:",
        "",
        "```bash",
        "python tests/benchmarks/tools/build_benchmark_report.py "
        "--report-root tests/benchmarks/reports "
        "--output tests/benchmarks/reports/BENCHMARK_REPORT.md",
        "```",
        "",
        "3. Commit the updated summary artifacts and this report together.",
        "",
    ]
    lines.extend(_overview_table(summaries))
    lines.extend(_evidence_quality_section(summaries))
    lines.extend(_dual_4090_section(summaries["dual_4090"]))
    lines.extend(_direct_pytorch_section(summaries["direct_pytorch"]))
    lines.extend(_ascend_validation_section(summaries["ascend_validation"]))
    lines.extend(_ascend_matrix_section(summaries["ascend_matrix"]))
    lines.extend(_cross_hardware_section(summaries["cross_hardware"]))
    lines.extend(_precision_section(summaries["rtx4090_precision"]))
    lines.extend(_a6000_native_ddp_scaling_section(summaries["a6000_native_ddp_scaling"]))
    lines.extend(_evidence_links(report_root, summaries))
    return "\n".join(lines).rstrip() + "\n"


def _overview_table(summaries: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = [
        "## Overview",
        "",
        "| Suite | Status | Hardware | Image | Summary |",
        "| --- | --- | --- | --- | --- |",
    ]
    labels = {
        "dual_4090": "Dual 4090 full validation",
        "direct_pytorch": "Direct PyTorch/DeepSpeed comparison",
        "ascend_validation": "Ascend functional validation",
        "ascend_matrix": "Ascend parallel matrix",
        "cross_hardware": "Cross-hardware CLIP DataComp",
        "rtx4090_precision": "RTX 4090 precision comparison",
        "a6000_native_ddp_scaling": "A6000 native-DDP scaling",
    }
    for name, label in labels.items():
        summary = summaries[name]
        status = _status(summary)
        hardware = summary.get("hardware") or _hardware_summary(summary)
        image = summary.get("image", "n/a")
        summary_link = _posix_path(SUMMARY_FILES[name])
        lines.append(
            f"| {label} | {status} | {hardware} | `{image}` | [{summary_link}]({summary_link}) |"
        )
    lines.append("")
    return lines


def _evidence_quality_section(summaries: Dict[str, Dict[str, Any]]) -> List[str]:
    evidence_rows = list(_evidence_quality_rows(summaries))
    if not evidence_rows:
        return []
    lines = [
        "## Evidence Quality",
        "",
        "| Suite | Run | Runtime status | Capability level | Warmup/measured |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in evidence_rows:
        lines.append(
            "| {suite} | {run} | {runtime} | {capability} | {window} |".format(
                suite=row["suite"],
                run=row["run"],
                runtime=row["runtime"],
                capability=row["capability"],
                window=row["window"],
            )
        )
    lines.append("")
    return lines


def _evidence_quality_rows(
    summaries: Dict[str, Dict[str, Any]]
) -> Iterable[Dict[str, str]]:
    for suite, summary in summaries.items():
        runs = summary.get("runs", [])
        if not isinstance(runs, list):
            continue
        for index, run in enumerate(runs, start=1):
            if not isinstance(run, dict):
                continue
            capability = run.get("capability_level")
            runtime = run.get("runtime_status")
            window = _measurement_window(run.get("measurement_window"))
            if not any((capability, runtime, window != "n/a")):
                continue
            yield {
                "suite": suite,
                "run": str(
                    run.get("run_id")
                    or run.get("label")
                    or run.get("scenario")
                    or index
                ),
                "runtime": str(runtime or "n/a"),
                "capability": str(capability or "n/a"),
                "window": window,
            }


def _dual_4090_section(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "## Dual RTX 4090 Validation",
        "",
        "| Model | OK runs | Total runs | Best backend | Throughput | Loss | Peak memory GB |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in summary.get("summaries", []):
        lines.append(
            "| {model} | {ok_runs} | {total_runs} | {backend} | {throughput} | {loss} | {memory} |".format(
                model=item.get("model", "n/a"),
                ok_runs=item.get("ok_runs", 0),
                total_runs=item.get("total_runs", 0),
                backend=item.get("best_backend") or "n/a",
                throughput=_number(item.get("best_throughput")),
                loss=_number(item.get("best_loss"), digits=6),
                memory=_bytes_to_gb(item.get("best_peak_memory_bytes")),
            )
        )
    lines.append("")
    return lines


def _direct_pytorch_section(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "## Direct Baseline Comparison",
        "",
        "| ParaScale backend | Direct baseline | ParaScale throughput | Direct throughput | Ratio |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in summary.get("comparisons", []):
        lines.append(
            "| {parascale} | {direct} | {pt} | {dt} | {ratio} |".format(
                parascale=item.get("parascale", "n/a"),
                direct=item.get("direct", "n/a"),
                pt=_number(item.get("parascale_throughput")),
                dt=_number(item.get("direct_throughput")),
                ratio=_ratio(item.get("parascale_vs_direct")),
            )
        )
    lines.extend(
        [
            "",
            "| DeepSpeed backend | Baseline | DeepSpeed throughput | Baseline throughput | Ratio |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in summary.get("deepspeed_comparisons", []):
        lines.append(
            "| {deepspeed} | {baseline} | {dt} | {bt} | {ratio} |".format(
                deepspeed=item.get("deepspeed", "n/a"),
                baseline=item.get("baseline", "n/a"),
                dt=_number(item.get("deepspeed_throughput")),
                bt=_number(item.get("baseline_throughput")),
                ratio=_ratio(item.get("deepspeed_vs_baseline")),
            )
        )
    lines.append("")
    return lines


def _ascend_validation_section(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "## Ascend Functional Validation",
        "",
        "| Run | OK | Mode | Backend | Step | Throughput | Loss | torch_npu | NPU count |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for run in summary.get("runs", []):
        lines.append(
            "| {run_id} | {ok} | {mode} | {backend} | {step} | {throughput} | {loss} | {torch_npu} | {npu_count} |".format(
                run_id=run.get("run_id", "n/a"),
                ok=run.get("ok"),
                mode=run.get("mode") or "n/a",
                backend=run.get("backend") or "n/a",
                step=run.get("global_step") or 0,
                throughput=_number(run.get("throughput")),
                loss=_number(run.get("loss"), digits=6),
                torch_npu=run.get("torch_npu"),
                npu_count=run.get("npu_device_count") or 0,
            )
        )
    lines.append("")
    return lines


def _ascend_matrix_section(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "## Ascend Parallel Matrix",
        "",
        "| Scenario | Containers | NPUs | OK | Aggregate pairs/s | Pairs/s/NPU | Loss | Peak memory GB max | Dataloader wait ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary.get("scenarios", []):
        lines.append(
            "| {scenario} | {containers} | {cards} | {ok} | {throughput} | {per_card} | {loss} | {memory} | {wait} |".format(
                scenario=item.get("scenario", "n/a"),
                containers=item.get("containers", 0),
                cards=item.get("cards", 0),
                ok=item.get("ok"),
                throughput=_number(item.get("aggregate_throughput")),
                per_card=_number(item.get("throughput_per_card")),
                loss=_number(item.get("mean_loss"), digits=6),
                memory=_bytes_to_gb(item.get("peak_memory_bytes_max")),
                wait=_number(item.get("mean_dataloader_wait_ms")),
            )
        )
    lines.append("")
    return lines


def _cross_hardware_section(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "## Cross-Hardware CLIP DataComp",
        "",
        f"- Dataset: `{summary.get('dataset', 'n/a')}`",
        f"- Model: `{summary.get('model', 'n/a')}`",
        f"- Precision: `{summary.get('precision', 'n/a')}`",
        "",
        "| Label | Hardware | Backend | Throughput | Relative to baseline | Loss | Peak memory GB | Dataloader wait ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    relatives = {
        item.get("label"): item.get("relative_to_baseline")
        for item in summary.get("comparisons", [])
    }
    for run in summary.get("runs", []):
        lines.append(
            "| {label} | {hardware} | {backend} | {throughput} | {relative} | {loss} | {memory} | {wait} |".format(
                label=run.get("label", "n/a"),
                hardware=run.get("hardware", "n/a"),
                backend=run.get("backend", "n/a"),
                throughput=_number(run.get("throughput")),
                relative=_ratio(relatives.get(run.get("label"))),
                loss=_number(run.get("loss"), digits=6),
                memory=_bytes_to_gb(run.get("peak_memory_bytes")),
                wait=_number(run.get("dataloader_wait_ms")),
            )
        )
    lines.append("")
    return lines


def _precision_section(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "## RTX 4090 CLIP Precision",
        "",
        "| Precision | Backend | Throughput | Relative to FP32 | Step time ms | Loss | Peak memory GB | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in summary.get("runs", []):
        lines.append(
            "| {precision} | {backend} | {throughput} | {relative} | {step_time} | {loss} | {memory} | {note} |".format(
                precision=run.get("precision", "n/a"),
                backend=run.get("backend", "n/a"),
                throughput=_number(run.get("throughput")),
                relative=_ratio(run.get("relative_to_fp32")),
                step_time=_number(run.get("step_time_ms")),
                loss=_number(run.get("loss"), digits=6),
                memory=_bytes_to_gb(run.get("peak_memory_bytes")),
                note=run.get("note", ""),
            )
        )
    lines.append("")
    return lines


def _a6000_native_ddp_scaling_section(summary: Dict[str, Any]) -> List[str]:
    if summary.get("missing"):
        return [
            "## A6000 Native-DDP Scaling",
            "",
            "No A6000 native-DDP scaling summary has been recorded yet.",
            "",
        ]
    lines = [
        "## A6000 Native-DDP Scaling",
        "",
        f"- Dataset: `{summary.get('dataset', 'n/a')}`",
        f"- Model: `{summary.get('model', 'n/a')}`",
        f"- Steps: {summary.get('steps', 'n/a')}",
        f"- Warmup steps: {summary.get('warmup_steps', 'n/a')}",
        "",
        "| Precision | 1 GPU | 2 GPU | 4 GPU | 1->2 | 2->4 | 1->4 | 4 GPU efficiency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary.get("scaling", []):
        lines.append(
            "| {precision} | {one} | {two} | {four} | {s12} | {s24} | {s14} | {eff} |".format(
                precision=item.get("precision", "n/a"),
                one=_number(item.get("one_gpu_throughput")),
                two=_number(item.get("two_gpu_throughput")),
                four=_number(item.get("four_gpu_throughput")),
                s12=_ratio(item.get("scale_1_to_2")),
                s24=_ratio(item.get("scale_2_to_4")),
                s14=_ratio(item.get("scale_1_to_4")),
                eff=_number(item.get("efficiency_1_to_4")),
            )
        )
    lines.extend(
        [
            "",
            "| GPUs | Precision | Hook | Baseline throughput | Hook throughput | Relative |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in summary.get("hook_comparisons", []):
        lines.append(
            "| {gpus} | {precision} | {hook} | {base} | {hook_t} | {relative} |".format(
                gpus=item.get("gpus", 0),
                precision=item.get("precision", "n/a"),
                hook=item.get("hook", "n/a"),
                base=_number(item.get("baseline_throughput")),
                hook_t=_number(item.get("hook_throughput")),
                relative=_ratio(item.get("relative_to_none")),
            )
        )
    if summary.get("bucket_comparisons"):
        lines.extend(
            [
                "",
                "| Bucket cap MB | Throughput | Relative to default | Dataloader wait ms |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for item in summary.get("bucket_comparisons", []):
            lines.append(
                "| {bucket} | {throughput} | {relative} | {wait} |".format(
                    bucket=item.get("bucket_cap_mb") or "default",
                    throughput=_number(item.get("throughput")),
                    relative=_ratio(item.get("relative_to_default")),
                    wait=_number(item.get("dataloader_wait_ms")),
                )
            )
    if summary.get("topology_comparisons"):
        lines.extend(
            [
                "",
                "| CUDA_VISIBLE_DEVICES | Bucket cap MB | Throughput | Dataloader wait ms |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in summary.get("topology_comparisons", []):
            lines.append(
                "| `{visible}` | {bucket} | {throughput} | {wait} |".format(
                    visible=item.get("visible_devices") or "all",
                    bucket=item.get("bucket_cap_mb") or "default",
                    throughput=_number(item.get("throughput")),
                    wait=_number(item.get("dataloader_wait_ms")),
                )
            )
    best = summary.get("best_dataloader")
    if isinstance(best, dict):
        lines.extend(
            [
                "",
                "Best dataloader candidate: `{run}` at {throughput} pairs/s, wait {wait} ms.".format(
                    run=best.get("run_id", "n/a"),
                    throughput=_number(best.get("throughput")),
                    wait=_number(best.get("dataloader_wait_ms")),
                ),
            ]
        )
    lines.append("")
    return lines


def _evidence_links(
    report_root: Path,
    summaries: Dict[str, Dict[str, Any]],
) -> List[str]:
    lines = [
        "## Evidence Files",
        "",
        "| Suite | Summary | Config snapshots |",
        "| --- | --- | --- |",
    ]
    for name, summary in summaries.items():
        summary_path = Path(summary.get("_summary_path", str(report_root / SUMMARY_FILES[name])))
        configs = ", ".join(
            f"[{_posix_path(path)}]({_posix_path(path)})"
            for path in CONFIG_FILES.get(name, ())
        )
        summary_link = _posix_path(summary_path.relative_to(report_root))
        lines.append(
            f"| {name} | [{summary_link}]({summary_link}) | {configs or 'n/a'} |"
        )
    lines.append("")
    return lines


def _status(summary: Dict[str, Any]) -> str:
    if summary.get("missing"):
        return "missing"
    if summary.get("blocked"):
        return "blocked"
    if summary.get("passed") is True:
        return "passed"
    if summary.get("passed") is False:
        return "failed"
    return "recorded"


def _number(value: Any, *, digits: int = 3) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _ratio(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return "n/a"


def _bytes_to_gb(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value) / 1024**3:.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _measurement_window(value: Any) -> str:
    if not isinstance(value, dict):
        return "n/a"
    warmup = value.get("warmup_steps_effective")
    measured = value.get("measured_batches")
    if warmup is None and measured is None:
        return "n/a"
    return f"{warmup or 0}/{measured or 0}"


def _hardware_summary(summary: Dict[str, Any]) -> str:
    runs = summary.get("runs", [])
    if not isinstance(runs, list):
        return "n/a"
    hardware = sorted(
        {
            str(run.get("hardware"))
            for run in runs
            if isinstance(run, dict) and run.get("hardware")
        }
    )
    if not hardware:
        return "n/a"
    if len(hardware) == 1:
        return hardware[0]
    return "multiple"


def _posix_path(path: Path) -> str:
    return path.as_posix()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default="tests/benchmarks/reports")
    parser.add_argument(
        "--output",
        default="tests/benchmarks/reports/BENCHMARK_REPORT.md",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report_root = Path(args.report_root)
    summaries = load_summaries(report_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_report_markdown(summaries, report_root=report_root),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
