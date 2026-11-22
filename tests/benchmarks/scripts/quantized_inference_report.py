# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pure-Python comparison of reproducible, real-model inference measurements."""

import math
import statistics
from typing import Any, Iterable


def quantized_cuda_kernels(events: Iterable[Any], *, backend: str) -> list[str]:
    """Return observed packed-matmul CUDA kernel names for the selected backend.

    CPU operator names, annotations and zero-duration activities are not evidence.
    Unknown backends or kernel names fail closed by returning an empty list.
    """
    markers = {
        "gemlite_int8": (
            "gemv_INT_kernel",
            "gemm_INT_kernel",
            "gemv_INT_splitK_kernel",
            "gemm_splitK_INT_kernel",
        ),
        "torchao_int4": ("tinygemm_m16n8k16_chunk_kernel",),
    }.get(backend, ())
    return sorted(
        {
            event.key
            for event in events
            if event.device_type.name == "CUDA"
            and not getattr(event, "is_user_annotation", False)
            and math.isfinite(event.self_device_time_total)
            and event.self_device_time_total > 0
            and event.count > 0
            and any(marker in event.key for marker in markers)
        }
    )


def summarize_profile(
    events: Iterable[Any], *, wall_ms: float, top_k: int = 15
) -> dict[str, Any]:
    """Summarize one diagnostic trace without double-counting linked CUDA time.

    Activity self-times are microseconds, not utilization. CPU waits remain CPU
    events; CUDA event intervals from the benchmark cannot estimate host overhead.
    This profile is deliberately excluded from throughput and acceptance timing.
    """
    if not math.isfinite(wall_ms) or wall_ms < 0 or type(top_k) is not int or top_k < 1:
        raise ValueError("Invalid profile duration or top_k")
    groups = {"CPU": [], "CUDA": []}
    for event in events:
        device = event.device_type.name
        if device not in groups or getattr(event, "is_user_annotation", False):
            continue
        elapsed = (
            event.self_cpu_time_total
            if device == "CPU"
            else event.self_device_time_total
        )
        if not math.isfinite(elapsed) or elapsed < 0 or event.count <= 0:
            raise ValueError("Invalid profile activity time or count")
        groups[device].append(
            {"name": event.key, "count": event.count, "self_us": elapsed}
        )
    summary = {
        "scope": "one profiled generation; activity sums are not utilization",
        "iterations": 1,
        "wall_ms": wall_ms,
        "timing_used_for_throughput": False,
    }
    for device, rows in groups.items():
        rows.sort(key=lambda row: (-row["self_us"], row["name"]))
        total = sum(row["self_us"] for row in rows)
        selected = [
            {
                **row,
                "mean_self_us": row["self_us"] / row["count"],
                "fraction_of_activity_time": row["self_us"] / total if total else 0.0,
            }
            for row in rows[:top_k]
        ]
        summary[device.lower()] = {
            "count": sum(row["count"] for row in rows),
            "self_us": total,
            "top": selected,
            "other_count": sum(row["count"] for row in rows[top_k:]),
            "other_self_us": sum(row["self_us"] for row in rows[top_k:]),
        }
    summary["cuda_activities_observed"] = summary["cuda"]["self_us"] > 0
    return summary


def compare_results(baseline, candidate, min_speedup=1.2, max_ppl_increase=0.05):
    if baseline["mode"] != "bf16" or candidate["mode"] not in ("int4", "int8"):
        raise ValueError("comparison requires a bf16 baseline and int4 candidate")
    expected_execution = f"{candidate['mode']}_weight_only_fused_bf16"
    if (candidate.get("conversion") or {}).get("execution") != expected_execution:
        raise ValueError("Candidate mode does not match actual conversion execution")
    for field in ("contract", "environment"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Results are not comparable: different {field}")
    for key in ("model_sha256", "data_sha256"):
        if not baseline["contract"].get(key):
            raise ValueError(f"Missing provenance: {key}")
    artifact = baseline["contract"].get("reference_artifact")
    if artifact is not None:
        conversion = candidate["conversion"]
        if (
            not artifact.get("sha256")
            or conversion.get("source_format") != "parascale.reference_quantization"
            or conversion.get("requantized") is not False
            or conversion.get("stored_codes_and_scales_preserved") is not True
        ):
            raise ValueError(
                "Missing artifact identity or preserved quantization evidence"
            )
    if not math.isfinite(min_speedup) or min_speedup <= 1:
        raise ValueError("min_speedup must be finite and > 1")
    if not math.isfinite(max_ppl_increase) or max_ppl_increase < 0:
        raise ValueError("max_ppl_increase must be finite and >= 0")
    samples = [
        item["timing"]["generation"]["wall_ms"] for item in (baseline, candidate)
    ]
    valid = all(
        len(values) >= 20 and all(math.isfinite(v) and v > 0 for v in values)
        for values in samples
    )
    quality = [item["quality"] for item in (baseline, candidate)]
    valid = (
        valid
        and all(
            q["tokens"] > 0 and math.isfinite(q["perplexity"]) and q["perplexity"] >= 1
            for q in quality
        )
        and quality[0]["tokens"] == quality[1]["tokens"]
    )
    speedup = (
        statistics.median(samples[0]) / statistics.median(samples[1]) if valid else None
    )
    increase = (
        quality[1]["perplexity"] / quality[0]["perplexity"] - 1 if valid else None
    )
    coverage = bool(candidate.get("conversion", {}).get("converted_layers"))
    kernel = candidate.get("kernel_verified") is True
    compilation = not baseline["contract"].get("compile", False) or all(
        item.get("profiled_compiled_regions") for item in (baseline, candidate)
    )
    passed = bool(
        valid
        and coverage
        and kernel
        and compilation
        and speedup >= min_speedup
        and increase <= max_ppl_increase
    )
    return {
        "engineering_gate_passed": passed,
        "speedup": speedup,
        "relative_perplexity_increase": increase,
        "valid_measurements": valid,
        "kernel_verified": kernel,
        "compilation_verified_if_requested": bool(compilation),
        "min_speedup": min_speedup,
        "max_relative_perplexity_increase": max_ppl_increase,
        "scope": "Fixed-length generation on this checkpoint, corpus, shape and GPU only; not contractual acceptance.",
    }
