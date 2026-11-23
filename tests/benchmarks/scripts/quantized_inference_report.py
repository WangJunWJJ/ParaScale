# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pure-Python comparison of reproducible, real-model inference measurements."""

import math
import statistics


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
