# -*- coding: utf-8 -*-
# @Time : 2026/6/15 下午5:21
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Backend matrix result summarization and backend recommendation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from parascale.communication import build_communication_plan

BACKENDS = ("native_ddp", "fsdp", "deepspeed", "deepspeed_zero2", "deepspeed_zero3")
MEMORY_METRIC = "peak_memory_bytes"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _split_run_backend(stem: str, backend_hint: Any = None) -> tuple[str, str]:
    for candidate in sorted(BACKENDS, key=len, reverse=True):
        suffix = f"_{candidate}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], candidate
    backend = str(backend_hint or "")
    if backend in BACKENDS:
        return stem, backend
    return stem, backend or "unknown"


def _primary_metric(metrics: Dict[str, Any]) -> float:
    for name in (
        "stable_end_to_end_image_text_pairs_per_second",
        "stable_end_to_end_images_per_second",
        "end_to_end_image_text_pairs_per_second",
        "end_to_end_images_per_second",
        "samples_per_second",
    ):
        value = _safe_float(metrics.get(name))
        if value > 0:
            return value
    return 0.0


def _load_result(path: Path) -> Dict[str, Any]:
    if path.name.endswith(".error.json"):
        payload = _read_json(path)
        stem = path.name[: -len(".error.json")]
        run_id, backend = _split_run_backend(stem, payload.get("backend"))
        return {
            "run_id": run_id,
            "backend": backend,
            "status": "error",
            "error": payload.get("error", "benchmark failed"),
            "returncode": payload.get("returncode"),
            "attempt": payload.get("attempt"),
            "retry_trigger": payload.get("retry_trigger"),
            "retry_terminated": bool(payload.get("retry_terminated", False)),
            "retry_termination_reason": payload.get(
                "retry_termination_reason"
            ),
            "path": str(path),
            "config_artifacts": (
                payload.get("config_artifacts")
                if isinstance(payload.get("config_artifacts"), dict)
                else _inferred_config_artifacts(path.parent / stem)
            ),
        }
    payload = _read_json(path)
    train = payload.get("train", {})
    run_id, backend = _split_run_backend(path.stem, train.get("backend"))
    metrics = dict(payload.get("metrics", {}))
    config = payload.get("config", {})
    config_artifacts = payload.get("config_artifacts") or train.get(
        "config_artifacts", {}
    )
    return {
        "run_id": run_id,
        "backend": backend,
        "status": "ok",
        "metrics": metrics,
        "config": config if isinstance(config, dict) else {},
        "config_artifacts": (
            config_artifacts if isinstance(config_artifacts, dict) else {}
        ),
        "throughput": _primary_metric(metrics),
        "step_time_ms": _safe_float(metrics.get("stable_step_time_ms"))
        or _safe_float(metrics.get("step_time_ms")),
        "peak_memory_gb": _safe_float(metrics.get(MEMORY_METRIC)) / 1024**3,
        "path": str(path),
    }


def _inferred_config_artifacts(run_dir: Path) -> Dict[str, Any]:
    resolved = run_dir / "config.resolved.json"
    deepspeed = run_dir / "backend.deepspeed.final.json"
    return {
        "run_dir": str(run_dir),
        "resolved_config": str(resolved) if resolved.exists() else None,
        "deepspeed_final_config": str(deepspeed) if deepspeed.exists() else None,
    }


def collect(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if (
            path.name.endswith(".config.json")
            or path.name.startswith("summary")
            or path.name in {"comparison.json", "payload.json"}
            or path.name.endswith(".payload.json")
        ):
            continue
        rows.append(_load_result(path))
    return rows


def _group_by_run(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_run: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run_id"]), {})[str(row["backend"])] = row
    return by_run


def compare_backend_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparisons = []
    for run_id, backend_rows in sorted(_group_by_run(rows).items()):
        baseline = backend_rows.get("native_ddp")
        ordered_backends = [
            backend
            for backend in ("fsdp", "deepspeed", "deepspeed_zero2", "deepspeed_zero3")
            if backend in backend_rows
        ]
        for backend in ordered_backends:
            target = backend_rows.get(backend)
            if not baseline or not target:
                comparisons.append(
                    {
                        "run_id": run_id,
                        "target_backend": backend,
                        "baseline_backend": "native_ddp",
                        "status": "missing_result",
                        "passed": False,
                    }
                )
                continue
            if baseline.get("status") != "ok" or target.get("status") != "ok":
                comparisons.append(
                    {
                        "run_id": run_id,
                        "target_backend": backend,
                        "baseline_backend": "native_ddp",
                        "status": "error_result",
                        "passed": False,
                    }
                )
                continue
            baseline_value = _safe_float(baseline.get("throughput"))
            target_value = _safe_float(target.get("throughput"))
            baseline_memory = _safe_float(baseline.get("peak_memory_gb"))
            target_memory = _safe_float(target.get("peak_memory_gb"))
            comparisons.append(
                {
                    "run_id": run_id,
                    "target_backend": backend,
                    "baseline_backend": "native_ddp",
                    "target_value": target_value,
                    "baseline_value": baseline_value,
                    "speedup_vs_native_ddp": (
                        target_value / baseline_value if baseline_value > 0 else 0.0
                    ),
                    "memory_ratio_vs_native_ddp": (
                        target_memory / baseline_memory if baseline_memory > 0 else 0.0
                    ),
                    "passed": target_value > 0 and baseline_value > 0,
                }
            )
    return comparisons


def _retry_base_run_id(run_id: str) -> str | None:
    marker = "_oom_retry"
    if marker not in run_id:
        return None
    return run_id.split(marker, 1)[0]


def _retry_attempt_number(run_id: str) -> int | None:
    marker = "_oom_retry"
    if marker not in run_id:
        return None
    suffix = run_id.split(marker, 1)[1]
    digits = ""
    for character in suffix:
        if not character.isdigit():
            break
        digits += character
    return int(digits) if digits else None


def summarize_oom_recovery(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recoveries: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        base_run_id = _retry_base_run_id(run_id)
        if not base_run_id:
            continue
        item = recoveries.setdefault(
            base_run_id,
            {
                "run_id": base_run_id,
                "status": "unrecovered",
                "recovered": False,
                "attempts": [],
                "selected_backend": None,
            },
        )
        attempt = {
            "run_id": run_id,
            "backend": row.get("backend"),
            "status": row.get("status"),
            "throughput": _safe_float(row.get("throughput")),
            "peak_memory_gb": _safe_float(row.get("peak_memory_gb")),
            "error": row.get("error", ""),
            "attempt": row.get("attempt") or _retry_attempt_number(run_id),
            "retry_trigger": row.get("retry_trigger") or "oom",
            "retry_terminated": bool(row.get("retry_terminated", False)),
            "retry_termination_reason": row.get("retry_termination_reason"),
            "config_artifacts": row.get("config_artifacts", {}),
        }
        item["attempts"].append(attempt)
        if row.get("status") == "ok" and not item["recovered"]:
            item["status"] = "recovered"
            item["recovered"] = True
            item["selected_backend"] = row.get("backend")
            item["throughput"] = _safe_float(row.get("throughput"))
            item["peak_memory_gb"] = _safe_float(row.get("peak_memory_gb"))
    return [recoveries[key] for key in sorted(recoveries)]


def _recommended_config_updates(backend: str) -> Dict[str, Any]:
    updates: Dict[str, Any] = {"training_backend": backend}
    if backend == "native_ddp":
        updates.update(
            {
                "ddp_gradient_as_bucket_view": True,
                "ddp_static_graph": False,
            }
        )
    elif backend == "fsdp":
        updates.update(
            {
                "fsdp_sharding_strategy": "full_shard",
                "fsdp_state_dict_type": "full",
                "fsdp_use_orig_params": True,
            }
        )
    elif backend == "deepspeed":
        updates.update({"zero_optimization": True, "zero_stage": 2})
    elif backend == "deepspeed_zero2":
        updates.update(
            {
                "training_backend": "deepspeed",
                "zero_optimization": True,
                "zero_stage": 2,
            }
        )
    elif backend == "deepspeed_zero3":
        updates.update(
            {
                "training_backend": "deepspeed",
                "zero_optimization": True,
                "zero_stage": 3,
            }
        )
    return updates


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _communication_plan_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
    config = _section(row.get("config", {}), "parascale")
    metrics = row.get("metrics", {})
    backend = str(row.get("backend") or config.get("training_backend") or "native")
    return build_communication_plan(
        backend=backend,
        precision=str(config.get("precision", "fp32") or "fp32"),
        task_type=str(config.get("task_type", "") or ""),
        model_family=str(config.get("model_family", "") or ""),
        gradient_accumulation_steps=int(
            config.get("gradient_accumulation_steps", 1) or 1
        ),
        trainable_ratio=config.get("trainable_ratio"),
        dataloader_wait_ms=_safe_float(metrics.get("dataloader_wait_ms")),
    ).to_dict()


def _candidate_evaluations(
    backend_rows: Dict[str, Dict[str, Any]],
    *,
    selected_backend: str | None,
    optimize_for: str,
    max_throughput: float,
    selected_memory: float,
    throughput_tolerance: float,
) -> List[Dict[str, Any]]:
    evaluations = []
    minimum_throughput = max_throughput * (
        1.0 - max(0.0, throughput_tolerance)
    )
    for backend, row in sorted(backend_rows.items()):
        throughput = _safe_float(row.get("throughput"))
        memory = _safe_float(row.get("peak_memory_gb"))
        valid = row.get("status") == "ok" and throughput > 0
        within_tolerance = valid and throughput >= minimum_throughput
        selected = valid and backend == selected_backend
        rejection_reasons = []
        if not valid:
            rejection_reasons.append("benchmark_failed")
        elif not selected:
            if optimize_for == "throughput":
                rejection_reasons.append("lower_throughput_than_selected")
            elif optimize_for == "memory":
                rejection_reasons.append("higher_memory_than_selected")
            elif not within_tolerance:
                rejection_reasons.append("outside_throughput_tolerance")
            elif selected_memory > 0 and (memory <= 0 or memory > selected_memory):
                rejection_reasons.append("higher_memory_than_selected")
            else:
                rejection_reasons.append("lost_deterministic_tie_break")
        evaluations.append(
            {
                "backend": backend,
                "status": row.get("status"),
                "eligible": valid,
                "selected": selected,
                "throughput": throughput,
                "peak_memory_gb": memory,
                "throughput_ratio_to_best": (
                    throughput / max_throughput if max_throughput > 0 else 0.0
                ),
                "within_throughput_tolerance": within_tolerance,
                "rejection_reasons": rejection_reasons,
            }
        )
    return evaluations


def _expected_trade_off(optimize_for: str) -> str:
    if optimize_for == "throughput":
        return "Maximizes measured throughput and may use more memory."
    if optimize_for == "memory":
        return "Minimizes measured peak memory and may reduce throughput."
    return (
        "Minimizes memory within the throughput tolerance and may not select "
        "the absolute fastest backend."
    )


def _recommendation_confidence(valid_candidate_count: int) -> str:
    if valid_candidate_count >= 3:
        return "high"
    if valid_candidate_count == 2:
        return "medium"
    return "low"


def recommend_backends(
    rows: List[Dict[str, Any]],
    *,
    optimize_for: str = "balanced",
    throughput_tolerance: float = 0.05,
) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []
    for run_id, backend_rows in sorted(_group_by_run(rows).items()):
        ok_rows = [
            row
            for row in backend_rows.values()
            if row.get("status") == "ok" and _safe_float(row.get("throughput")) > 0
        ]
        if not ok_rows:
            recommendations.append(
                {
                    "run_id": run_id,
                    "selected_backend": None,
                    "status": "no_valid_backend",
                    "reason": "没有可用的成功 benchmark 结果。",
                    "evidence": {},
                    "candidate_evaluations": _candidate_evaluations(
                        backend_rows,
                        selected_backend=None,
                        optimize_for=optimize_for,
                        max_throughput=0.0,
                        selected_memory=0.0,
                        throughput_tolerance=throughput_tolerance,
                    ),
                    "expected_trade_off": _expected_trade_off(optimize_for),
                    "confidence": "low",
                    "actionable": False,
                    "recommended_config_updates": {},
                }
            )
            continue
        best_throughput_row = max(
            ok_rows, key=lambda row: _safe_float(row["throughput"])
        )
        best_memory_row = min(
            ok_rows,
            key=lambda row: (
                _safe_float(row.get("peak_memory_gb"))
                if _safe_float(row.get("peak_memory_gb")) > 0
                else float("inf")
            ),
        )
        max_throughput = _safe_float(best_throughput_row["throughput"])
        candidates = [
            row
            for row in ok_rows
            if _safe_float(row["throughput"])
            >= max_throughput * (1.0 - max(0.0, throughput_tolerance))
        ]
        if optimize_for == "throughput":
            selected = best_throughput_row
            policy = "highest_throughput"
        elif optimize_for == "memory":
            selected = best_memory_row
            policy = "lowest_memory"
        else:
            selected = min(
                candidates,
                key=lambda row: (
                    _safe_float(row.get("peak_memory_gb"))
                    if _safe_float(row.get("peak_memory_gb")) > 0
                    else float("inf")
                ),
            )
            policy = "balanced_throughput_memory"
        selected_backend = str(selected["backend"])
        selected_throughput = _safe_float(selected["throughput"])
        selected_memory = _safe_float(selected.get("peak_memory_gb"))
        speedup_vs_best = (
            selected_throughput / max_throughput if max_throughput > 0 else 0.0
        )
        best_memory = _safe_float(best_throughput_row.get("peak_memory_gb"))
        memory_vs_best_throughput = (
            selected_memory / best_memory if best_memory > 0 else 0.0
        )
        valid_candidate_count = len(ok_rows)
        reason = (
            f"选择 {selected_backend}: policy={policy}, 吞吐为最高吞吐的 "
            f"{speedup_vs_best:.3f}，显存相对最高吞吐后端为 {memory_vs_best_throughput:.3f}。"
        )
        recommendations.append(
            {
                "run_id": run_id,
                "selected_backend": selected_backend,
                "status": "ok",
                "policy": policy,
                "reason": reason,
                "communication_plan": _communication_plan_for_row(selected),
                "candidate_evaluations": _candidate_evaluations(
                    backend_rows,
                    selected_backend=selected_backend,
                    optimize_for=optimize_for,
                    max_throughput=max_throughput,
                    selected_memory=selected_memory,
                    throughput_tolerance=throughput_tolerance,
                ),
                "expected_trade_off": _expected_trade_off(optimize_for),
                "confidence": _recommendation_confidence(valid_candidate_count),
                "actionable": valid_candidate_count >= 2,
                "evidence": {
                    "selected_throughput": selected_throughput,
                    "selected_peak_memory_gb": selected_memory,
                    "best_throughput_backend": best_throughput_row.get("backend"),
                    "best_throughput": max_throughput,
                    "lowest_memory_backend": best_memory_row.get("backend"),
                    "lowest_peak_memory_gb": _safe_float(
                        best_memory_row.get("peak_memory_gb")
                    ),
                    "throughput_tolerance": throughput_tolerance,
                    "valid_candidate_count": valid_candidate_count,
                    "failed_candidate_count": len(backend_rows)
                    - valid_candidate_count,
                },
                "recommended_config_updates": _recommended_config_updates(
                    selected_backend
                ),
            }
        )
    return recommendations


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        f"# {report['title']}",
        "",
        "## 测试范围",
        "",
        f"- Workload: {report['workload_label']}",
        "- Backends: native-DDP, FSDP, DeepSpeed。",
        "- 指标: 稳态端到端吞吐、step time、CUDA peak allocated memory。",
        f"- 推荐策略: {report['optimize_for']}。",
        "",
        "## 结果",
        "",
        "| Run | Backend | Status | Throughput | Step ms | Peak memory GB | Error |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["results"]:
        lines.append(
            "| {run} | {backend} | {status} | {throughput:.3f} | {step:.3f} | {memory:.3f} | {error} |".format(
                run=row.get("run_id", ""),
                backend=row.get("backend", ""),
                status=row.get("status", ""),
                throughput=_safe_float(row.get("throughput")),
                step=_safe_float(row.get("step_time_ms")),
                memory=_safe_float(row.get("peak_memory_gb")),
                error=str(row.get("error", ""))[:140],
            )
        )
    lines.extend(["", "## 推荐后端", ""])
    for item in report["recommendations"]:
        communication_plan = item.get("communication_plan", {})
        lines.append(
            "- {run}: {backend}。{reason}".format(
                run=item.get("run_id"),
                backend=item.get("selected_backend") or "n/a",
                reason=item.get("reason", ""),
            )
        )
        if communication_plan:
            lines.append(
                "  - Communication Plan: backend={backend}, ddp_hook={hook}, no_sync={no_sync}, overlap_h2d={overlap}".format(
                    backend=communication_plan.get("backend", "unknown"),
                    hook=communication_plan.get("ddp_hook", "none"),
                    no_sync=communication_plan.get("use_no_sync", False),
                    overlap=communication_plan.get("overlap_h2d", False),
                )
            )
        lines.append(
            "  - Expected trade-off: {trade_off}; confidence={confidence}; "
            "actionable={actionable}".format(
                trade_off=item.get("expected_trade_off", "n/a"),
                confidence=item.get("confidence", "low"),
                actionable=item.get("actionable", False),
            )
        )
        candidates = item.get("candidate_evaluations", [])
        if candidates:
            lines.extend(
                [
                    "",
                    "### Candidate Evaluation: {run}".format(
                        run=item.get("run_id", "")
                    ),
                    "",
                    "| Backend | Status | Selected | Throughput | Peak memory GB | Reasons |",
                    "| --- | --- | --- | ---: | ---: | --- |",
                ]
            )
            for candidate in candidates:
                lines.append(
                    "| {backend} | {status} | {selected} | {throughput:.3f} | "
                    "{memory:.3f} | {reasons} |".format(
                        backend=candidate.get("backend", ""),
                        status=candidate.get("status", ""),
                        selected=candidate.get("selected", False),
                        throughput=_safe_float(candidate.get("throughput")),
                        memory=_safe_float(candidate.get("peak_memory_gb")),
                        reasons=", ".join(
                            candidate.get("rejection_reasons", [])
                        ),
                    )
                )
    if report.get("tuner_explanations"):
        lines.extend(["", "## 选择依据", ""])
        for item in report["tuner_explanations"]:
            explain = item.get("explain", {})
            tuning = item.get("runtime_tuning", {})
            decisions = tuning.get("decisions", []) if isinstance(tuning, dict) else []
            summary = explain.get("summary", "") if isinstance(explain, dict) else ""
            lines.append(
                "- {run}/{backend}: {summary}".format(
                    run=item.get("run_id"),
                    backend=item.get("backend"),
                    summary=summary or item.get("error", "no tuner explanation"),
                )
            )
            for decision in decisions[:2]:
                evidence = decision.get("evidence", {})
                dominant = evidence.get("dominant_pipeline_stage")
                if dominant:
                    lines.append(
                        "  - dominant_pipeline_stage={stage}, dataloader_wait_ms={wait}".format(
                            stage=dominant,
                            wait=evidence.get("dataloader_wait_ms", "n/a"),
                        )
                    )
    lines.extend(["", "## 对照", ""])
    for item in report["comparisons"]:
        lines.append(
            "- {run}: {target} vs native-DDP, speedup={speedup:.4f}, memory_ratio={memory:.4f}, status={status}".format(
                run=item.get("run_id"),
                target=item.get("target_backend"),
                speedup=_safe_float(item.get("speedup_vs_native_ddp")),
                memory=_safe_float(item.get("memory_ratio_vs_native_ddp")),
                status=item.get("status", "ok"),
            )
        )
    lines.extend(["", "## 结论", "", report["conclusion"], ""])
    if report.get("oom_recovery"):
        lines.extend(["", "## OOM Recovery", ""])
        for item in report["oom_recovery"]:
            lines.append(
                "- {run}: {status}, selected={backend}, throughput={throughput:.3f}, peak_memory_gb={memory:.3f}".format(
                    run=item.get("run_id"),
                    status=item.get("status"),
                    backend=item.get("selected_backend") or "n/a",
                    throughput=_safe_float(item.get("throughput")),
                    memory=_safe_float(item.get("peak_memory_gb")),
                )
            )
            for attempt in item.get("attempts", []):
                lines.append(
                    "  - {run}: {backend}, {status}, {error}".format(
                        run=attempt.get("run_id"),
                        backend=attempt.get("backend"),
                        status=attempt.get("status"),
                        error=str(attempt.get("error", ""))[:120],
                    )
                )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(
    input_dir: Path,
    *,
    title: str,
    workload_label: str,
    optimize_for: str = "balanced",
    throughput_tolerance: float = 0.05,
) -> Dict[str, Any]:
    rows = collect(input_dir)
    comparisons = compare_backend_rows(rows)
    recommendations = recommend_backends(
        rows,
        optimize_for=optimize_for,
        throughput_tolerance=throughput_tolerance,
    )
    oom_recovery = summarize_oom_recovery(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    has_errors = any(row.get("status") != "ok" for row in rows)
    closed_runs = []
    for backend_rows in _group_by_run(ok_rows).values():
        actual = {str(row.get("backend")) for row in backend_rows.values()}
        closed_runs.append("native_ddp" in actual and len(actual) >= 2)
    if rows and not has_errors and closed_runs and all(closed_runs):
        conclusion = (
            "三类后端均完成同口径运行，ParaScale 已给出基于吞吐和显存的推荐后端。"
        )
    elif oom_recovery and all(item.get("recovered") for item in oom_recovery):
        conclusion = (
            "矩阵发现部分后端 OOM 或 retry 兼容性失败，但 OOM retry 已通过 "
            "fallback 后端恢复；建议采用报告中的推荐后端，并将失败后端作为优化项跟踪。"
        )
    else:
        conclusion = "矩阵未完全闭环，需优先处理失败后端或缺失依赖后再做性能判断。"
    evidence_summary = _evidence_summary(
        rows=rows,
        recommendations=recommendations,
        oom_recovery=oom_recovery,
    )
    return {
        "title": title,
        "workload_label": workload_label,
        "optimize_for": optimize_for,
        "throughput_tolerance": throughput_tolerance,
        "results": rows,
        "comparisons": comparisons,
        "recommendations": recommendations,
        "oom_recovery": oom_recovery,
        "evidence_summary": evidence_summary,
        "conclusion": conclusion,
    }


def _evidence_summary(
    *,
    rows: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    oom_recovery: List[Dict[str, Any]],
) -> Dict[str, Any]:
    selected_backends = [
        str(item.get("selected_backend"))
        for item in recommendations
        if item.get("selected_backend")
    ]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    return {
        "result_count": len(rows),
        "ok_result_count": len(ok_rows),
        "failed_result_count": len(failed_rows),
        "recommendation_count": len(recommendations),
        "selected_backends": selected_backends,
        "oom_recovery_count": len(oom_recovery),
        "recovered_oom_count": len(
            [item for item in oom_recovery if item.get("recovered")]
        ),
    }
