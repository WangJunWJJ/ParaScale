# -*- coding: utf-8 -*-
# @Time : 2026/6/15 下午3:39
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Summarize P3 mixed-precision and checkpoint stress validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "path": str(path), "error": str(exc)}
    payload.setdefault("path", str(path))
    return payload


def _checkpoint_ok(payload: Dict[str, Any]) -> bool:
    return bool(payload.get("checkpoint_validation", {}).get("ok", False))


def _resume_loaded(payload: Dict[str, Any]) -> bool:
    return bool(payload.get("resumed_from"))


def _train_check(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("last_metrics", {})
    return {
        "name": name,
        "ok": _checkpoint_ok(payload),
        "backend": payload.get("backend"),
        "global_step": payload.get("global_step"),
        "checkpoint_ok": _checkpoint_ok(payload),
        "resume_loaded": _resume_loaded(payload),
        "peak_memory_bytes": metrics.get("peak_memory_bytes"),
        "pairs_per_second": metrics.get("end_to_end_image_text_pairs_per_second"),
        "path": payload.get("path"),
        "status": payload.get("status", "ok" if payload.get("mode") else "unknown"),
        "error": payload.get("error"),
    }


def _resume_check(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    item = _train_check(name, payload)
    item["ok"] = _checkpoint_ok(payload) and _resume_loaded(payload)
    return item


def _hf_check(path: Path) -> Dict[str, Any]:
    payload = _load(path)
    item = _train_check("hf_clip_pretrained_offline_smoke", payload)
    if payload.get("status") == "skipped":
        item["ok"] = True
        item["skipped"] = True
        item["skip_reason"] = payload.get("reason")
    else:
        item["skipped"] = False
        item["ok"] = _checkpoint_ok(payload) or bool(payload.get("global_step"))
    return item


def build_report(input_dir: Path) -> Dict[str, Any]:
    checks = [
        _train_check(
            "native_bf16_activation_ckpt_train",
            _load(input_dir / "native_bf16_train.json"),
        ),
        _resume_check(
            "native_bf16_activation_ckpt_resume",
            _load(input_dir / "native_bf16_resume.json"),
        ),
        _train_check(
            "deepspeed_zero3_train", _load(input_dir / "deepspeed_zero3_train.json")
        ),
        _resume_check(
            "deepspeed_zero3_resume", _load(input_dir / "deepspeed_zero3_resume.json")
        ),
        _train_check(
            "deepspeed_zero3_activation_ckpt_train",
            _load(input_dir / "deepspeed_zero3_activation_ckpt_train.json"),
        ),
        _hf_check(input_dir / "hf_clip_pretrained_offline_smoke.json"),
    ]
    return {
        "mode": "p3_mixed_precision_checkpoint_validation",
        "input_dir": str(input_dir),
        "passed": all(item.get("ok") for item in checks),
        "checks": checks,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = [
        "# P3 Mixed Precision / ZeRO-3 / Resume Validation Report",
        "",
        f"- Passed: {report['passed']}",
        f"- Input directory: `{report['input_dir']}`",
        "",
        "| Check | OK | Backend | Step | Checkpoint | Resume loaded | Pairs/s | Peak memory GB |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for item in report["checks"]:
        memory_gb = float(item.get("peak_memory_bytes") or 0.0) / 1024**3
        pairs = float(item.get("pairs_per_second") or 0.0)
        lines.append(
            "| {name} | {ok} | {backend} | {step} | {ckpt} | {resume} | {pairs:.3f} | {memory:.3f} |".format(
                name=item.get("name"),
                ok=item.get("ok"),
                backend=item.get("backend", "n/a"),
                step=item.get("global_step") or 0,
                ckpt=item.get("checkpoint_ok"),
                resume=item.get("resume_loaded"),
                pairs=pairs,
                memory=memory_gb,
            )
        )
    skipped = [item for item in report["checks"] if item.get("skipped")]
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for item in skipped:
            lines.append(f"- `{item['name']}`: {item.get('skip_reason')}")
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
