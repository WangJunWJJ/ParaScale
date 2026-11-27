# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Strict functional evidence checks, without inferring quality or speedups."""

import hashlib
import json
import math
from pathlib import Path


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite evidence")
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def real_payload(payload):
    finite_tree(payload)
    if any(payload.get(name) is not False for name in ("dry_run", "mock", "synthetic")):
        raise ValueError("real execution evidence required; no synthetic/mock/dry-run fallback")


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite measurement required")
    return value


def check_inference(payload: dict, kind: str) -> dict:
    """Require real, nonempty, completely published inference results."""
    real_payload(payload)
    expected = "causal_lm_generation" if kind == "transformer" else "diffusion_generation"
    if payload.get("task") != expected or payload.get("runtime_status") != "real_local":
        raise ValueError("unexpected inference task/runtime")
    metrics, stream = payload["metrics"], payload["output_stream"]
    count = positive(metrics["requests"])
    if stream.get("complete") is not True or stream.get("records") != count:
        raise ValueError("incomplete output stream")
    identity = payload["execution"]
    if not identity.get("model_assets_sha256") or not identity.get("input_sha256"):
        raise ValueError("missing input/model identity")
    if kind == "transformer":
        positive(metrics["generated_tokens"])
        positive(metrics["generated_tokens_per_second"])
    else:
        positive(metrics["images"])
        positive(metrics["images_per_second"])
    return dict(metrics)


def check_training(train: dict, resume: dict, steps: int) -> dict:
    """Check bounded updates and fresh-process continuation, not numerical parity."""
    for payload, initial, final in ((train, 0, steps), (resume, steps, 2 * steps)):
        real_payload(payload)
        if payload.get("backend") != "native_parallel" or payload.get("runtime_status") != "real_distributed":
            raise ValueError("unexpected training runtime")
        if (payload.get("initial_step"), payload.get("global_step"), payload.get("completed_updates")) != (initial, final, steps):
            raise ValueError("training did not continue the expected updates")
        if payload.get("checkpoint", {}).get("step") != final or not payload["checkpoint"].get("path"):
            raise ValueError("missing final checkpoint")
        if payload.get("resume_selection", {}).get("step") != initial:
            raise ValueError("unexpected recovery selection")
        history = payload.get("metrics_history", [])
        if not history or any(not isinstance(row.get("loss"), (int, float)) for row in history):
            raise ValueError("missing training loss evidence")
        positive(payload["scored_tokens_per_second"])
    return {"trained_step": steps, "resumed_step": 2 * steps,
            "train_scored_tokens_per_second": train["scored_tokens_per_second"],
            "resume_scored_tokens_per_second": resume["scored_tokens_per_second"],
            "timing_scope": train.get("timing_scope"),
            "peak_memory_allocated_bytes": resume.get("peak_memory_allocated_bytes")}


def verify_stream(payload, kind, directory):
    """Verify published records and physical images, not generation quality."""
    stream = Path(payload["output_stream"]["path"]).resolve()
    if not stream.is_relative_to(directory.resolve()) or not stream.is_file():
        raise ValueError("published output stream missing or outside case output")
    count, tokens, seen = 0, 0, set()
    with stream.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("output record must be an object")
            count += 1
            if kind == "diffusion":
                from PIL import Image

                path = (stream.parent / record["file"]).resolve()
                if not path.is_relative_to(stream.parent) or not path.is_file() or path in seen:
                    raise ValueError("missing, repeated or external image")
                seen.add(path)
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("image checksum mismatch")
                with Image.open(path) as image:
                    if image.format != "PNG" or image.size != (record["width"], record["height"]):
                        raise ValueError("image dimensions/format mismatch")
                    image.verify()
            else:
                ids = record["generated_token_ids"]
                if not isinstance(record["text"], str) or not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                    raise ValueError("invalid generated text/token record")
                if record["generated_tokens"] != len(ids):
                    raise ValueError("generated token count mismatch")
                tokens += len(ids)
    if count != payload["output_stream"]["records"]:
        raise ValueError("published output count differs from summary")
    if kind == "transformer" and tokens != payload["metrics"]["generated_tokens"]:
        raise ValueError("published tokens differ from measured tokens")
