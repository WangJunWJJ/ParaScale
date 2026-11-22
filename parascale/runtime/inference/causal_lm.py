# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:31
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded local generation job lifecycle and output publication."""

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

from parascale.workloads.specs.causal_lm_inference import CausalLmInferenceSpec

from .runner import InferenceRunner


def _output_paths(spec):
    if spec.output_jsonl is None:
        return None, None
    output = Path(spec.output_jsonl).resolve()
    partial = output.with_name(output.name + ".partial")
    model = Path(spec.model_path).resolve()
    inputs = {Path(spec.input_path).resolve()}
    if spec.artifact_path is not None:
        inputs.add(Path(spec.artifact_path).resolve())
    for path in (output, partial):
        if path.is_relative_to(model) or path in inputs:
            raise ValueError("Generation output overlaps model or input assets")
        if path.exists():
            raise FileExistsError(path)
    return output, partial


def validate_command_paths(
    spec: CausalLmInferenceSpec,
    *,
    config_path: str,
    summary_path: str | None,
    run_dir: Path,
) -> None:
    """Reject colliding job paths before CLI config writes or model loading.

    Inputs must live outside the dedicated run directory. Stream/summary files
    may live inside it but cannot use configuration-artifact names. Existing
    summary and stream files are never reused. This is a single-owner local job,
    not protection against concurrent adversarial filesystem mutation.
    """
    output, partial = _output_paths(spec)
    model = Path(spec.model_path).resolve()
    source = Path(config_path).resolve()
    protected = {source, Path(spec.input_path).resolve()}
    if spec.artifact_path:
        protected.add(Path(spec.artifact_path).resolve())
    directory = run_dir.resolve()
    if (
        directory.is_relative_to(model)
        or model.is_relative_to(directory)
        or any(path.is_relative_to(directory) for path in protected)
    ):
        raise ValueError("Run directory overlaps model or input assets")
    summary = Path(summary_path).resolve() if summary_path else None
    destinations = [path for path in (output, partial, summary) if path is not None]
    for index, path in enumerate(destinations):
        if path.is_relative_to(model) or path in protected:
            raise ValueError("Inference output overlaps model or input assets")
        if any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in destinations[:index]
        ):
            raise ValueError("Inference output paths overlap")
        if path.parent == directory and (
            path.name in {"config.resolved.json", "backend.deepspeed.final.json"}
            or path.name.startswith(
                (".config.resolved.json.tmp-", ".backend.deepspeed.final.json.tmp-")
            )
        ):
            raise ValueError("Inference output overlaps configuration artifacts")
        if path == directory or directory.is_relative_to(path):
            raise ValueError("Inference output overlaps run directory")
    if summary is not None and summary.exists():
        raise FileExistsError(f"Inference output already exists: {summary}")


def run_causal_lm_inference(model, batches):
    """Execute one isolated model job and publish a new JSONL file on success.

    The synchronous sink is bounded; an interrupted/failed stream retains its
    `.partial` suffix. Final publication uses an exclusive hard link, never
    overwrites a target, and requires a filesystem supporting local hard links.
    Prediction timing excludes tokenization/transfer/output I/O; job wall time
    includes them, setup, warmup and publication. Model-owned caches are released
    on both success and failure. All validation/execution/I/O errors propagate.
    """
    started = time.perf_counter()
    spec = model.spec
    output, partial = _output_paths(spec)
    written = 0
    try:
        device, memory = model.resolve_device()
        with ExitStack() as stack:
            stream = None
            if partial is not None:
                partial.parent.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(
                    partial.open("x", encoding="utf-8", newline="\n")
                )

            def emit(batch):
                nonlocal written
                for record in batch["records"]:
                    stream.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    written += 1

            payload = InferenceRunner(
                model=model,
                task="causal_lm_generation",
                device=device,
                memory_getter=memory,
            ).run(
                batches,
                warmup_steps=spec.warmup_steps,
                warmup_policy="replay_first",
                max_outputs=spec.max_outputs,
                on_output=emit if stream is not None else None,
            )
            execution = model.execution_metadata
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
        if output is not None:
            os.link(partial, output)
            partial.unlink()
    finally:
        model.close()
    payload["timing"]["job_wall_seconds"] = time.perf_counter() - started
    payload["timing"]["job_wall_scope"] = (
        "setup, warmup, input, prediction, output and publication"
    )
    return {
        "mode": "infer",
        "dry_run": False,
        "mock": False,
        "synthetic": False,
        "runtime_status": "real_local" if device.startswith("cuda") else "local_cpu",
        "capability_level": "local_causal_lm_generation",
        "workload": "causal_lm",
        **payload,
        "execution": execution,
        "output_stream": {
            "enabled": output is not None,
            "path": str(output) if output else None,
            "records": written,
            "complete": True,
        },
    }
