# -*- coding: utf-8 -*-
# @Time : 2026/11/24 下午9:18
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded diffusion image publication, separate from generation timing."""

import json
import os
import time
from pathlib import Path

from parascale.workloads.causal_lm_assets import file_sha256

from .runner import InferenceRunner
from .tasks.text import TextInferenceTaskAdapter


class DiffusionTaskAdapter(TextInferenceTaskAdapter):
    name = "diffusion_generation"

    def prepare_model_batch(self, model, batch):
        return model.prepare_batch(batch)

    def metric_counts(self, batch):
        return {"requests": len(batch["prompts"]), "images": len(batch["prompts"])}


def validate_diffusion_paths(
    spec, *, config_path=None, summary_path=None, run_dir=None
):
    """Reject output/input overlaps before any command or model writes."""
    model = Path(spec.model_path).resolve()
    output = Path(spec.output_dir).resolve()
    inputs = [Path(spec.input_path).resolve()]
    if config_path:
        inputs.append(Path(config_path).resolve())
    if (
        output.is_relative_to(model)
        or model.is_relative_to(output)
        or any(path.is_relative_to(output) for path in inputs)
    ):
        raise ValueError("Diffusion output overlaps model or input assets")
    if output.exists():
        raise FileExistsError(f"Diffusion output already exists: {output}")
    if run_dir is not None:
        run = Path(run_dir).resolve()
        if (
            run.is_relative_to(model)
            or model.is_relative_to(run)
            or run.is_relative_to(output)
            or any(path.is_relative_to(run) for path in inputs)
        ):
            raise ValueError("Diffusion run directory overlaps assets or image output")
        if output.parent == run and (
            output.name in {"config.resolved.json", "backend.deepspeed.final.json"}
            or output.name.startswith(".")
        ):
            raise ValueError("Diffusion output overlaps configuration artifacts")
    if summary_path:
        summary = Path(summary_path).resolve()
        if (
            summary.is_relative_to(model)
            or summary in inputs
            or summary.is_relative_to(output)
            or output.is_relative_to(summary)
        ):
            raise ValueError("Diffusion summary overlaps assets or image output")
        if run_dir is not None and (
            summary == run
            or run.is_relative_to(summary)
            or (
                summary.parent == run
                and (
                    summary.name
                    in {"config.resolved.json", "backend.deepspeed.final.json"}
                    or summary.name.startswith(".")
                )
            )
        ):
            raise ValueError("Diffusion summary overlaps run/configuration output")
        if summary.exists():
            raise FileExistsError(f"Diffusion summary already exists: {summary}")


def run_diffusion_inference(model, batches):
    """Publish bounded PNG output with a final manifest only on complete success.

    The output directory is exclusively created for one job. Failed jobs retain
    a partial manifest and any completed images, never a success manifest.
    Trusted local filesystems must support hard links; no output is overwritten.
    """
    from PIL import Image

    started = time.perf_counter()
    spec = model.spec
    output = Path(spec.output_dir).resolve()
    written = 0
    try:
        validate_diffusion_paths(spec)
        device, memory = model.resolve_device()
        output.mkdir(parents=True, exist_ok=False)
        partial = output / "manifest.jsonl.partial"
        with partial.open("x", encoding="utf-8", newline="\n") as stream:

            def emit(batch):
                nonlocal written
                for index, pixels in enumerate(batch["images"]):
                    path = output / f"{written:08d}.png"
                    with path.open("xb") as image_stream:
                        Image.fromarray((pixels * 255).round().astype("uint8")).save(
                            image_stream, format="PNG"
                        )
                        image_stream.flush()
                        os.fsync(image_stream.fileno())
                    flags = batch["safety_flags"]
                    record = {
                        "id": batch["ids"][index],
                        "seed": batch["seeds"][index],
                        "file": path.name,
                        "sha256": file_sha256(path),
                        "width": spec.width,
                        "height": spec.height,
                        "nsfw_content_detected": bool(flags[index])
                        if flags is not None
                        else None,
                    }
                    stream.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    written += 1

            payload = InferenceRunner(
                model=model,
                task="diffusion_generation",
                task_adapter=DiffusionTaskAdapter(),
                device=device,
                memory_getter=memory,
            ).run(
                batches,
                warmup_steps=spec.warmup_steps,
                warmup_policy="replay_first",
                max_outputs=0,
                on_output=emit,
            )
            if not written:
                raise ValueError("Diffusion input contains no prompts")
            stream.flush()
            os.fsync(stream.fileno())
        execution = model.execution_metadata
        os.link(partial, output / "manifest.jsonl")
        partial.unlink()
    finally:
        model.close()
    payload["timing"].update(
        job_wall_seconds=time.perf_counter() - started,
        job_wall_scope="setup, warmup, input, generation, PNG encoding and publication",
    )
    payload["metrics"]["latency_scope"] = (
        "amortized image generation; includes safety checks and CPU image return, excludes PNG encoding/writes"
    )
    return {
        "mode": "infer",
        "workload": "diffusion",
        "dry_run": False,
        "mock": False,
        "synthetic": False,
        "runtime_status": "real_local" if device.startswith("cuda") else "local_cpu",
        "capability_level": "local_diffusion_generation",
        **payload,
        "execution": execution,
        "output_stream": {
            "path": str(output / "manifest.jsonl"),
            "records": written,
            "complete": True,
        },
    }
