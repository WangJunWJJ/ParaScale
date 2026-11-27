# -*- coding: utf-8 -*-
# @Time : 2026/11/24 下午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Isolated native/product eager/compiled diffusion benchmark with image gates.

Generated results and models belong outside the checkout. Compilation is an
upstream optimization, not a claim of an exclusive ParaScale algorithm. Fixed
prompt pixel fidelity is not semantic image-quality certification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MODES = ("native-eager", "parascale-eager", "native-compiled", "parascale-compiled")


def validate_reports(rows):
    """Reject changed work, incomplete pairs and invalid performance measures."""
    if not rows:
        raise ValueError("No repeat results")
    groups = {}
    baseline = rows[0]
    for row in rows:
        for field in ("model_assets_sha256", "input_sha256", "settings", "images"):
            if field not in row or row[field] != baseline[field]:
                raise ValueError(f"Mismatched {field}")
        if row.get("environment") != baseline.get("environment"):
            raise ValueError("Mismatched environment")
        speed = row.get("images_per_second")
        if type(speed) not in {int, float} or not math.isfinite(speed) or speed <= 0:
            raise ValueError("Invalid throughput")
        if type(row["images"]) is not int or row["images"] < 1:
            raise ValueError("Invalid images count")
        if (
            type(row.get("repeat")) is not int
            or row["repeat"] < 0
            or row.get("mode") not in MODES
        ):
            raise ValueError("Invalid repeat or mode")
        group = groups.setdefault(row["repeat"], {})
        if row["mode"] in group:
            raise ValueError("duplicate mode in repeat")
        group[row["mode"]] = row
    if sorted(groups) != list(range(len(groups))) or any(
        set(group) != set(MODES) for group in groups.values()
    ):
        raise ValueError("Incomplete repeat groups")
    return groups


def throughput_ratios(rows):
    groups = validate_reports(rows)
    pairs = {
        "parascale_eager_vs_native_eager": ("parascale-eager", "native-eager"),
        "native_compiled_vs_eager": ("native-compiled", "native-eager"),
        "parascale_compiled_vs_eager": ("parascale-compiled", "parascale-eager"),
        "parascale_compiled_vs_native_compiled": (
            "parascale-compiled",
            "native-compiled",
        ),
    }
    result = {}
    for name, (candidate, baseline) in pairs.items():
        samples = [
            group[candidate]["images_per_second"] / group[baseline]["images_per_second"]
            for _, group in sorted(groups.items())
        ]
        result[name] = {
            "samples": samples,
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
        }
    return result


def image_pair_metrics(left, right):
    import numpy as np
    from PIL import Image

    with Image.open(left) as image:
        a = np.asarray(image.convert("RGB"), dtype=np.float64)
    with Image.open(right) as image:
        b = np.asarray(image.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Mismatched image shape")
    difference = a - b
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    psnr = 10 * math.log10(255**2 / mse) if mse else None
    return {
        "exact": mse == 0,
        "mean_absolute_pixel_error": mae,
        "max_absolute_pixel_error": float(np.max(np.abs(difference))),
        "psnr_db": psnr,
        "passed": mse == 0 or (psnr >= 35 and mae <= 1.5),
    }


def compare_images(reference, candidate, *, exact):
    from parascale.workloads.causal_lm_assets import file_sha256

    manifests = [Path(row["manifest"]) for row in (reference, candidate)]
    records = [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in manifests
    ]
    if any(len(batch) != reference["images"] for batch in records):
        raise ValueError("Manifest count differs from measured images")
    errors = []
    for left, right in zip(*records):
        for key in ("id", "seed", "width", "height", "nsfw_content_detected"):
            if left[key] != right[key]:
                raise ValueError(f"Image identity/safety mismatch: {key}")
        paths = []
        for record, manifest in zip((left, right), manifests):
            path = (manifest.parent / record["file"]).resolve()
            if (
                not path.is_relative_to(manifest.parent.resolve())
                or file_sha256(path) != record["sha256"]
            ):
                raise ValueError("Invalid image path or checksum")
            paths.append(path)
        error = image_pair_metrics(*paths)
        error.update(id=left["id"], seed=left["seed"])
        if exact:
            error["passed"] = error["exact"]
        errors.append(error)
    return {
        "reference": reference["mode"],
        "candidate": candidate["mode"],
        "repeat": reference["repeat"],
        "passed": all(row["passed"] for row in errors),
        "images": errors,
    }


def configuration(args, mode, directory):
    return {
        "schema_version": 1,
        "model": {"pretrained_model_name_or_path": str(args.model.resolve())},
        "data": {"type": "prompts_jsonl", "path": str(args.prompts.resolve())},
        "runtime": {"device": "cuda:0", "run_dir": str(directory)},
        "inference": {
            "workload": "diffusion",
            "output_dir": str(directory / "images"),
            "precision": "fp16",
            "batch_size": args.batch_size,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": 7.5,
            "seed": 42,
            "warmup_steps": 1,
            "max_requests": args.requests,
            "compile_mode": "max-autotune" if mode.endswith("compiled") else "off",
        },
    }


def validate_native_assets(root: Path) -> None:
    """Reject custom component code before the independent Diffusers control loads."""
    index = root / "model_index.json"
    definition = (
        json.loads(index.read_text(encoding="utf-8")) if index.is_file() else None
    )
    if (
        not isinstance(definition, dict)
        or definition.get("_class_name") != "StableDiffusionPipeline"
    ):
        raise ValueError(
            "Native control requires a local StableDiffusionPipeline directory"
        )
    for name, component in definition.items():
        if (
            name.startswith("_")
            or not isinstance(component, list)
            or component == [None, None]
        ):
            continue
        if len(component) != 2 or not all(
            isinstance(value, str) for value in component
        ):
            raise ValueError(f"Invalid native diffusion component: {name}")
        library, _ = component
        folder = (root / name).resolve()
        if (
            library not in {"diffusers", "transformers", "stable_diffusion"}
            or not folder.is_relative_to(root.resolve())
            or (folder / f"{library}.py").is_file()
        ):
            raise ValueError(f"Unsupported custom native diffusion component: {name}")


def native_run(config):
    """Direct Diffusers control; never calls a ParaScale model or inference runner."""
    import diffusers
    import numpy as np
    import torch
    from PIL import Image

    from parascale.data.text.prompts import iter_prompt_batches
    from parascale.runtime.inference.diffusion import validate_diffusion_paths
    from parascale.workloads.causal_lm_assets import file_sha256
    from parascale.workloads.specs.diffusion import DiffusionInferenceSpec

    spec = DiffusionInferenceSpec.from_config(config)
    validate_diffusion_paths(spec)
    root = Path(spec.model_path)
    started = time.perf_counter()
    validate_native_assets(root)
    hashes = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".txt", ".safetensors"}
    }
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
        root, local_files_only=True, use_safetensors=True, torch_dtype=torch.float16
    ).to("cuda:0")
    pipeline.set_progress_bar_config(disable=True)
    for module in pipeline.components.values():
        if isinstance(module, torch.nn.Module):
            module.eval().requires_grad_(False)
    if spec.compile_mode != "off":
        pipeline.unet = torch.compile(
            pipeline.unet, mode=spec.compile_mode, fullgraph=True, dynamic=False
        )
    batches = list(
        iter_prompt_batches(
            spec.input_path,
            batch_size=spec.batch_size,
            max_line_bytes=spec.max_line_bytes,
            max_requests=spec.max_requests,
        )
    )
    if (
        not batches
        or sum(len(batch["prompts"]) for batch in batches) != spec.max_requests
    ):
        raise ValueError("Input has fewer than the requested benchmark prompts")
    ordinal = 0
    for batch in batches:
        batch["seeds"] = [
            42 + ordinal + index for index in range(len(batch["prompts"]))
        ]
        ordinal += len(batch["prompts"])
        tokens = pipeline.tokenizer(
            batch["prompts"] + [""], padding=False, truncation=False
        )["input_ids"]
        if any(len(row) > pipeline.tokenizer.model_max_length for row in tokens):
            raise ValueError("Benchmark prompt would be truncated")

    def generate(batch):
        with torch.inference_mode():
            result = pipeline(
                prompt=batch["prompts"],
                negative_prompt=[""] * len(batch["prompts"]),
                height=spec.height,
                width=spec.width,
                num_inference_steps=spec.num_inference_steps,
                guidance_scale=spec.guidance_scale,
                generator=[
                    torch.Generator(device="cpu").manual_seed(seed)
                    for seed in batch["seeds"]
                ],
                output_type="np",
                return_dict=True,
            )
        if (
            result.images.shape != (len(batch["prompts"]), spec.height, spec.width, 3)
            or not np.isfinite(result.images).all()
            or (result.images < 0).any()
            or (result.images > 1).any()
        ):
            raise ValueError("Invalid native image")
        return result

    torch.cuda.synchronize()
    setup = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    warmup = time.perf_counter()
    generate(batches[0])
    torch.cuda.synchronize()
    warmup = time.perf_counter() - warmup
    destination = Path(spec.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    elapsed = 0.0
    count = 0
    partial = destination / "manifest.jsonl.partial"
    with partial.open("x", encoding="utf-8") as stream:
        for batch in batches:
            torch.cuda.synchronize()
            before = time.perf_counter()
            result = generate(batch)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - before
            for index, pixels in enumerate(result.images):
                path = destination / f"{count:08d}.png"
                with path.open("xb") as target:
                    Image.fromarray((pixels * 255).round().astype("uint8")).save(
                        target, format="PNG"
                    )
                record = {
                    "id": batch["ids"][index],
                    "seed": batch["seeds"][index],
                    "width": spec.width,
                    "height": spec.height,
                    "file": path.name,
                    "sha256": file_sha256(path),
                    "nsfw_content_detected": bool(result.nsfw_content_detected[index])
                    if result.nsfw_content_detected is not None
                    else None,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                count += 1
    os.link(partial, destination / "manifest.jsonl")
    partial.unlink()
    return {
        "metrics": {
            "images": count,
            "images_per_second": count / elapsed,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        },
        "timing": {
            "setup_seconds": setup,
            "warmup_seconds": warmup,
            "job_wall_seconds": time.perf_counter() - started,
        },
        "execution": {
            "model_assets_sha256": hashes,
            "input_sha256": file_sha256(spec.input_path),
        },
        "output_stream": {"path": str(destination / "manifest.jsonl")},
    }


def worker(args):
    import diffusers
    import torch
    import transformers

    from parascale.cli import main as cli

    if (
        diffusers.__version__,
        transformers.__version__,
        torch.__version__.split("+")[0],
    ) != ("0.35.1", "4.51.3", "2.9.1"):
        raise RuntimeError(
            "Benchmark requires Diffusers 0.35.1 / Transformers 4.51.3 / Torch 2.9.1"
        )
    torch.cuda.set_device(0)
    directory = args.output.resolve()
    config = configuration(args, args.worker, directory)
    cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    cache_preexisting = cache.is_dir() and any(cache.iterdir())
    if args.worker.startswith("native"):
        payload = native_run(config)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        source = directory.parent / f"{directory.name}.config.json"
        source.write_text(json.dumps(config), encoding="utf-8")
        result = directory / "product-summary.json"
        if cli(["infer", "--config", str(source), "--output", str(result)]) != 0:
            raise RuntimeError("Product inference failed; partial evidence retained")
        payload = json.loads(result.read_text(encoding="utf-8"))
    if payload["metrics"]["images"] != args.requests:
        raise ValueError("Measured images differ from requested budget")
    settings = {
        key: value
        for key, value in config["inference"].items()
        if key not in {"workload", "output_dir", "compile_mode"}
    }
    report = {
        "mode": args.worker,
        "repeat": args.repeat_index,
        "settings": settings,
        "model_assets_sha256": payload["execution"]["model_assets_sha256"],
        "input_sha256": payload["execution"]["input_sha256"],
        "environment": {
            "torch": str(torch.__version__),
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "sdpa": {
                name: getattr(torch.backends.cuda, f"{name}_sdp_enabled")()
                for name in ("flash", "math", "mem_efficient", "cudnn")
            },
        },
        "images": payload["metrics"]["images"],
        "images_per_second": payload["metrics"]["images_per_second"],
        "peak_memory_bytes": payload["metrics"]["peak_memory_bytes"],
        "timing": payload["timing"],
        "manifest": payload["output_stream"]["path"],
        "compile_cache_preexisting": cache_preexisting,
    }
    (directory / "measurement.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--worker", choices=MODES)
    parser.add_argument("--repeat-index", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.requests <= 64 or args.repeats < 1:
        parser.error("requests must be 1..64 and repeats must be positive")
    if args.worker:
        worker(args)
        return 0
    from parascale.runtime.inference.diffusion import validate_diffusion_paths
    from parascale.workloads.specs.diffusion import DiffusionInferenceSpec

    output = args.output.resolve()
    validate_diffusion_paths(
        DiffusionInferenceSpec.from_config(configuration(args, "native-eager", output)),
        run_dir=output,
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    comparisons = []
    for repeat in range(args.repeats):
        for mode in MODES[repeat % 4 :] + MODES[: repeat % 4]:
            directory = output / f"r{repeat}-{mode}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--repeat-index",
                str(repeat),
                "--model",
                str(args.model.resolve()),
                "--prompts",
                str(args.prompts.resolve()),
                "--output",
                str(directory),
                "--requests",
                str(args.requests),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
            ]
            env = {
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": str(output / f"cache-{mode}"),
            }
            started = time.perf_counter()
            with (output / f"r{repeat}-{mode}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    env=env,
                    check=True,
                    timeout=1800,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            row = json.loads(
                (directory / "measurement.json").read_text(encoding="utf-8")
            )
            row["wall_seconds"] = time.perf_counter() - started
            rows.append(row)
            (output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                f"repeat={repeat} mode={mode} images/s={row['images_per_second']:.4f}",
                flush=True,
            )
        group = validate_reports(rows)[repeat]
        for baseline, candidate in (
            ("native-eager", "parascale-eager"),
            ("native-eager", "native-compiled"),
            ("parascale-eager", "parascale-compiled"),
            ("native-compiled", "parascale-compiled"),
        ):
            comparisons.append(
                compare_images(
                    group[baseline],
                    group[candidate],
                    exact=baseline == "native-eager" and candidate == "parascale-eager",
                )
            )
        if not all(comparison["passed"] for comparison in comparisons):
            (output / "comparison.json").write_text(
                json.dumps(
                    {
                        "status": "fidelity_failed",
                        "comparisons": comparisons,
                        "runs": rows,
                    },
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            return 1
    ratios = throughput_ratios(rows)
    eligible = (
        args.repeats >= 3
        and args.requests >= 8
        and min(args.height, args.width) >= 512
        and args.steps >= 30
    )
    gates = {
        "pixel_fidelity": all(row["passed"] for row in comparisons),
        "compiled_gain": ratios["parascale_compiled_vs_eager"]["median"] >= 1.10,
        "eager_native_parity": ratios["parascale_eager_vs_native_eager"]["median"]
        >= 0.95,
        "compiled_native_parity": ratios["parascale_compiled_vs_native_compiled"][
            "median"
        ]
        >= 0.95,
    }
    report = {
        "status": "smoke_only"
        if not eligible
        else "gates_passed"
        if all(gates.values())
        else "gates_failed",
        "scope": "fixed-prompt execution fidelity and paired performance, not semantic image quality",
        "gates": gates,
        "ratios": ratios,
        "comparisons": comparisons,
        "runs": rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return 0 if all(gates.values()) or not eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
