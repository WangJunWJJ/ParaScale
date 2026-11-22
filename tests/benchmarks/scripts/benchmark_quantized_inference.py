# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real pretrained causal-model BF16/INT4 comparison; no synthetic accuracy.

Run --mode compare to isolate models in separate child processes. Supply a local
pretrained Hugging Face checkpoint and a held-out UTF-8 corpus. Outputs include
all timing samples, hashes, environment, quality and an engineering gate result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(folder):
    files = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and (
            p.suffix in (".json", ".safetensors", ".model", ".txt")
            or p.name.startswith("pytorch_model")
            and p.suffix == ".bin"
        )
        and ".cache" not in p.relative_to(folder).parts
    )
    if not any(p.suffix in (".bin", ".safetensors") for p in files):
        raise ValueError("--model must contain pretrained model weights")
    manifest = {str(p.relative_to(folder)): sha256(p) for p in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest, digest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--mode", choices=("compare", "bf16", "int4", "int8"), default="compare"
    )
    result.add_argument("--model", type=Path, required=True)
    result.add_argument(
        "--reference-artifact",
        type=Path,
        help="Canonical INT8 PTQ/QAT checkpoint; compare dense BF16 vs fused same weights",
    )
    result.add_argument("--data", type=Path, required=True)
    result.add_argument(
        "--data-description", required=True, help="Corpus origin and evaluation split"
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--sequence-length", type=int, default=128)
    result.add_argument("--new-tokens", type=int, default=32)
    result.add_argument("--eval-tokens", type=int, default=32768)
    result.add_argument("--eval-offset", type=int, default=0)
    result.add_argument(
        "--group-size", type=int, choices=(32, 64, 128, 256), default=128
    )
    result.add_argument("--exclude-modules", nargs="+", default=["lm_head"])
    result.add_argument(
        "--backend", choices=("torchao_int4", "gemlite_int8"), default="torchao_int4"
    )
    result.add_argument("--static-cache", action="store_true")
    result.add_argument(
        "--gemlite-gemv-autotune",
        choices=("fast", "max"),
        help="Experiment only: reset backend tuning cache and search GEMV candidates",
    )
    result.add_argument(
        "--export-profile-trace",
        action="store_true",
        help="Export one diagnostic generation trace outside throughput timing",
    )
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--compile",
        action="store_true",
        help="Compile forward equally for both variants",
    )
    return result


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def configure_gemlite_tuning(mode: str | None) -> dict | None:
    """Configure an isolated worker's GEMV experiment before model preparation.

    None preserves backend defaults. Explicit fast/max uses the same empty
    initial cache for all modes; only the GEMV search space changes. Raises
    ValueError for an unsupported mode or RuntimeError for an unqualified backend.
    """
    if mode is None:
        return None
    if mode not in ("fast", "max"):
        raise ValueError("Unsupported GemLite GEMV tuning mode")
    from parascale.quantization.inference_gemlite import _load_gemlite

    _load_gemlite()
    import gemlite

    gemlite.reset_config()
    gemlite.set_autotune({"GEMV_SPLITK": mode})
    return {"gemv_splitk": mode, "initial_cache": "empty_all_modes"}


def gemlite_tuning_evidence(required_shapes: set[tuple[int, int]]) -> dict:
    """Validate observed M1 INT8 K/N winners; reject missing tuning evidence.

    Exported configurations cover warmup, timing, profiling and quality calls,
    not only the throughput measurement window. Collection is outside timing.
    """
    import gemlite

    from parascale.quantization.inference_gemlite import _validate_kernel_cache

    with tempfile.TemporaryDirectory(prefix="parascale-gemlite-tuning-") as folder:
        path = Path(folder) / "config.json"
        gemlite.cache_config(str(path))
        configs = json.loads(path.read_text(encoding="utf-8"))
        _validate_kernel_cache(configs)
        observed = set()
        for signature in configs.get("GEMV_SPLITK", {}):
            rows, outputs, features, group, elements, dtype_id = ast.literal_eval(
                signature
            )
            if (rows, elements, dtype_id) == (1, 1, 108) and group == features:
                observed.add((features, outputs))
        if not required_shapes or not required_shapes.issubset(observed):
            raise RuntimeError("Missing GEMV tuning evidence for converted shapes")
        return {
            "config_sha256": sha256(path),
            "winning_configs": configs,
            "scope": "warmup, timing, profiling and quality calls",
            "required_gemv_shapes_kn": sorted(required_shapes),
        }


def run_comparison(args):
    from quantized_inference_report import compare_results

    results = {}
    candidate_mode = "int8" if args.backend == "gemlite_int8" else "int4"
    for mode in ("bf16", candidate_mode):
        target = args.output.with_name(args.output.stem + f".{mode}.json")
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode]
        for name, value in vars(args).items():
            if name in ("mode", "output") or value is None:
                continue
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif isinstance(value, list):
                command.extend([flag, *value])
            else:
                command.extend([flag, str(value)])
        command.extend(["--output", str(target)])
        # A failed worker cannot be mistaken for success or an old result.
        subprocess.run(command, cwd=ROOT, check=True)
        results[mode] = json.loads(target.read_text(encoding="utf-8"))
    summary = compare_results(
        results["bf16"],
        results[candidate_mode],
        max_ppl_increase=0.01 if args.reference_artifact else 0.05,
    )
    write_json(args.output, {"summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


def measure(torch, function, args):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.warmup):
        output = function()
        del output
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wall, device, trials = [], [], []
    for repeat in range(args.repeats):
        trial = []
        for _ in range(args.iterations):
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            begin.record()
            output = function()
            end.record()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            wall.append(elapsed)
            trial.append(elapsed)
            device.append(begin.elapsed_time(end))
            del output
        trials.append({"repeat": repeat, "median_wall_ms": statistics.median(trial)})
    return {
        "wall_ms": wall,
        "cuda_ms": device,
        "trials": trials,
        "median_wall_ms": statistics.median(wall),
        "p95_wall_ms": sorted(wall)[math.ceil(len(wall) * 0.95) - 1],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "warmup_including_compile_seconds": warmup_seconds,
    }


def compile_generation_forward(model):
    import torch

    # Dynamic KV cache tensors survive across forward calls. CUDA Graph replay
    # can overwrite them; use fusion/autotuning without graph-owned output reuse.
    model.forward = torch.compile(
        model.forward,
        mode="max-autotune-no-cudagraphs",
        fullgraph=False,
    )


def fixed_length_generation(
    model,
    prompt,
    attention,
    new_tokens,
    pad_token_id,
    *,
    static_cache=False,
    compile_static=False,
):
    from transformers import CompileConfig, GenerationConfig

    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        return_dict_in_generate=False,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        eos_token_id=None,
        forced_eos_token_id=None,
        pad_token_id=pad_token_id,
        use_cache=True,
        cache_implementation="static" if static_cache else None,
        disable_compile=not compile_static,
        compile_config=CompileConfig(mode="max-autotune", fullgraph=True),
    )
    # Transformers 4.51 exposes this switch; newer APIs may omit it.
    defaults = (
        {"use_model_defaults": False}
        if "use_model_defaults" in signature(model.generate).parameters
        else {}
    )
    # Explicit None kwargs prevent EOS defaults from being filled by the model.
    return model.generate(
        input_ids=prompt,
        attention_mask=attention,
        generation_config=config,
        **defaults,
        do_sample=False,
        eos_token_id=None,
        forced_eos_token_id=None,
    )


def run_worker(args):
    import torch
    import torch.nn.functional as functional
    from quantized_inference_report import quantized_cuda_kernels, summarize_profile
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from parascale.quantization import (
        InferenceQuantizationConfig,
        prepare_quantized_inference_model,
        quantize_inference_model,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("This benchmark requires a CUDA SM80+ GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    tuning_policy = configure_gemlite_tuning(args.gemlite_gemv_autotune)
    args.model, args.data = args.model.resolve(), args.data.resolve()
    manifest, model_hash = model_manifest(args.model)
    data_hash = sha256(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    text = args.data.read_text(encoding="utf-8")
    tokens = tokenizer(
        text, add_special_tokens=False, return_tensors="pt", verbose=False
    )["input_ids"][0]
    if tokens.numel() < max(
        args.sequence_length * args.batch_size, args.eval_offset + args.eval_tokens
    ):
        raise ValueError(
            "Evaluation corpus has too few tokens; use a larger corpus or smaller --eval-tokens"
        )
    evaluation = tokens[args.eval_offset : args.eval_offset + args.eval_tokens]
    prompt = (
        tokens[: args.sequence_length * args.batch_size]
        .reshape(args.batch_size, args.sequence_length)
        .cuda()
    )
    attention = torch.ones_like(prompt)
    started = time.perf_counter()
    artifact_contract = None
    if args.reference_artifact:
        from quantized_artifact_fixture import (
            load_reference_artifact,
            materialize_reference_weights,
        )

        artifact_hash = sha256(args.reference_artifact)
        model, artifact_config = load_reference_artifact(
            args.reference_artifact, args.model
        )
        if sha256(args.reference_artifact) != artifact_hash:
            raise RuntimeError("Reference artifact changed during loading")
        artifact_contract = {
            "path": str(args.reference_artifact),
            "sha256": artifact_hash,
            "config": artifact_config,
            "float_control": "dequantized_once_bf16",
        }
        model = model.eval().cuda().bfloat16()
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .eval()
            .cuda()
        )
    model.requires_grad_(False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    conversion, conversion_seconds = None, 0.0
    tuning_shapes = set()
    if args.mode in ("int4", "int8"):
        started = time.perf_counter()
        if args.reference_artifact:
            model, conversion = prepare_quantized_inference_model(model, inplace=True)
        else:
            model, conversion = quantize_inference_model(
                model,
                InferenceQuantizationConfig(
                    group_size=args.group_size,
                    exclude_modules=args.exclude_modules,
                    backend=args.backend,
                ),
                inplace=True,
            )
        torch.cuda.synchronize()
        conversion_seconds = time.perf_counter() - started
        if tuning_policy is not None:
            from parascale.quantization.inference_gemlite import _requires_safe_gemm

            for name in conversion["converted_layers"]:
                layer = model.get_submodule(name)
                if not _requires_safe_gemm(layer.in_features, layer.out_features, 1):
                    tuning_shapes.add((layer.in_features, layer.out_features))
    elif args.reference_artifact:
        started = time.perf_counter()
        model = materialize_reference_weights(model)
        torch.cuda.synchronize()
        conversion_seconds = time.perf_counter() - started
    # Profile the eager path outside timing to confirm packed-kernel dispatch.
    kernel_names = []
    with torch.inference_mode():
        if args.mode in ("int4", "int8"):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as trace:
                profiled = model(
                    input_ids=prompt, attention_mask=attention, use_cache=False
                )
                del profiled
            kernel_names = sorted(
                {event.key for event in trace.key_averages() if "int4" in event.key}
            )
        if args.compile and not args.static_cache:
            compile_generation_forward(model)

        def prefill():
            return model(input_ids=prompt, attention_mask=attention, use_cache=True)

        def generate():
            output = fixed_length_generation(
                model,
                prompt,
                attention,
                args.new_tokens,
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                static_cache=args.static_cache,
                compile_static=args.compile and args.static_cache,
            )
            if output.shape[-1] != args.sequence_length + args.new_tokens:
                raise RuntimeError("Generation length mismatch invalidates comparison")
            return output

        timings = {
            "prefill": measure(torch, prefill, args),
            "generation": measure(torch, generate, args),
        }
        generated = generate().cpu().tolist()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as generation_trace:
            profile_started = time.perf_counter()
            profiled_generation = generate()
            torch.cuda.synchronize()
            profile_wall_ms = (time.perf_counter() - profile_started) * 1000
            del profiled_generation
        profile_summary = summarize_profile(
            generation_trace.key_averages(), wall_ms=profile_wall_ms
        )
        packed_cuda_kernels = quantized_cuda_kernels(
            generation_trace.key_averages(), backend=args.backend
        )
        if args.export_profile_trace:
            trace_path = args.output.with_suffix(".trace.json")
            if trace_path.exists():
                raise FileExistsError(
                    f"Refusing to replace profile trace: {trace_path}"
                )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            generation_trace.export_chrome_trace(str(trace_path))
            profile_summary["trace_path"] = str(trace_path)
            profile_summary["trace_sha256"] = sha256(trace_path)
        generation_events = sorted(
            {event.key for event in generation_trace.key_averages()}
        )
        generation_int4_ops = [
            name
            for name in generation_events
            if "int4" in name.lower()
            or "gemlite" in name.lower()
            or "gemv_INT" in name
            or "gemm_INT" in name
        ]
        compiled_regions = [
            name for name in generation_events if "compiled" in name.lower()
        ]
        # Non-overlapping fixed-context blocks. Same tokens for both variants;
        # the first token of each block has no preceding context and is omitted.
        total_nll, count = 0.0, 0
        for offset in range(0, evaluation.numel() - 1, args.sequence_length):
            block = (
                evaluation[offset : offset + args.sequence_length].unsqueeze(0).cuda()
            )
            if block.shape[-1] < 2:
                continue
            logits = model(input_ids=block, use_cache=False).logits
            nll = functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                block[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            count += block.shape[-1] - 1
        perplexity = math.exp(total_nll / count)
        if not math.isfinite(perplexity):
            raise RuntimeError("Nonfinite perplexity; accuracy validation failed")
    packages = {}
    for package in (
        "torch",
        "torchao",
        "transformers",
        "tokenizers",
        "gemlite",
        "triton",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    props = torch.cuda.get_device_properties(0)
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_files = [
        Path(__file__),
        Path(__file__).with_name("quantized_inference_report.py"),
        ROOT / "parascale/quantization/inference.py",
        ROOT / "parascale/quantization/inference_config.py",
        ROOT / "parascale/quantization/inference_gemlite.py",
    ]
    source_files.extend(sorted((ROOT / "parascale/quantization").glob("*.py")))
    source_files.append(Path(__file__).with_name("quantized_artifact_fixture.py"))
    result = {
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "model_path": str(args.model),
            "model_sha256": model_hash,
            "reference_artifact": artifact_contract,
            "data_path": str(args.data),
            "data_sha256": data_hash,
            "data_description": args.data_description,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "new_tokens": args.new_tokens,
            "eval_tokens_requested": args.eval_tokens,
            "eval_offset": args.eval_offset,
            "seed": args.seed,
            "compile": args.compile,
            "static_cache": args.static_cache,
            "gemlite_tuning": tuning_policy,
            "quality_execution": (
                "eager" if args.static_cache or not args.compile else "compiled_forward"
            ),
            "prefill_measurement": "separate forward with dynamic cache; generation includes its own prefill",
            "compile_mode": (
                (
                    "max-autotune static decode"
                    if args.static_cache
                    else "max-autotune-no-cudagraphs"
                )
                if args.compile
                else None
            ),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "activation_dtype": "bfloat16",
            "generation_policy": "greedy_fixed_length_no_model_defaults",
            "attention": "sdpa",
            "tf32": False,
            "source_sha256": {
                str(p.relative_to(ROOT)): sha256(p) for p in source_files
            },
        },
        "environment": {
            **packages,
            "python": platform.python_version(),
            "os": platform.platform(),
            "hostname": platform.node(),
            "cuda": torch.version.cuda,
            "gpu": props.name,
            "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "gpu_memory_bytes": props.total_memory,
            "gpu_inventory": inventory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model_files": manifest,
        "quality": {
            "perplexity": perplexity,
            "nll_sum": total_nll,
            "tokens": count,
            "generated_token_ids": generated,
        },
        "timing": timings,
        "conversion": conversion,
        "eager_kernel_verified": any(
            "_weight_int4pack_mm" in name for name in kernel_names
        ),
        "eager_profiled_int4_ops": kernel_names,
        "kernel_verified": bool(packed_cuda_kernels),
        "profiled_quantized_cuda_kernels": packed_cuda_kernels,
        "profiled_int4_ops": generation_int4_ops,
        "profiled_compiled_regions": compiled_regions,
        "generation_profile": profile_summary,
        "gemlite_tuning_evidence": (
            gemlite_tuning_evidence(tuning_shapes)
            if tuning_policy is not None and args.mode == "int8"
            else None
        ),
        "load_seconds": load_seconds,
        "conversion_seconds": conversion_seconds,
        "generation_tokens_per_second": args.batch_size
        * args.new_tokens
        * 1000
        / timings["generation"]["median_wall_ms"],
    }
    write_json(args.output, result)
    print(f"Saved {args.mode} measurements: {args.output}", flush=True)


def main():
    args = parser().parse_args()
    if args.gemlite_gemv_autotune is not None and args.backend != "gemlite_int8":
        raise ValueError("--gemlite-gemv-autotune requires --backend gemlite_int8")
    if args.gemlite_gemv_autotune is not None and args.batch_size != 1:
        raise ValueError("--gemlite-gemv-autotune requires --batch-size 1")
    expected_mode = "int8" if args.backend == "gemlite_int8" else "int4"
    if args.mode in ("int4", "int8") and args.mode != expected_mode:
        raise ValueError(
            "--mode and --backend must select the same quantization precision"
        )
    # Workers run from the repository root; preserve the caller's path meaning.
    args.model = args.model.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if args.reference_artifact:
        args.reference_artifact = args.reference_artifact.resolve()
        if args.backend != "gemlite_int8":
            raise ValueError("--reference-artifact requires --backend gemlite_int8")
        if args.exclude_modules != ["lm_head"] or args.group_size != 128:
            raise ValueError(
                "Reference artifacts define their own layer coverage and scales"
            )
    for key in ("batch_size", "new_tokens", "warmup", "iterations", "repeats"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.sequence_length < 2 or args.eval_tokens < 2:
        raise ValueError("sequence length and evaluation tokens must be >= 2")
    if args.eval_offset < 0:
        raise ValueError("--eval-offset must be nonnegative")
    if args.iterations * args.repeats < 20:
        raise ValueError("At least 20 timing samples are required")
    try:
        if args.mode == "compare":
            run_comparison(args)
        else:
            run_worker(args)
    except Exception as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        failure_path = args.output.with_name(f"{args.output.stem}.failure-{stamp}.json")
        write_json(
            failure_path,
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
        )
        raise


if __name__ == "__main__":
    main()
