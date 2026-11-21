# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午11:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Matched observer A/B diagnostic, not pretrained quality or production throughput."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def tensor_digest(items):
    result = hashlib.sha256()
    for name, tensor in sorted(items):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def run(args):
    if args.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from parascale.quantization import QuantizationAwareTraining, QuantizationConfig
    from parascale.quantization.observers import BaseObserver

    if min(args.widths + [args.steps, args.repeats]) < 1 or args.warmup < 0:
        raise ValueError("widths/steps/repeats must be positive and warmup nonnegative")
    if any(width % 4 for width in args.widths):
        raise ValueError("widths must be divisible by four attention heads")
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_path = args.baseline_source / "parascale/quantization/observers.py"
    spec = importlib.util.spec_from_file_location("baseline_observers", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    current = BaseObserver._range
    implementations = {"baseline": baseline.BaseObserver._range, "candidate": current}
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    try:
        for width in args.widths:
            for repeat in range(args.repeats):
                order = ("baseline", "candidate")
                if repeat % 2:
                    order = tuple(reversed(order))
                for variant in order:
                    BaseObserver._range = implementations[variant]
                    torch.manual_seed(42)
                    config = LlamaConfig(
                        vocab_size=512,
                        hidden_size=width,
                        intermediate_size=width * 4,
                        num_hidden_layers=4,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        attention_dropout=0.0,
                    )
                    source = LlamaForCausalLM(config)
                    controller = QuantizationAwareTraining(
                        source,
                        QuantizationConfig(
                            enabled=True,
                            mode="qat",
                            bits=4,
                            per_channel=True,
                            exclude_modules=["lm_head"],
                            quantize_activations=False,
                        ),
                    )
                    model = controller.prepare().to(device).train()
                    del source
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
                    inputs = torch.randint(3, 512, (4, 128), device=device)
                    times = []
                    for index in range(args.steps + args.warmup):
                        synchronize()
                        started = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        loss = model(
                            input_ids=inputs, labels=inputs, use_cache=False
                        ).loss
                        loss.backward()
                        optimizer.step()
                        synchronize()
                        if index >= args.warmup:
                            times.append((time.perf_counter() - started) * 1000)
                    parameters = list(model.named_parameters())
                    if not all(torch.isfinite(p).all() for _, p in parameters):
                        raise RuntimeError("Nonfinite QAT parameters")
                    gradients = [
                        (name, p.grad) for name, p in parameters if p.grad is not None
                    ]
                    if not gradients or not all(
                        torch.isfinite(g).all() for _, g in gradients
                    ):
                        raise RuntimeError("Missing or nonfinite QAT gradients")
                    row = {
                        "width": width,
                        "repeat": repeat,
                        "variant": variant,
                        "loss": float(loss.detach()),
                        "step_ms": times,
                        "median_ms": statistics.median(times),
                        "samples_per_second": 4000 / statistics.median(times),
                        "state_sha256": tensor_digest(model.state_dict().items()),
                        "gradient_sha256": tensor_digest(gradients),
                        "input_sha256": tensor_digest([("input_ids", inputs)]),
                    }
                    rows.append(row)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "step_ms"}),
                        flush=True,
                    )
                    del (
                        model,
                        controller,
                        optimizer,
                        loss,
                        inputs,
                        parameters,
                        gradients,
                    )
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        BaseObserver._range = current
    summary = []
    for width in args.widths:
        group = [row for row in rows if row["width"] == width]
        for key in ("state_sha256", "gradient_sha256", "input_sha256"):
            if len({row[key] for row in group}) != 1:
                raise RuntimeError(f"A/B parity failed: width={width}, {key}")
        rates = {
            variant: statistics.median(
                [
                    row["samples_per_second"]
                    for row in group
                    if row["variant"] == variant
                ]
            )
            for variant in implementations
        }
        summary.append(
            {"width": width, **rates, "speedup": rates["candidate"] / rates["baseline"]}
        )
    payload = {
        "scope": "Synthetic FP32 QAT forward/backward plus SGD; excludes data loading and quality evaluation",
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_sha256": {
            "baseline": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "candidate": hashlib.sha256(
                (ROOT / "parascale/quantization/observers.py").read_bytes()
            ).hexdigest(),
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())
