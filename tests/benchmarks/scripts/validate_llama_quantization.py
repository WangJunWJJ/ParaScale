# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Bounded, single-device reference quantization validation for HF Llama."""

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tiny", action="store_true", help="Offline random Llama smoke test"
    )
    source.add_argument("--model", help="Pretrained Llama local path or HF repository")
    result.add_argument("--revision", default="main")
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--calibration-text", type=Path, nargs="+")
    result.add_argument("--evaluation-text", type=Path, nargs="+")
    result.add_argument("--mode", choices=["ptq", "qat", "both"], default="both")
    result.add_argument("--bits", type=int, choices=[4, 8], default=8)
    result.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--seqlen", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--batches", type=int, default=4)
    result.add_argument("--qat-steps", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument(
        "--output",
        type=Path,
        default=Path("tests/benchmarks/reports/llama-quantization"),
    )
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_evidence(calibration, evaluation):
    if not calibration or not evaluation:
        raise ValueError(
            "Pretrained runs require calibration and evaluation text files"
        )

    def entries(paths):
        return [
            {
                "path": str(Path(p).resolve()),
                "sha256": file_hash(p),
                "bytes": Path(p).stat().st_size,
            }
            for p in paths
        ]

    train, test = entries(calibration), entries(evaluation)
    if {x["path"] for x in train} & {x["path"] for x in test} or {
        x["sha256"] for x in train
    } & {x["sha256"] for x in test}:
        raise ValueError(
            "Calibration/training and evaluation files must be distinct in path and content"
        )
    return {"calibration_and_qat_training": train, "evaluation": test}


def text_batches(paths, tokenizer, seqlen, batch_size, limit):
    """Tokenize fixed-size text chunks; never tokenize the entire corpus at once."""
    import torch

    pending = []
    emitted = 0
    width = seqlen * batch_size
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for text in iter(lambda: stream.read(8192), ""):
                pending.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
                while len(pending) >= width:
                    ids = torch.tensor(pending[:width], dtype=torch.long).reshape(
                        batch_size, seqlen
                    )
                    del pending[:width]
                    yield {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                    emitted += 1
                    if emitted == limit:
                        return
    if emitted < limit:
        raise ValueError(
            f"Text corpus has only {emitted} full batches; requested {limit}"
        )


def batch_hash(batch):
    return hashlib.sha256(batch["input_ids"].numpy().tobytes()).hexdigest()


def sync(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class PeakMemory:
    """CUDA allocator peak and optional sampled process RSS, not Python heap."""

    def __init__(self, device):
        self.device = device
        self.peak = None
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        import torch

        sync(self.device)
        self.start_allocated = None
        if self.device.type == "cuda":
            self.start_allocated = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            import psutil

            process = psutil.Process()
            self.peak = process.memory_info().rss

            def sample():
                while not self.stop.wait(0.01):
                    self.peak = max(self.peak, process.memory_info().rss)

            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        except ImportError:
            pass
        return self

    def __exit__(self, *exc):
        import torch

        sync(self.device)
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.result = {
            "process_rss_peak_sampled_bytes": self.peak,
            "rss_sampling_interval_ms": 10,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
        }
        self.result["cuda_start_allocated_bytes"] = self.start_allocated
        self.result["cuda_incremental_peak_bytes"] = (
            self.result["cuda_peak_allocated_bytes"] - self.start_allocated
            if self.start_allocated is not None
            else None
        )


def forward(model, batch, device):
    return model(**{k: v.to(device) for k, v in batch.items()}, use_cache=False).logits


def evaluate(model, batches, device, reference=None, save=None, parity=False):
    import torch
    import torch.nn.functional as F

    model.eval()
    loss_sum = 0.0
    tokens = elements = 0
    error_sum = error_max = 0.0
    with torch.no_grad():
        for index, batch in enumerate(batches()):
            logits = forward(model, batch, device).float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite evaluation logits")
            targets = batch["input_ids"][:, 1:].to(device)
            loss_sum += F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            ).item()
            tokens += targets.numel()
            cpu = logits.cpu()
            if reference is not None:
                expected = torch.load(reference / f"{index}.pt", weights_only=True)
                delta = (cpu - expected).abs()
                error_sum += delta.sum().item()
                elements += delta.numel()
                error_max = max(error_max, delta.max().item())
                if parity:
                    torch.testing.assert_close(cpu, expected, rtol=1e-5, atol=1e-5)
            if save is not None:
                torch.save(cpu, save / f"{index}.pt")
    loss = loss_sum / tokens
    perplexity = math.exp(loss)
    if not math.isfinite(loss) or not math.isfinite(perplexity):
        raise RuntimeError("Nonfinite loss/perplexity")
    return {
        "loss": loss,
        "perplexity": perplexity,
        "evaluated_tokens": tokens,
        "logits_mae": error_sum / elements if elements else 0.0,
        "logits_max_abs_error": error_max,
    }


def measure(model, batches, device, warmup, repeats):
    import torch

    batch = {k: v.to(device) for k, v in next(batches()).items()}
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            forward(model, batch, device)
        sync(device)
        # No surrounding PeakMemory scope: resetting allocator peaks is global.
        with PeakMemory(device) as memory:
            start = time.perf_counter()
            for _ in range(repeats):
                forward(model, batch, device)
            sync(device)
            elapsed = time.perf_counter() - start
    return {
        "seconds_per_forward": elapsed / repeats,
        "input_tokens_per_second": batch["input_ids"].numel() * repeats / elapsed,
        "warmup": warmup,
        "measured_forwards": repeats,
        "inference_memory": memory.result,
        "scope": "prefill forward only; excludes transfers, tokenization and loading",
    }


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def train_qat(model, batches, device, steps, learning_rate):
    import torch
    import torch.nn.functional as F

    # Hash quantized projection weights, not just embeddings or the excluded head.
    tracked = {
        name + ".weight": module.weight
        for name, module in model.named_modules()
        if hasattr(module, "weight_fake_quant") and hasattr(module, "weight")
    }
    if not tracked:
        raise RuntimeError("QAT prepared no quantized weights")
    before = {name: tensor_hash(value) for name, value in tracked.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    nonzero = 0
    stream = iter(batches())
    for _ in range(steps):
        batch = next(stream, None)
        if batch is None:
            stream = iter(batches())
            batch = next(stream)
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, batch, device).float()
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            batch["input_ids"][:, 1:].to(device).reshape(-1),
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite QAT loss")
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise RuntimeError("Nonfinite QAT gradient")
        if not any(
            p.grad is not None and torch.count_nonzero(p.grad) > 0
            for p in tracked.values()
        ):
            raise RuntimeError("QAT quantized weights have no nonzero gradients")
        nonzero += 1
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite QAT parameter")
    changed = sum(tensor_hash(value) != before[name] for name, value in tracked.items())
    if not changed:
        raise RuntimeError("QAT quantized weights did not change")
    optimizer.zero_grad(set_to_none=True)
    return {
        "steps": steps,
        "nonzero_gradient_steps": nonzero,
        "changed_parameters": changed,
    }


def model_bytes(model):
    return sum(
        value.numel() * value.element_size() for value in model.state_dict().values()
    )


def run(args):
    if (
        args.seqlen < 2
        or min(args.batch_size, args.batches, args.qat_steps, args.repeats) < 1
    ):
        raise ValueError("seqlen must be >= 2 and batch/step/repeat counts positive")
    if (
        args.warmup < 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        raise ValueError("Invalid warmup or learning rate")
    evidence = (
        {"kind": "synthetic smoke only"}
        if args.tiny
        else corpus_evidence(args.calibration_text, args.evaluation_text)
    )
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from parascale.quantization import (
        PostTrainingQuantization,
        QuantizationAwareTraining,
        QuantizationConfig,
        load_quantized_model,
    )

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA reference validation is supported")
    torch.manual_seed(args.seed)
    source_evidence = {
        "kind": "random_llama" if args.tiny else "pretrained",
        "model": args.model,
        "requested_revision": None if args.tiny else args.revision,
        "resolved_model_commit": None,
        "resolved_tokenizer_commit": None,
    }
    if args.tiny:
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=max(128, args.seqlen),
            attention_dropout=0.0,
        )

        def factory():
            torch.manual_seed(args.seed)
            return LlamaForCausalLM(config).float()

        def batches(seed):
            generator = torch.Generator().manual_seed(seed)
            for _ in range(args.batches):
                ids = torch.randint(
                    3, 64, (args.batch_size, args.seqlen), generator=generator
                )
                yield {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def training():
            return batches(args.seed + 1)

        def evaluation():
            return batches(args.seed + 2)
    else:
        options = {
            "revision": args.revision,
            "local_files_only": args.local_files_only,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
        source_evidence["resolved_tokenizer_commit"] = tokenizer.init_kwargs.get(
            "_commit_hash"
        )
        source_evidence["tokenizer_name_or_path"] = tokenizer.name_or_path

        def factory():
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float32, **options
            )
            if model.config.model_type != "llama":
                raise ValueError("Validation requires a Llama model")
            return model

        def training():
            return text_batches(
                args.calibration_text,
                tokenizer,
                args.seqlen,
                args.batch_size,
                args.batches,
            )

        def evaluation():
            return text_batches(
                args.evaluation_text,
                tokenizer,
                args.seqlen,
                args.batch_size,
                args.batches,
            )

    train_hashes = [batch_hash(b) for b in training()]
    eval_hashes = [batch_hash(b) for b in evaluation()]
    if set(train_hashes) & set(eval_hashes):
        raise ValueError("Training and evaluation token batches must be distinct")
    evidence.update(
        training_batch_sha256=train_hashes, evaluation_batch_sha256=eval_hashes
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "compute": "reference floating compute; packed weights do not promise speedup",
        "arguments": {
            k: str(v)
            if isinstance(v, Path)
            else [str(p) for p in v]
            if isinstance(v, list)
            else v
            for k, v in vars(args).items()
        },
        "data": evidence,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "device": str(device),
        "dtype": "float32",
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "cuda_runtime": torch.version.cuda,
        "model_source": source_evidence,
    }
    with tempfile.TemporaryDirectory(prefix="llama-validation-") as temporary:
        reference = Path(temporary) / "baseline"
        reference.mkdir()
        with PeakMemory(device) as memory:
            model = factory().to(device)
            report["model_config"] = model.config.to_dict()
            source_evidence["resolved_model_commit"] = getattr(
                model.config, "_commit_hash", None
            )
        baseline_loading_memory = memory.result
        report["baseline"] = evaluate(model, evaluation, device, save=reference)
        report["baseline"]["model_state_bytes"] = model_bytes(model)
        with PeakMemory(device) as memory:
            baseline_checkpoint = Path(temporary) / "baseline.pt"
            torch.save(model.state_dict(), baseline_checkpoint)
            report["baseline"]["checkpoint_bytes"] = baseline_checkpoint.stat().st_size
            report["baseline"]["checkpoint_sha256"] = file_hash(baseline_checkpoint)
            baseline_checkpoint.unlink()
        report["baseline"]["serialization_memory"] = memory.result
        report["baseline"]["loading_memory"] = baseline_loading_memory
        cleanup()
        report["baseline"]["timing"] = measure(
            model, evaluation, device, args.warmup, args.repeats
        )
        del model
        cleanup()
        modes = ["ptq", "qat"] if args.mode == "both" else [args.mode]
        for mode in modes:
            checkpoint = args.output / f"{mode}-{args.bits}bit.pt"
            saved = Path(temporary) / mode
            saved.mkdir()
            with PeakMemory(device) as memory:
                source = factory()
                quant_config = QuantizationConfig(
                    enabled=True,
                    mode=mode,
                    bits=args.bits,
                    fuse_modules=False,
                    per_channel=True,
                    quantize_activations=False,
                    exclude_modules=["lm_head"],
                    calib_batches=args.batches,
                )
                engine = (
                    PostTrainingQuantization
                    if mode == "ptq"
                    else QuantizationAwareTraining
                )(source, quant_config)
                prepared = engine.prepare().to(device)
                if prepared is source:
                    raise RuntimeError("prepare must return an independent model")
                if mode == "ptq":
                    engine.calibrate(
                        [{k: v.to(device) for k, v in b.items()} for b in training()]
                    )
                    training_result = None
                else:
                    training_result = train_qat(
                        prepared, training, device, args.qat_steps, args.learning_rate
                    )
                    engine.freeze_observer()
                model = engine.convert().to(device).eval()
                engine.export(str(checkpoint))
                del engine, prepared, source
                cleanup()
            procedure_memory = memory.result
            result = evaluate(
                model, evaluation, device, reference=reference, save=saved
            )
            cleanup()
            result.update(
                model_state_bytes=model_bytes(model),
                checkpoint_bytes=checkpoint.stat().st_size,
                checkpoint_sha256=file_hash(checkpoint),
                timing=measure(model, evaluation, device, args.warmup, args.repeats),
            )
            if training_result is not None:
                result["training"] = training_result
            del model
            cleanup()
            result["procedure_memory"] = procedure_memory
            template = factory()
            loaded, loaded_config, metadata = load_quantized_model(
                str(checkpoint), model=template
            )
            del template
            loaded.to(device).eval()
            if loaded_config.bits != args.bits or loaded_config.mode != mode:
                raise RuntimeError("Reloaded quantization config mismatch")
            parity = evaluate(loaded, evaluation, device, reference=saved, parity=True)
            result["roundtrip_max_abs_error"] = parity["logits_max_abs_error"]
            result["roundtrip_metadata_type"] = type(metadata).__name__
            report[mode] = result
            del loaded
            cleanup()
    report["status"] = "passed"
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main():
    args = parser().parse_args()
    try:
        report = run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "compute": "reference floating compute; no promised speedup",
        }
        (args.output / "report.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
