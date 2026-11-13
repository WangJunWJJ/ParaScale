# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Bounded FP32, one-process Llama distillation verification (not a trainer)."""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "llama_quantization_helpers",
    Path(__file__).with_name("validate_llama_quantization.py"),
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true")
    source.add_argument("--teacher-model")
    result.add_argument("--student-model")
    result.add_argument("--teacher-revision", default="main")
    result.add_argument("--student-revision", default="main")
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--calibration-text", "--training-text", type=Path, nargs="+")
    result.add_argument("--evaluation-text", type=Path, nargs="+")
    result.add_argument("--device", default="cpu")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--seqlen", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--batches", type=int, default=4)
    result.add_argument("--steps", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--temperature", type=float, default=2.0)
    result.add_argument("--alpha", type=float, default=0.5)
    result.add_argument("--output", type=Path, required=True)
    return result


def check_tokenizers(teacher, student, vocab_size):
    """Require token identity, not merely matching logit dimensions."""
    vocab = teacher.get_vocab()
    if vocab != student.get_vocab():
        raise ValueError("Teacher/student tokenizer token-to-ID mapping differs")
    if teacher.special_tokens_map != student.special_tokens_map:
        raise ValueError("Teacher/student tokenizer special-token mapping differs")
    if set(vocab.values()) != set(range(vocab_size)):
        raise ValueError("Tokenizer mapping does not cover model vocabulary exactly")
    return hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def evaluate(student, teacher, batches, device):
    """Independent FP32 next-token CE and KL(P_teacher || P_student), T=1."""
    import torch
    import torch.nn.functional as F

    student.eval()
    teacher.eval()
    ce, kl, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in batches:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            valid = mask[:, :-1].bool() & mask[:, 1:].bool()
            s = student(input_ids=ids, attention_mask=mask, use_cache=False).logits
            t = teacher(input_ids=ids, attention_mask=mask, use_cache=False).logits
            s, t = s[:, :-1].float()[valid], t[:, :-1].float()[valid]
            ce += F.cross_entropy(s, ids[:, 1:][valid], reduction="sum").item()
            kl += F.kl_div(s.log_softmax(-1), t.softmax(-1), reduction="sum").item()
            count += int(valid.sum())
    ce /= count
    return {
        "ce": ce,
        "perplexity": math.exp(ce),
        "kl_teacher_to_student": kl / count,
        "valid_count": count,
    }


def train(student, teacher, batches, args, device, initial, kd):
    import torch

    from parascale.distillation import DistillationConfig, OnlineDistillationLoss

    config = DistillationConfig(
        enabled=True,
        task="causal_lm",
        temperature=args.temperature,
        alpha=args.alpha,
        ignore_index=-100,
        input_keys=["input_ids", "attention_mask"],
    )
    objective = OnlineDistillationLoss(student, teacher, config) if kd else None
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    student.train()
    torch.manual_seed(args.seed + 3)
    times, losses, nonzero = [], [], 0
    with _helpers.PeakMemory(device) as memory:
        for step in range(args.steps):
            batch = {k: v.to(device) for k, v in batches[step % len(batches)].items()}
            batch["labels"] = batch["input_ids"].clone()
            _helpers.sync(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            loss = (
                objective(output, batch)
                if kd
                else torch.nn.functional.cross_entropy(
                    output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                    batch["labels"][:, 1:].reshape(-1),
                )
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            grads = [p.grad for p in student.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError("Missing or nonfinite student gradients")
            nonzero += int(any(torch.count_nonzero(g).item() for g in grads))
            optimizer.step()
            _helpers.sync(device)
            times.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
    changed = sum(
        not torch.equal(p.detach().cpu(), initial[n])
        for n, p in student.named_parameters()
    )
    if not changed or nonzero != args.steps:
        raise RuntimeError("Student did not receive real gradients and updates")
    return {
        "losses": losses,
        "nonzero_gradient_steps": nonzero,
        "changed_parameters": changed,
        "step_seconds": times,
        "mean_step_seconds": sum(times) / len(times),
        "timing_scope": "No warmup; forward, loss, backward, checks, optimizer; excludes batch transfer",
        "memory": memory.result,
        "memory_scope": "Training loop; teacher and active student resident; initial weights on CPU; RSS is whole process",
        "last_metrics": {k: float(v) for k, v in objective.last_metrics.items()}
        if objective
        else {},
    }


def run(args):
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
    )

    if min(args.batch_size, args.batches, args.steps) < 1 or args.seqlen < 2:
        raise ValueError("Positive batch sizes/steps and seqlen >= 2 required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Positive finite learning rate required")
    if (
        not math.isfinite(args.temperature)
        or args.temperature <= 0
        or not 0 <= args.alpha <= 1
    ):
        raise ValueError("Positive finite temperature and alpha in [0,1] required")
    if args.tiny and args.student_model:
        raise ValueError("--student-model cannot be combined with --tiny")
    evidence = {}
    if not args.tiny:
        evidence = _helpers.corpus_evidence(args.calibration_text, args.evaluation_text)
        if not args.student_model:
            raise ValueError("Pretrained runs require --student-model")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu/cuda devices supported")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Output directory must be empty; use a fresh output directory")
    torch.manual_seed(args.seed)
    if args.tiny:

        def model(hidden, layers):
            return LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=64,
                    hidden_size=hidden,
                    intermediate_size=hidden * 2,
                    num_hidden_layers=layers,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=max(64, args.seqlen),
                    attention_dropout=0.0,
                    use_cache=False,
                )
            )

        teacher, student = model(64, 2), model(32, 1)
        source = {
            "kind": "random_llama",
            "tokenizer_alignment": "shared synthetic token IDs 0..63",
        }

        def batches(seed):
            generator = torch.Generator().manual_seed(seed)
            result = []
            for _ in range(args.batches):
                ids = torch.randint(
                    0, 64, (args.batch_size, args.seqlen), generator=generator
                )
                result.append(
                    {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                )
            return result

        training, heldout = batches(args.seed + 1), batches(args.seed + 2)
    else:
        models, tokenizers = [], []
        for name, revision in (
            (args.teacher_model, args.teacher_revision),
            (args.student_model, args.student_revision),
        ):
            kwargs = {
                "revision": revision,
                "local_files_only": args.local_files_only,
                "trust_remote_code": False,
            }
            models.append(
                AutoModelForCausalLM.from_pretrained(
                    name, torch_dtype=torch.float32, **kwargs
                )
            )
            tokenizers.append(AutoTokenizer.from_pretrained(name, **kwargs))
        teacher, student = models
        if teacher.config.model_type != "llama" or student.config.model_type != "llama":
            raise ValueError("Both pretrained models must be Llama")
        if teacher.config.vocab_size != student.config.vocab_size:
            raise ValueError("Teacher/student vocab sizes differ")
        mapping = check_tokenizers(*tokenizers, student.config.vocab_size)
        source = {
            "kind": "pretrained_llama",
            "tokenizer_mapping_sha256": mapping,
            "teacher": args.teacher_model,
            "student": args.student_model,
        }
        training, heldout = [
            list(
                _helpers.text_batches(
                    paths, tokenizers[1], args.seqlen, args.batch_size, args.batches
                )
            )
            for paths in (args.calibration_text, args.evaluation_text)
        ]
    evidence.update(
        train_batch_hashes=[_helpers.batch_hash(b) for b in training],
        eval_batch_hashes=[_helpers.batch_hash(b) for b in heldout],
        train_seed=args.seed + 1 if args.tiny else None,
        eval_seed=args.seed + 2 if args.tiny else None,
    )
    if set(evidence["train_batch_hashes"]) & set(evidence["eval_batch_hashes"]):
        raise ValueError("Training/evaluation token batches overlap")
    teacher = teacher.float().to(device).eval()
    student = student.float().to(device)
    initial = {n: t.detach().cpu().clone() for n, t in student.state_dict().items()}
    teacher_before = state_hash(teacher)
    student_before = state_hash(student)
    source.update(
        teacher_config=teacher.config.to_dict(), student_config=student.config.to_dict()
    )
    report = {
        "arguments": {
            k: str(v)
            if isinstance(v, Path)
            else [str(p) for p in v]
            if isinstance(v, list)
            else v
            for k, v in vars(args).items()
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "device": str(device),
            "processor": platform.processor(),
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "torch_num_threads": torch.get_num_threads(),
        },
        "model_source": source,
        "data": evidence,
        "parameter_counts": {
            "teacher": sum(p.numel() for p in teacher.parameters()),
            "student": sum(p.numel() for p in student.parameters()),
        },
        "student_initial_sha256": student_before,
    }
    before = evaluate(student, teacher, heldout, device)
    training_result = train(student, teacher, training, args, device, initial, True)
    report["distillation"] = {
        "before": before,
        "training": training_result,
        "after": evaluate(student, teacher, heldout, device),
    }
    teacher_after = state_hash(teacher)
    if (
        teacher_before != teacher_after
        or any(p.grad is not None or p.requires_grad for p in teacher.parameters())
        or teacher.training
    ):
        raise RuntimeError(
            "Teacher was changed, unfrozen, trained, or received gradients"
        )
    report["teacher_state"] = {
        "before": teacher_before,
        "after": teacher_after,
        "frozen": True,
        "eval": True,
        "no_gradients": True,
    }
    export = args.output / "student_state_dict.pt"
    torch.save({n: t.detach().cpu() for n, t in student.state_dict().items()}, export)
    reloaded = LlamaForCausalLM(copy.deepcopy(student.config)).float()
    reloaded.load_state_dict(
        torch.load(export, map_location="cpu", weights_only=True), strict=True
    )
    reloaded.to(device).eval()
    error = 0.0
    with torch.no_grad():
        for batch in heldout:
            kwargs = {k: v.to(device) for k, v in batch.items()}
            expected = student(**kwargs, use_cache=False).logits
            actual = reloaded(**kwargs, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            error = max(error, (actual - expected).abs().max().item())
    report["student_export"] = {
        "path": str(export),
        "sha256": _helpers.file_hash(export),
        "bytes": export.stat().st_size,
        "max_abs_error": error,
    }
    del reloaded
    student.load_state_dict(initial, strict=True)
    if state_hash(student) != student_before:
        raise RuntimeError("CE baseline initialization differs")
    baseline_before = evaluate(student, teacher, heldout, device)
    baseline_train = train(student, teacher, training, args, device, initial, False)
    report["ce_baseline"] = {
        "before": baseline_before,
        "training": baseline_train,
        "after": evaluate(student, teacher, heldout, device),
    }
    if state_hash(teacher) != teacher_before:
        raise RuntimeError("Teacher changed during baseline")
    report["passed"] = True
    report["quality_scope"] = (
        "Descriptive only; random tiny models do not establish accuracy improvement"
    )
    serialized = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "report.json").write_text(serialized, encoding="utf-8")
    return report


def main():
    args = parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Distillation verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verification passed: {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
