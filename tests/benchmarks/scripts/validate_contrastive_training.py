# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午11:37
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Offline pretrained CLIP one-update numerical gate, not a speed benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import io
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from parascale.config import ParaScaleConfig
from parascale.runtime.training import TrainEngine
from parascale.workloads.clip import build_clip_contrastive_components
from parascale.workloads.contrastive_loss import ContrastiveLoss
from parascale.workloads.datacomp import _iter_datacomp_tar_entries
from parascale.workloads.specs.clip import ClipContrastiveSpec

ATOL, RTOL = 1e-5, 1e-4


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_hash(model):
    return json_hash(
        {
            name: tensor_hash(
                value.reshape(1) if name in {"logit_scale", "logit_bias"} else value
            )
            for name, value in model.state_dict().items()
        }
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eager(model):
    # HF 4.51 CLIP attention dispatch reads the tower configs at forward time.
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None:
            config._attn_implementation = "eager"
            for name in ("text_config", "vision_config"):
                child = getattr(config, name, None)
                if child is not None:
                    child._attn_implementation = "eager"
    return model.float().train()


def load_batch(args):
    images, captions, samples = [], [], []
    for shard in sorted(args.data_dir.glob("*.tar")):
        for stem, entry in _iter_datacomp_tar_entries(shard):
            if "image_bytes" not in entry or not entry.get("text"):
                continue
            raw, caption = entry["image_bytes"], entry["text"].splitlines()[0]
            with Image.open(io.BytesIO(raw)) as image:
                images.append(image.convert("RGB"))
            captions.append(caption)
            samples.append(
                {
                    "shard": shard.name,
                    "key": stem,
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
                }
            )
            if len(samples) == args.global_batch:
                break
        if len(samples) == args.global_batch:
            break
    if len(samples) != args.global_batch:
        raise ValueError(
            f"WDS has fewer than {args.global_batch} paired image/caption samples"
        )
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    batch = dict(
        processor(
            images=images,
            text=captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
    )
    return batch, {
        "samples": samples,
        "samples_sha256": json_hash(samples),
        "tensor_sha256": {k: tensor_hash(v) for k, v in batch.items()},
    }


def compare(actual, expected):
    if actual is None or expected is None:
        return {"passed": actual is expected, "max_abs": None, "max_rel": None}
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.numel() == expected.numel() == 1:
        actual = actual.reshape_as(expected)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    delta = (actual.double() - expected.double()).abs()
    result = {
        "max_abs": delta.max().item() if finite else None,
        "max_rel": (delta / expected.double().abs().clamp_min(1e-30)).max().item()
        if finite
        else None,
        "finite": finite,
    }
    try:
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        result["passed"] = finite
    except AssertionError as exc:
        result.update(passed=False, error=str(exc))
    return result


def run(args, rank, world, device):
    batch, evidence = load_batch(args)
    identities = [None] * world
    dist.all_gather_object(identities, evidence)
    if any(item != evidence for item in identities):
        raise ValueError("ranks loaded different global samples or tensors")
    torch.manual_seed(42)
    reference = eager(
        CLIPModel.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    ).to(device)
    model_config = reference.config.to_dict()
    initial_hash = state_hash(reference)
    reference_optimizer = torch.optim.SGD(
        reference.parameters(), lr=1e-4, momentum=0.9, foreach=False
    )
    torch.manual_seed(42)
    reference_loss = reference(
        **{k: v.to(device) for k, v in batch.items()}, return_loss=True
    ).loss
    reference_loss.backward()
    expected_loss = reference_loss.detach().cpu()
    gradients = {
        n: p.grad.detach().cpu().clone() if p.grad is not None else None
        for n, p in reference.named_parameters()
    }
    reference_optimizer.step()
    parameters = {n: p.detach().cpu().clone() for n, p in reference.named_parameters()}
    del reference_loss, reference_optimizer, reference
    gc.collect()
    torch.cuda.empty_cache()

    spec = ClipContrastiveSpec(
        model_type="hf_clip",
        pretrained_model_name_or_path=str(args.model),
        data_type="datacomp_wds",
        data_dir=str(args.data_dir),
        streaming=True,
        batch_size=args.global_batch // world,
        num_batches=1,
        text_length=77,
    )
    product, unused_optimizer, unused_batches, objective = (
        build_clip_contrastive_components(spec)
    )
    del unused_optimizer, unused_batches
    if not isinstance(objective, ContrastiveLoss) or objective.family != "clip":
        raise TypeError("product builder did not return the CLIP ContrastiveLoss")
    hf = eager(product.wrapped_model)
    if hf.config.to_dict() != model_config or state_hash(hf) != initial_hash:
        raise ValueError("reference/product initial model config or tensors differ")
    if set(dict(hf.named_parameters())) != set(parameters):
        raise ValueError("reference/product parameter names differ")
    optimizer = torch.optim.SGD(
        product.parameters(), lr=1e-4, momentum=0.9, foreach=False
    )
    config = ParaScaleConfig(
        training_backend="native_ddp" if world > 1 else "native",
        data_parallel_size=world,
        precision="fp32",
        grad_clip_norm=None,
        ddp_comm_hook="none",
        gradient_accumulation_steps=1,
    )
    engine = TrainEngine(config)
    checks, updates = {}, []
    original_step = engine.step

    def checked_step(optimizer=None):
        checks["gradients"] = {
            n: compare(p.grad, gradients[n]) for n, p in hf.named_parameters()
        }
        updated = original_step(optimizer)
        updates.append(updated)
        checks["parameters"] = {
            n: compare(p, parameters[n]) for n, p in hf.named_parameters()
        }
        return updated

    engine.step = checked_step
    local = args.global_batch // world
    inputs = {k: v[rank * local : (rank + 1) * local] for k, v in batch.items()}
    torch.manual_seed(42)
    state = engine.fit(
        [inputs], max_steps=1, model=product, optimizer=optimizer, loss_fn=objective
    )
    loss = torch.tensor(float(state.last_metrics["loss"]), device=device)
    dist.all_reduce(loss)
    loss /= world
    checks["loss"] = compare(loss, expected_loss)
    passed = (
        updates == [True]
        and state.global_step == 1
        and checks["loss"]["passed"]
        and all(
            item["passed"]
            for group in ("gradients", "parameters")
            for item in checks[group].values()
        )
    )
    sources = {
        name: {
            "path": str(Path(module.__file__).resolve()),
            "sha256": file_hash(module.__file__),
        }
        for name, module in list(sys.modules.items())
        if name.startswith("parascale")
        and (getattr(module, "__file__", None) or "").endswith(".py")
    }
    return {
        "status": "passed" if passed else "failed",
        "rank": rank,
        "world_size": world,
        "gpu": torch.cuda.get_device_name(device),
        "checks": checks,
        "updates": updates,
        "global_step": state.global_step,
        "reference_loss": expected_loss.item()
        if torch.isfinite(expected_loss)
        else None,
        "product_rank_average_loss": loss.item() if torch.isfinite(loss) else None,
        "data": evidence,
        "model_config": model_config,
        "model_config_sha256": json_hash(model_config),
        "initial_state_sha256": initial_hash,
        "source": sources,
        "script_sha256": file_hash(__file__),
        "hf_clip_source_sha256": file_hash(inspect.getfile(CLIPModel)),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--global-batch", type=int, choices=[32, 128], default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir() or not args.data_dir.is_dir():
        parser.error("model and data-dir must be existing local directories")
    if args.output.exists():
        parser.error("output must be a new JSON path")
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if world not in (1, 2, 4) or not torch.cuda.is_available():
        parser.error("requires torchrun with 1/2/4 CUDA ranks")
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    try:
        result = run(args, rank, world, torch.device("cuda", local))
        results = [None] * world
        dist.all_gather_object(results, result)
        passed = all(item["status"] == "passed" for item in results)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "status": "passed" if passed else "failed",
                        "atol": ATOL,
                        "rtol": RTOL,
                        "precision": "fp32",
                        "attention": "eager",
                        "tf32": False,
                        "optimizer": {"type": "sgd", "lr": 1e-4, "momentum": 0.9},
                        "global_batch": args.global_batch,
                        "ranks": results,
                        "scope": "one-update numerical parity; no throughput claim",
                    },
                    stream,
                    indent=2,
                    allow_nan=False,
                )
        return 0 if passed else 1
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = args.output.with_name(f"{args.output.stem}.rank{rank}.failure.json")
        failure.write_text(
            json.dumps(
                {"status": "failed", "rank": rank, "error": repr(exc)}, indent=2
            ),
            encoding="utf-8",
        )
        if rank == 0 and not args.output.exists():
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "status": "failed",
                        "error": repr(exc),
                        "rank": rank,
                        "atol": ATOL,
                        "rtol": RTOL,
                        "script_sha256": file_hash(__file__),
                    },
                    stream,
                    indent=2,
                    allow_nan=False,
                )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
