# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Held-out multi-positive CLIP retrieval through the product inference adapter."""

import argparse
import inspect
import io
import json
import tarfile
import time
from pathlib import Path

import torch
import transformers

from parascale.runtime.inference.runner import InferenceRunner
from parascale.workloads.inference import HFClipInferenceModel
from tests.benchmarks.scripts.production_soak import hf_state_from_training
from tests.benchmarks.scripts.validate_contrastive_training import json_hash, state_hash
from tests.benchmarks.tools.prepare_flickr8k_retrieval import (
    FILES,
    REVISION,
    check_unique,
    file_hash,
)


def validate_manifest(manifest):
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source") != "jxie/flickr8k"
        or manifest.get("revision") != REVISION
        or manifest.get("source_sha256") != FILES
    ):
        raise ValueError("unexpected source identity")
    seen = set()
    for split, count in (("train", 5999), ("validation", 1000)):
        records = manifest.get("splits", {}).get(split, [])
        if len(records) != count:
            raise ValueError(f"incomplete {split} split")
        for index, record in enumerate(records):
            source_rows = (
                [1027, 1359]
                if split == "train" and index == 1027
                else [index + int(split == "train" and index >= 1359)]
            )
            if (
                record["key"] != f"{index:06d}"
                or record["shard"] != f"{split}/{index // 1500:05d}.tar"
                or record.get("source_rows") != source_rows
                or len(record["captions"]) != 5 * len(source_rows)
                or any(
                    not isinstance(c, str) or not c.strip() for c in record["captions"]
                )
            ):
                raise ValueError("invalid record identity or captions")
            check_unique(split, record["image_sha256"], record["pixel_sha256"], seen)
    return manifest["splits"]["validation"]


def select_accelerator(device):
    selected = torch.device(device)
    if selected.type == "cpu":
        return None
    if selected.type != "cuda":
        raise ValueError("this evaluator qualifies CPU/CUDA only")
    if selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(selected)
    return torch.cuda


def compare_quality(
    metrics, baseline, manifest_hash, evaluation_identity, initial_hash
):
    if (
        baseline.get("status") != "passed"
        or baseline.get("manifest_sha256") != manifest_hash
        or baseline.get("evaluation_identity") != evaluation_identity
        or baseline.get("model_sha256") != initial_hash
        or baseline.get("checkpoint") is not None
    ):
        raise ValueError("baseline evaluation identity differs")
    return {
        name: {
            "baseline": baseline["metrics"][name],
            "current": metrics[name],
            "max_absolute_drop": 0.01,
            "passed": metrics[name] + 0.01 >= baseline["metrics"][name] - 1e-7,
        }
        for name in ("image_to_text_r1", "text_to_image_r1")
    }


def retrieval_metrics(scores, caption_owners):
    """Recall fractions with any owned caption positive and stable ties."""
    owners = torch.as_tensor(caption_owners, dtype=torch.long, device=scores.device)
    if (
        scores.ndim != 2
        or not all(scores.shape)
        or owners.shape != (scores.shape[1],)
        or set(owners.tolist()) != set(range(scores.shape[0]))
    ):
        raise ValueError("invalid gallery shape or caption ownership")
    if not torch.isfinite(scores).all():
        raise ValueError("retrieval scores must be finite")
    image_order = torch.argsort(scores, dim=1, descending=True, stable=True)
    text_order = torch.argsort(scores.T, dim=1, descending=True, stable=True)
    image_ids = torch.arange(scores.shape[0], device=scores.device)[:, None]
    metrics = {}
    for k in (1, 5, 10):
        metrics[f"image_to_text_r{k}"] = (
            (owners[image_order[:, :k]] == image_ids).any(dim=1).float().mean().item()
        )
        metrics[f"text_to_image_r{k}"] = (
            (text_order[:, :k] == owners[:, None]).any(dim=1).float().mean().item()
        )
    return metrics


def validation_batches(root, records, batch_size):
    import hashlib

    from PIL import Image

    images, texts = [], []
    with tarfile.open(root / "validation/00000.tar") as archive:
        for record in records:
            raw = archive.extractfile(record["key"] + ".jpg").read()
            if hashlib.sha256(raw).hexdigest() != record["image_sha256"]:
                raise ValueError("validation image checksum mismatch")
            with Image.open(io.BytesIO(raw)) as image:
                images.append(image.convert("RGB"))
            texts.extend(record["captions"])
            if len(images) == batch_size:
                yield {
                    "images": images,
                    "texts": texts,
                    "num_images": len(images),
                    "num_pairs": len(texts),
                }
                images, texts = [], []
    if images:
        yield {
            "images": images,
            "texts": texts,
            "num_images": len(images),
            "num_pairs": len(texts),
        }


def evaluate(args):
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = validate_manifest(manifest)
    if (
        file_hash(args.data / "validation/00000.tar")
        != manifest["shards_sha256"]["validation/00000.tar"]
    ):
        raise ValueError("validation shard checksum mismatch")
    accelerator = select_accelerator(args.device)
    model = HFClipInferenceModel(str(args.model)).to(args.device).eval()
    initial_hash = state_hash(model._model)
    checkpoint = None
    if args.checkpoint_root is not None:
        from parascale.checkpoint import CheckpointManager

        manager = CheckpointManager(str(args.checkpoint_root))
        validation = manager.validate(args.checkpoint_step)
        if not validation.ok:
            raise ValueError(f"checkpoint validation failed: {validation.to_dict()}")
        path = manager.payload_path(args.checkpoint_step, "backend_state.pt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = hf_state_from_training(
            payload["backend_state"]["model_state_dict"], model._model.state_dict()
        )
        model._model.load_state_dict(state, strict=True)
        checkpoint = {
            "step": args.checkpoint_step,
            "payload_sha256": file_hash(path),
            "manifest_sha256": file_hash(manager.manifest_path(args.checkpoint_step)),
        }
        del payload, state
    identity = state_hash(model._model)
    evaluation_identity = {
        "config_sha256": json_hash(model._model.config.to_dict()),
        "image_processor_sha256": json_hash(model._processor.image_processor.to_dict()),
        "tokenizer_vocab_sha256": json_hash(model._processor.tokenizer.get_vocab()),
        "processor_assets": {
            path.name: file_hash(path)
            for path in sorted(args.model.iterdir())
            if path.is_file() and path.suffix in {".json", ".txt", ".model"}
        },
        "text_length": model._model.config.text_config.max_position_embeddings,
        "transformers": transformers.__version__,
        "adapter_sha256": file_hash(inspect.getfile(HFClipInferenceModel)),
    }
    images, texts = [], []

    def collect(output):
        for key, target in (("image_embeddings", images), ("text_embeddings", texts)):
            features = torch.tensor(output[key], dtype=torch.float32)
            if features.ndim != 2 or not torch.isfinite(features).all():
                raise ValueError("invalid product features")
            torch.testing.assert_close(
                features.norm(dim=1), torch.ones(len(features)), atol=1e-5, rtol=1e-5
            )
            target.append(features)

    first = next(validation_batches(args.data, records, args.batch_size))
    actual = model.embed(first)
    inputs = model._processor(
        images=first["images"],
        text=first["texts"],
        padding=True,
        truncation=True,
        max_length=model._model.config.text_config.max_position_embeddings,
        return_tensors="pt",
    )
    with torch.inference_mode():
        native = model._model(**{k: v.to(args.device) for k, v in inputs.items()})
    for key, reference in (
        ("image_embeddings", native.image_embeds),
        ("text_embeddings", native.text_embeds),
    ):
        torch.testing.assert_close(
            torch.tensor(actual[key]), reference.cpu(), atol=1e-6, rtol=1e-5
        )
    del inputs, native, actual, first
    started = time.perf_counter()
    runner = InferenceRunner(
        model, "multimodal_embedding", args.device, lambda: accelerator
    )
    report = runner.run(
        validation_batches(args.data, records, args.batch_size),
        warmup_steps=0,
        max_outputs=0,
        on_output=collect,
    )
    embedding_seconds = time.perf_counter() - started
    image_features, text_features = torch.cat(images), torch.cat(texts)
    if len(image_features) != 1000 or len(text_features) != 5000:
        raise ValueError("incomplete validation gallery")
    metrics = retrieval_metrics(
        image_features @ text_features.T, [i for i in range(1000) for _ in range(5)]
    )
    manifest_hash = file_hash(manifest_path)
    checks = {}
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        checks = compare_quality(
            metrics, baseline, manifest_hash, evaluation_identity, initial_hash
        )
        if checkpoint is not None and identity == baseline["model_sha256"]:
            raise ValueError("trained checkpoint did not change model weights")
    return {
        "status": "passed" if all(c["passed"] for c in checks.values()) else "failed",
        "quality_checks": checks,
        "checkpoint": checkpoint,
        "metrics": metrics,
        "images": 1000,
        "captions": 5000,
        "embedding_wall_seconds": embedding_seconds,
        "inference_metrics": report["metrics"],
        "model_sha256": identity,
        "initial_model_sha256": initial_hash,
        "evaluation_identity": evaluation_identity,
        "manifest_sha256": manifest_hash,
        "script_sha256": file_hash(__file__),
        "torch": torch.__version__,
        "scope": "held out from this fine-tuning train split, not known to be absent from pretraining; native HF feature parity on first batch; no training quality acceptance without a matched baseline",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, choices=[8, 16, 32], default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    if (args.checkpoint_root is None) != (args.checkpoint_step is None):
        parser.error("checkpoint-root and checkpoint-step must be supplied together")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        result = evaluate(args)
    except Exception as exc:
        result = {"status": "failed", "error": repr(exc)}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
