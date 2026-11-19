# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Offline real-data YOLO KD validation through the standard ParaScale engine.

Only trusted local models are accepted. Supply matching-family pretrained .pt
models for a pretrained comparison; .yaml models are explicitly random-init.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "student-model",
        "teacher-model",
        "image-dir",
        "label-dir",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    for name, default in (
        ("image-size", 320),
        ("batch-size", 4),
        ("steps", 40),
        ("warmup", 10),
        ("num-samples", 96),
        ("repeats", 3),
        ("num-workers", 0),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument(
        "--evaluation-data", type=Path, help="Local YOLO validation YAML"
    )
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if (
        not 0 <= args.warmup < args.steps
        or min(args.batch_size, args.num_samples, args.repeats) < 1
    ):
        parser.error(
            "Require 0 <= warmup < steps and positive batch-size, num-samples, repeats"
        )
    if args.image_size < 32 or args.image_size % 32 or args.num_workers < 0:
        parser.error(
            "image-size must be a positive multiple of 32; num-workers must be nonnegative"
        )
    for name in (
        "student_model",
        "teacher_model",
        "image_dir",
        "label_dir",
        "evaluation_data",
    ):
        value = getattr(args, name)
        if value is not None:
            value = value.expanduser().resolve()
            if not (value.is_dir() if name.endswith("_dir") else value.is_file()):
                parser.error(f"Missing local asset: {value}")
            setattr(args, name, value)
    if any(
        getattr(args, n).suffix.lower() not in (".pt", ".yaml", ".yml")
        for n in ("student_model", "teacher_model")
    ):
        parser.error("Models must be trusted local .pt or .yaml files")
    return args


def save(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_native_device(torch, device):
    from parascale.runtime.backends.devices import resolve_torch_device

    actual = resolve_torch_device(
        torch, local_rank=int(os.environ.get("LOCAL_RANK", "0"))
    )
    require(
        device == actual,
        f"Native backend selects {actual}, not {device}; select CUDA hardware with "
        "CUDA_VISIBLE_DEVICES and use its logical device index, or hide CUDA for CPU runs",
    )


def run(args):
    import torch
    import ultralytics
    from ultralytics import YOLO

    from parascale.checkpoint import CheckpointManager
    from parascale.runtime import train_runner
    from parascale.workloads import build_training_components
    from parascale.workloads.specs.yolo import YoloDetectionSpec
    from parascale.workloads.yolo import _yolo_cached_detection_samples

    require(
        int(os.environ.get("WORLD_SIZE", "1")) == 1,
        "Run without a distributed launcher",
    )
    device = torch.device(args.device)
    require(device.type in ("cuda", "cpu"), "Only CUDA or CPU validation is supported")
    cuda = device.type == "cuda"
    if cuda:
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
    validate_native_device(torch, device)
    if cuda:
        torch.cuda.set_device(device)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(
        not any(output.iterdir()),
        "Use an empty output directory to preserve earlier evidence",
    )
    base = {
        "backend": {"training_backend": "native", "precision": "fp32"},
        "optimizer": {
            "type": "adamw",
            "lr": args.learning_rate,
            "weight_decay": 0.0005,
        },
        "runtime": {"device": str(device)},
        "training": {
            "workload": "yolo_detection",
            "max_steps": args.steps,
            "checkpoint_interval": args.steps,
            "seed": 42,
        },
        "model": {"path": str(args.student_model)},
        "data": {
            "image_dir": str(args.image_dir),
            "label_dir": str(args.label_dir),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "num_workers": args.num_workers,
        },
    }
    samples = _yolo_cached_detection_samples(YoloDetectionSpec.from_config(base))
    manifest = [
        {
            kind: {
                "path": str(path),
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
            }
            for kind, path in sample.items()
        }
        for sample in samples
    ]
    save(output / "data_manifest.json", manifest)
    teacher_config = deepcopy(base)
    teacher_config["model"]["path"] = str(args.teacher_model)
    teacher_config["runtime"]["device"] = "cpu"
    teacher_config["training"]["checkpoint_dir"] = str(output / "teacher_unused")
    teacher, optimizer, loader, loss = build_training_components(teacher_config)
    teacher_path = output / "teacher_proxy_state.pt"
    torch.save(teacher.state_dict(), teacher_path)
    teacher_initial = state_hash(teacher.state_dict())
    teacher_file_initial = file_hash(teacher_path)
    teacher_parameters = sum(p.numel() for p in teacher.parameters())
    del teacher, optimizer, loader, loss
    initial_config = deepcopy(base)
    initial_config["runtime"]["device"] = "cpu"
    initial, optimizer, loader, loss = build_training_components(initial_config)
    initial_raw = {
        k: v.detach().cpu().clone()
        for k, v in initial.wrapped_model.state_dict().items()
    }
    parameter_keys = set(dict(initial.wrapped_model.named_parameters()))
    student_parameters = sum(p.numel() for p in initial.parameters())
    require(
        teacher_parameters > student_parameters,
        "Teacher must have more parameters than student",
    )
    del initial, optimizer, loader, loss
    save(output / "teacher_config.json", teacher_config)
    summary = {
        "status": "running",
        "scope": "short real-data functional and training-cost validation",
        "metadata": {
            "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if cuda
            else platform.processor(),
            "workload": "yolo_detection",
            "backend": "native",
            "precision": "fp32",
            "student_parameters": student_parameters,
            "teacher_parameters": teacher_parameters,
            "models": {
                n: {
                    "path": str(getattr(args, n)),
                    "sha256": file_hash(getattr(args, n)),
                    "initialization": "pretrained"
                    if getattr(args, n).suffix == ".pt"
                    else "random_yaml",
                }
                for n in ("student_model", "teacher_model")
            },
            "data_manifest_sha256": file_hash(output / "data_manifest.json"),
            "selected_samples": len(samples),
            "arguments": vars(args),
        },
        "teacher_initial_state_sha256": teacher_initial,
        "runs": [],
        "evaluation": {},
        "limitations": [
            "No guaranteed quality gain; mAP is unknown unless official validation is enabled.",
            "Step throughput excludes data wait and checkpointing; wall time includes engine setup, audit hashing and saving.",
            "The engine resets CUDA peaks before fit; reported peaks include training and export, not all transient setup allocations.",
            "Small fixed dataset, fixed-seed repeats; not a multi-seed quality, convergence or distributed-scaling study.",
            "Prediction parity uses one real image and raw pre-NMS output.",
        ],
    }

    def evaluate(source, name, raw=None):
        model = YOLO(str(source))
        if raw is not None:
            model.model.load_state_dict(raw, strict=True)
        metrics = model.val(
            data=str(args.evaluation_data),
            device=str(device),
            imgsz=args.image_size,
            batch=args.batch_size,
            workers=args.num_workers,
            half=False,
            plots=False,
            save=False,
            project=str(output / "evaluation"),
            name=name,
            exist_ok=True,
        )
        return {
            "mAP50_95": float(metrics.box.map),
            "mAP50": float(metrics.box.map50),
            "metrics": metrics.results_dict,
        }

    original_builder = train_runner.build_detection_distillation_components
    original_components = train_runner.build_training_components
    try:
        if args.evaluation_data:
            summary["metadata"]["evaluation_data_sha256"] = file_hash(
                args.evaluation_data
            )
            teacher_raw = torch.load(
                teacher_path, map_location="cpu", weights_only=True
            )
            teacher_raw = {
                k.removeprefix("wrapped_model."): v for k, v in teacher_raw.items()
            }
            summary["evaluation"]["initial_teacher"] = evaluate(
                args.teacher_model, "initial_teacher", teacher_raw
            )
            del teacher_raw
            summary["evaluation"]["initial_student"] = evaluate(
                args.student_model, "initial_student", initial_raw
            )
        for repeat in range(args.repeats):
            for mode in ("baseline", "kd") if repeat % 2 == 0 else ("kd", "baseline"):
                directory = output / f"{mode}_{repeat}"
                directory.mkdir()
                config = deepcopy(base)
                config["training"]["checkpoint_dir"] = str(directory / "checkpoints")
                # Identical initialization for YAML as well as pretrained paired runs.
                config["training"]["seed"] = 42
                if mode == "kd":
                    config["distillation"] = {
                        "enabled": True,
                        "task": "detection",
                        "feature_weight": 0.1,
                        "teacher": {
                            "config": teacher_config,
                            "checkpoint": str(teacher_path),
                        },
                    }
                save(directory / "config.json", config)
                observed = {}

                def observe_components(*values, **kwargs):
                    built = original_components(*values, **kwargs)
                    observed["student_initial"] = state_hash(
                        built[0].wrapped_model.state_dict()
                    )
                    return built

                def observe(*values, **kwargs):
                    built = original_builder(*values, **kwargs)
                    observed["teacher"] = built[1].teacher
                    observed["initial"] = state_hash(built[1].teacher.state_dict())
                    return built

                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                with (
                    patch.object(
                        train_runner, "build_detection_distillation_components", observe
                    ),
                    patch.object(
                        train_runner, "build_training_components", observe_components
                    ),
                ):
                    result = train_runner.run_train_from_config(config)
                require(
                    result["train_device"] == str(device),
                    "Training device differs from measured device",
                )
                if cuda:
                    torch.cuda.synchronize(device)
                wall = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated(device) if cuda else None
                reserved = torch.cuda.max_memory_reserved(device) if cuda else None
                save(directory / "result.json", result)
                history = result["metrics_history"]
                require(
                    len(history) == args.steps and result["global_step"] == args.steps,
                    "Incomplete training budget",
                )
                measured = history[args.warmup :]
                durations = [float(row["step_time_seconds"]) for row in measured]
                data_wait = sum(
                    float(row["dataloader_wait_ms"]) / 1000 for row in measured
                )
                images = [int(row["images"]) for row in measured]
                require(
                    all(math.isfinite(t) and t > 0 for t in durations)
                    and min(images) > 0,
                    "Invalid engine timing/image counts",
                )
                require(
                    all(math.isfinite(float(row["loss"])) for row in history),
                    "Nonfinite loss",
                )
                require(
                    result["checkpoint_validation"]["ok"],
                    "Checkpoint validation failed",
                )
                manager = CheckpointManager(config["training"]["checkpoint_dir"])
                checkpoint = manager.read_manifest_path(result["checkpoint"])
                entry = next(
                    e for e in checkpoint.files if e.get("role") == "backend_state"
                )
                payload = torch.load(
                    manager.resolve_payload_path(checkpoint, entry),
                    map_location="cpu",
                    weights_only=True,
                )
                state = payload["backend_state"]["model_state_dict"]
                prefix = "student.wrapped_model." if mode == "kd" else "wrapped_model."
                raw = {
                    k[len(prefix) :]: v
                    for k, v in state.items()
                    if k.startswith(prefix)
                }
                require(
                    set(raw) == set(initial_raw),
                    "Student checkpoint keys differ from the initial model",
                )
                checks = {
                    "student_weights_changed": any(
                        not torch.equal(raw[k], initial_raw[k]) for k in parameter_keys
                    ),
                    "same_initial_student": observed["student_initial"]
                    == state_hash(initial_raw),
                    "no_teacher_in_model_checkpoint": not any(
                        "teacher" in k.lower() for k in state
                    ),
                    "projectors_present": any(
                        k.startswith("projectors.") for k in state
                    ),
                    "teacher_file_unchanged": file_hash(teacher_path)
                    == teacher_file_initial,
                }
                if mode == "kd":
                    export = torch.load(
                        result["distillation"]["student_checkpoint"],
                        map_location="cpu",
                        weights_only=True,
                    )
                    checks["export_exact_raw_student"] = set(export) == set(
                        raw
                    ) and all(torch.equal(raw[k], export[k]) for k in raw)
                    checks["teacher_state_unchanged"] = (
                        observed["initial"]
                        == state_hash(observed["teacher"].state_dict())
                        == teacher_initial
                    )
                    checks["teacher_frozen_no_grad"] = all(
                        not p.requires_grad and p.grad is None
                        for p in observed["teacher"].parameters()
                    )
                    import numpy as np
                    from PIL import Image

                    with Image.open(samples[0]["image"]) as image:
                        tensor = (
                            torch.from_numpy(
                                np.array(
                                    image.convert("RGB").resize(
                                        (args.image_size, args.image_size)
                                    ),
                                    copy=True,
                                )
                            )
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .float()
                            / 255
                        )
                    predictions = []
                    for weights in (raw, export):
                        model = YOLO(str(args.student_model))
                        model.model.load_state_dict(weights, strict=True)
                        with torch.inference_mode():
                            prediction = model.model.cpu().float().eval()(tensor)
                        predictions.append(
                            prediction[0]
                            if isinstance(prediction, (tuple, list))
                            else prediction
                        )
                        del model
                    checks["export_prediction_parity"] = torch.allclose(
                        *predictions, rtol=1e-5, atol=1e-6
                    )
                    summary["teacher_final_state_sha256"] = state_hash(
                        observed["teacher"].state_dict()
                    )
                else:
                    torch.save(raw, directory / "student_raw.pt")
                record = {
                    "mode": mode,
                    "repeat": repeat,
                    "wall_seconds": wall,
                    "steady_state_images_per_second": sum(images) / sum(durations),
                    "end_to_end_images_per_second": sum(images)
                    / (sum(durations) + data_wait),
                    "dataloader_wait_seconds": data_wait,
                    "measured_images": sum(images),
                    "measured_steps": len(measured),
                    "measured_step_seconds": sum(durations),
                    "cuda_peak_allocated_bytes": peak,
                    "cuda_peak_reserved_bytes": reserved,
                    "checks": checks,
                }
                if mode == "kd":
                    record.update(
                        teacher_initial_state_sha256=observed["initial"],
                        teacher_final_state_sha256=summary[
                            "teacher_final_state_sha256"
                        ],
                    )
                summary["runs"].append(record)
                save(directory / "validation.json", record)
                require(
                    all(v for k, v in checks.items() if k != "projectors_present")
                    and checks["projectors_present"] == (mode == "kd"),
                    f"Validation failed: {checks}",
                )
                observed.clear()
                if args.evaluation_data:
                    summary["evaluation"][directory.name] = evaluate(
                        args.student_model, directory.name, raw
                    )
                del state, raw, payload
                save(output / "summary.json", summary)
        summary["aggregates"] = {}
        for mode in ("baseline", "kd"):
            rows = [r for r in summary["runs"] if r["mode"] == mode]
            summary["aggregates"][mode] = {"repeats": len(rows)}
            for key in (
                "steady_state_images_per_second",
                "end_to_end_images_per_second",
                "wall_seconds",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
            ):
                values = [r[key] for r in rows if r[key] is not None]
                summary["aggregates"][mode][key] = (
                    {
                        "mean": statistics.mean(values),
                        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                        "repeats": values,
                    }
                    if values
                    else None
                )
        summary["teacher_file_final_sha256"] = file_hash(teacher_path)
        summary["teacher_file_initial_sha256"] = teacher_file_initial
        summary["status"] = "passed"
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(output / "summary.json", summary)
    print(str(output / "summary.json"))


if __name__ == "__main__":
    options = arguments()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    os.environ["YOLO_OFFLINE"] = "true"
    # Fail closed if model loading/official validation attempts an asset download.
    with patch(
        "socket.socket.connect",
        side_effect=RuntimeError("Offline benchmark: provision all assets first"),
    ):
        run(options)
