# -*- coding: utf-8 -*-
# @Time : 2026/11/20 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Linux-only real CLIP soak supervisor; instrumentation, not a serving system."""

import argparse
import io
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def build_soak_config(args):
    """Extend the qualified example without silently changing its training policy."""
    from parascale.commands.common import load_config_file
    from parascale.configuration.resolver import normalize_execution_config

    if not math.isfinite(args.hours) or args.hours <= 0:
        raise ValueError("hours must be finite positive")
    config = load_config_file(args.config)
    backend = config["parascale"]
    training = config["training"]
    if not (
        backend["training_backend"] == "native_ddp"
        and backend["precision"] == "fp32"
        and backend["data_parallel_size"] == 4
        and backend.get("ddp_comm_hook") == "none"
        and backend.get("ddp_find_unused_parameters") is True
        and backend.get("ddp_static_graph") is False
        and training["batch_size"] == 32
        and training.get("gradient_accumulation_steps", 1) == 1
        and config["data"].get("cpu_prefetch") is True
    ):
        raise ValueError("soak requires the qualified FP32 CLIP DDP4 policy")
    config["model"]["pretrained_model_name_or_path"] = args.model
    config["data"]["data_dir"] = args.data
    config["runtime"]["run_dir"] = str(args.output / "run")
    backend.update(checkpoint_keep_last=3, metrics_history_limit=1000)
    training.update(
        max_steps=100000000,
        checkpoint_interval=100 if args.hours < 1 else 5000,
        checkpoint_dir=str(args.output / "checkpoints"),
        skip_final_checkpoint=False,
    )
    _, effective = normalize_execution_config(config)
    if (
        effective.checkpoint_keep_last != 3
        or effective.metrics_history_limit != 1000
        or effective.batch_size != 32
        or effective.precision != "fp32"
    ):
        raise ValueError("configuration aliases override the qualified soak policy")
    return config


def check_embedding_output(output, expected):
    """Check every returned vector element, not just a shared aggregate."""
    for name in ("image_embeddings", "text_embeddings"):
        reference = expected[name]
        actual = output[name]
        shape = [len(reference), len(reference[0])]
        valid = output[name + "_shape"] == shape and len(actual) == shape[0]
        if valid:
            valid = all(
                len(row) == shape[1]
                and all(
                    math.isfinite(a)
                    and math.isfinite(b)
                    and math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                    for a, b in zip(row, target)
                )
                for row, target in zip(actual, reference)
            )
        if not valid:
            raise ValueError("Inference embedding differs from direct HF reference")


def snapshot_checkpoint_residue(root):
    """Record invalid checkpoint file sizes only after the writer has stopped."""
    from parascale.checkpoint import CheckpointManager

    manager = CheckpointManager(str(root))
    residue = {}
    for path in sorted(root.glob("step-*")):
        if not path.is_dir():
            continue
        try:
            if manager.validate(int(path.name.removeprefix("step-"))).ok:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            pass
        residue[path.name] = {
            file.relative_to(path).as_posix(): file.stat().st_size
            for file in path.rglob("*")
            if file.is_file()
        }
    return residue


def check_checkpoint_retention(root, *, final=False, interrupted=None):
    """Bound complete states separately from known, non-growing interruption debris."""
    count = sum(path.is_dir() for path in root.glob("step-*"))
    if final:
        residue = snapshot_checkpoint_residue(root)
        if len(residue) > 1 or any(
            (interrupted or {}).get(name) != files for name, files in residue.items()
        ):
            raise RuntimeError(
                "Checkpoint retention residue changed after interruption"
            )
        complete = count - len(residue)
    else:
        complete = count
    if complete > (3 if final else 4):
        raise RuntimeError("Checkpoint retention bound exceeded")
    return count


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_training_record(metrics, history_size, limit):
    if not math.isfinite(float(metrics["loss"])) or history_size > limit:
        raise ValueError("nonfinite training loss or unbounded metric history")


def check_progress(record, now, free_bytes):
    if now - record["time"] > 600:
        raise RuntimeError("No progress for 600 seconds")
    if free_bytes < 20 * 1024**3:
        raise RuntimeError("Less than 20 GiB free; stopping before disk exhaustion")


def rss_bytes():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def hf_state_from_training(state, target_state=None):
    for prefix in ("module.wrapped_model.", "wrapped_model."):
        if state and all(key.startswith(prefix) for key in state):
            converted = {key[len(prefix) :]: value for key, value in state.items()}
            if target_state is not None:
                for name in ("logit_scale", "logit_bias"):
                    if name in converted and name in target_state:
                        if (
                            converted[name].numel() != 1
                            or target_state[name].numel() != 1
                        ):
                            raise ValueError("Invalid CLIP scalar storage")
                        converted[name] = converted[name].reshape_as(target_state[name])
            return converted
    raise ValueError("Unexpected trained CLIP state schema")


def train_worker(args):
    from parascale.runtime.train_runner import run_train_from_config
    from parascale.runtime.training.engine import TrainEngine

    rank = int(os.environ["RANK"])
    original = TrainEngine.record_step
    phase = "resume" if args.resume else "train"

    def record(engine, metrics):
        state = original(engine, metrics)
        check_training_record(
            metrics, len(state.metrics_history), engine.config.metrics_history_limit
        )
        if state.global_step == 1 or state.global_step % 100 == 0:
            row = {
                "time": time.time(),
                "step": state.global_step,
                "rank": rank,
                "metrics": metrics,
                "rss_high_water_bytes": rss_bytes(),
                "history_size": len(state.metrics_history),
            }
            write_json(args.output / f"{phase}-rank{rank}.json", row)
            with (args.output / f"{phase}-rank{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return state

    # Observe the public step-recording boundary without changing training math.
    TrainEngine.record_step = record
    config = json.loads((args.output / "train.config.json").read_text())
    if args.resume:
        config["training"]["max_steps"] = 100
    result = run_train_from_config(config, resume_latest=args.resume)
    if rank == 0:
        write_json(args.output / f"{phase}-result.json", result)


def inference_worker(args):
    import torch
    from PIL import Image

    from parascale.runtime.inference.runner import InferenceRunner
    from parascale.workloads.inference import HFClipInferenceModel

    model = HFClipInferenceModel(args.model).to("cuda:0").eval()
    if args.checkpoint:
        from parascale.checkpoint import CheckpointManager

        manager = CheckpointManager(str(args.output / "checkpoints"))
        manifest, _ = manager.latest_valid_training_checkpoint()
        payload = torch.load(
            manager.payload_path(manifest.step, "backend_state.pt"),
            map_location="cpu",
            weights_only=True,
        )
        state = payload["backend_state"]["model_state_dict"]
        model._model.load_state_dict(
            hf_state_from_training(state, model._model.state_dict()), strict=True
        )
        del payload, state
    samples = []
    for shard in sorted(Path(args.data).glob("*.tar")):
        with tarfile.open(shard, "r:") as archive:
            pending = {}
            for member in archive:
                key, _, suffix = member.name.rpartition(".")
                if not member.isfile() or suffix not in ("jpg", "txt"):
                    continue
                pending.setdefault(key, {})[suffix] = archive.extractfile(member).read()
                if len(pending[key]) != 2:
                    continue
                pair = pending.pop(key)
                caption = pair["txt"].decode("utf-8").splitlines()[0].strip()
                if (
                    not caption
                    or len(model._processor.tokenizer(caption)["input_ids"]) > 77
                ):
                    continue
                with Image.open(io.BytesIO(pair["jpg"])) as image:
                    samples.append((image.convert("RGB"), caption))
                if len(samples) == 64:
                    break
        if len(samples) == 64:
            break
    if len(samples) < 16:
        raise ValueError("Insufficient real-image inference inputs")
    runner = InferenceRunner(
        model, "multimodal_embedding", "cuda:0", lambda: torch.cuda
    )
    expected = {}
    batches = {}
    for size in (1, 4, 8, 16):
        batch = {
            "images": [item[0] for item in samples[:size]],
            "texts": [item[1] for item in samples[:size]],
            "num_pairs": size,
        }
        batches[size] = batch
        inputs = model._processor(
            text=batch["texts"],
            images=batch["images"],
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            reference = model._model(
                **{key: value.to("cuda:0") for key, value in inputs.items()}
            )
        expected[size] = {
            "mean": float(reference.logits_per_image.float().mean()),
            "image_embeddings": reference.image_embeds.float().cpu().tolist(),
            "text_embeddings": reference.text_embeds.float().cpu().tolist(),
        }
        if not torch.isfinite(reference.logits_per_image).all():
            raise ValueError("Nonfinite reference logits")
    requests = 0
    started = time.monotonic()
    name = "trained-infer" if args.checkpoint else "infer"
    while not (args.output / "STOP_INFERENCE").exists() or args.checkpoint:
        windows = []
        for size, batch in batches.items():
            runner.run([batch], warmup_steps=0)
            latencies = []
            for _ in range(25):
                before = time.monotonic()
                result = runner.run([batch], warmup_steps=0)
                latencies.append((time.monotonic() - before) * 1000)
                output = result["outputs"][0]
                value = output["logits_per_image_mean"]
                if output["logits_per_image_shape"] != [
                    size,
                    size,
                ] or not math.isfinite(value):
                    raise ValueError("Invalid inference output")
                if not math.isclose(
                    value, expected[size]["mean"], rel_tol=1e-5, abs_tol=1e-5
                ):
                    raise ValueError("Inference disagrees with direct HF reference")
                check_embedding_output(output, expected[size])
                requests += 1
            ordered = sorted(latencies)
            windows.append(
                {
                    "batch_size": size,
                    "requests": 25,
                    "p50_ms": statistics.median(ordered),
                    "p95_ms": ordered[23],
                    "p99_ms": ordered[24],
                    "images_per_second": 25 * size * 1000 / sum(latencies),
                }
            )
        row = {
            "time": time.time(),
            "requests": requests,
            "elapsed_seconds": time.monotonic() - started,
            "windows": windows,
            "rss_high_water_bytes": rss_bytes(),
            "cuda_allocated": torch.cuda.memory_allocated(),
            "cuda_reserved": torch.cuda.memory_reserved(),
            "reference_check": "full image/text embeddings, shapes and mean logits; not retrieval quality",
            "input_policy": "first 64 usable training pairs; first caption line; omit captions over 77 tokens",
        }
        write_json(args.output / f"{name}.json", row)
        with (args.output / f"{name}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if args.checkpoint:
            return


def require_running(children):
    if any(child.poll() is not None for child in children):
        raise RuntimeError("Unexpected worker exit")


def group_alive(group):
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                return True
        except (OSError, IndexError, ValueError):
            continue
    return False


def stop_process(process):
    if getattr(process, "_soak_stopped", False):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + 10
    while group_alive(process.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError("Owned worker group did not stop")
        time.sleep(0.1)
    process._soak_stopped = True


def request_termination(signum, frame):
    raise RuntimeError(f"Supervisor termination requested: signal {signum}")


def supervise(args):
    from tests.benchmarks.scripts.validate_clip_prefetch_recovery import (
        validate_input_binding,
    )

    config = build_soak_config(args)
    binding = validate_input_binding(config, Path(args.data).parent, Path(args.model))
    args.output.mkdir(parents=True, exist_ok=False)
    check_progress(
        {"time": time.time()}, time.time(), shutil.disk_usage(args.output).free
    )
    write_json(args.output / "train.config.json", config)
    children = []
    handles = []
    base = [
        str(Path(__file__).resolve()),
        "--output",
        str(args.output),
        "--model",
        args.model,
        "--data",
        args.data,
        "--config",
        str(args.config),
    ]

    def launch(mode, devices, extra=()):
        command = [sys.executable]
        if mode == "train":
            command += [
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc_per_node=4",
                "--master_addr=127.0.0.1",
                "--master_port=29500",
            ]
        command += base + ["--mode", mode] + list(extra)
        log = (args.output / f"{mode}-{len(children)}.log").open("w")
        handles.append(log)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=devices,
            OMP_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(process)
        return process

    status = {
        "status": "starting",
        "hours_requested": args.hours,
        "source_revision": args.revision,
        "input_binding": binding,
        "quality_gate": "not evaluated; training data is not held-out evaluation",
        "service_gate": "offline inference only; no HTTP service tested",
    }
    write_json(args.output / "status.json", status)
    previous_handler = signal.signal(signal.SIGTERM, request_termination)
    try:
        training = launch("train", "0,1,3,4")
        inference = launch("infer", "2")
        deadline = time.monotonic() + 600
        paths = [args.output / f"train-rank{rank}.json" for rank in range(4)] + [
            args.output / "infer.json"
        ]
        while not all(path.exists() for path in paths):
            if (
                any(child.poll() is not None for child in children)
                or time.monotonic() > deadline
            ):
                raise RuntimeError("Worker failed or startup exceeded 600 seconds")
            time.sleep(5)
        started = time.monotonic()
        status.update(status="running", started_at_unix=time.time())
        while time.monotonic() - started < args.hours * 3600:
            require_running(children)
            for path in paths:
                record = json.loads(path.read_text())
                check_progress(record, time.time(), shutil.disk_usage(args.output).free)
                if record["rss_high_water_bytes"] > 64 * 1024**3:
                    raise RuntimeError("Worker RSS exceeded 64 GiB")
            status.update(
                elapsed_seconds=time.monotonic() - started,
                last_check_unix=time.time(),
                checkpoint_count=check_checkpoint_retention(
                    args.output / "checkpoints"
                ),
            )
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,temperature.gpu,memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            with (args.output / "resources.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "gpu_csv": gpu.stdout,
                            "disk_free_bytes": shutil.disk_usage(args.output).free,
                        }
                    )
                    + "\n"
                )
            write_json(args.output / "status.json", status)
            time.sleep(10)
        status["soak_elapsed_seconds"] = time.monotonic() - started
        require_running(children)
        stop_process(training)
        residue = snapshot_checkpoint_residue(args.output / "checkpoints")
        if len(residue) > 1:
            raise RuntimeError("Multiple incomplete checkpoints after interruption")
        status["interruption_residue"] = residue
        (args.output / "STOP_INFERENCE").touch()
        inference.wait(timeout=120)
        if inference.returncode:
            raise RuntimeError("Inference failed during shutdown")
        status["status"] = "checking_recovery"
        write_json(args.output / "status.json", status)
        resumed = launch("train", "0,1,3,4", ["--resume"])
        if resumed.wait(timeout=1200):
            raise RuntimeError("Post-soak recovery failed")
        result = json.loads((args.output / "resume-result.json").read_text())
        if not result["checkpoint_validation"]["ok"]:
            raise RuntimeError("Final checkpoint validation failed")
        final_count = check_checkpoint_retention(
            args.output / "checkpoints", final=True, interrupted=residue
        )
        if (
            result["global_step"]
            != result["checkpoint_recovery"]["selected_step"] + 100
        ):
            raise RuntimeError("Resume did not complete 100 additional optimizer steps")
        trained = launch("infer", "2", ["--checkpoint"])
        if trained.wait(timeout=300):
            raise RuntimeError("Trained checkpoint inference failed")
        if (
            validate_input_binding(config, Path(args.data).parent, Path(args.model))
            != binding
        ):
            raise RuntimeError("Input assets changed during soak")
        status.update(
            status="stability_checks_passed_quality_pending",
            finished_at_unix=time.time(),
            final_checkpoint_count=final_count,
        )
    except BaseException as exc:
        status.update(status="failed", error=str(exc), finished_at_unix=time.time())
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cleanup_errors = []
        for child in children:
            try:
                stop_process(child)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        for handle in handles:
            handle.close()
        if cleanup_errors:
            status.update(status="failed", cleanup_errors=cleanup_errors)
        write_json(args.output / "status.json", status)
        signal.signal(signal.SIGTERM, previous_handler)
        if cleanup_errors:
            raise RuntimeError(f"Worker cleanup failed: {cleanup_errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["supervise", "train", "infer"], default="supervise"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="unknown")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("hours must be finite positive")
    {"supervise": supervise, "train": train_worker, "infer": inference_worker}[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
