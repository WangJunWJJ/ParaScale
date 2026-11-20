# -*- coding: utf-8 -*-
# @Time : 2026/11/20 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded real-image CLIP recovery validation; not a DataComp quality benchmark."""

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_data(coco, scratch):
    from PIL import Image

    annotations = json.loads((coco / "captions_val2017.json").read_text())
    captions = {}
    for item in sorted(annotations["annotations"], key=lambda item: item["id"]):
        captions.setdefault(item["image_id"], item["caption"])
    images = sorted(annotations["images"], key=lambda item: item["id"])[:512]
    folder = scratch / "wds"
    folder.mkdir()
    records = []
    with tarfile.open(folder / "coco-000000.tar", "w") as archive:
        for item in images:
            path = coco / "val2017" / item["file_name"]
            with Image.open(path) as image:
                image.convert("RGB").load()
            caption = captions[item["id"]]
            assert caption.strip()
            records.append(
                {"id": item["id"], "image_sha256": digest(path), "caption": caption}
            )
            for suffix, payload in (
                ("jpg", path.read_bytes()),
                ("txt", caption.encode()),
            ):
                entry = tarfile.TarInfo(f"{item['id']:012d}.{suffix}")
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
    return folder, records


def inspect_wds(folder):
    from PIL import Image

    records = []
    seen = set()
    for shard in sorted(folder.glob("*.tar")):
        samples = {}
        with tarfile.open(shard) as archive:
            for member in archive:
                key, _, suffix = member.name.rpartition(".")
                if not member.isfile() or suffix not in ("jpg", "txt") or not key:
                    raise ValueError("Unexpected WDS member")
                if member.name in seen:
                    raise ValueError("Duplicate WDS member")
                seen.add(member.name)
                blob = archive.extractfile(member).read()
                sample = samples.setdefault(key, {"id": key, "shard": shard.name})
                if suffix == "jpg":
                    with Image.open(io.BytesIO(blob)) as image:
                        image.convert("RGB").load()
                    sample["image_sha256"] = hashlib.sha256(blob).hexdigest()
                else:
                    sample["caption"] = blob.decode("utf-8")
        for sample in samples.values():
            if "image_sha256" not in sample or not sample.get("caption", "").strip():
                raise ValueError("Unpaired or empty WDS sample")
            records.append(sample)
    if not records:
        raise ValueError("Empty WDS dataset")
    return records


def validate_subset(root):
    from PIL import Image

    summary = json.loads((root / "summary.json").read_text())
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records = [row for row in records if row.get("status") == "success"]
    if not records or len(records) != summary["successful"]:
        raise ValueError("Subset must have nonempty, matching manifest records")
    if len({row["uid"] for row in records}) != len(records):
        raise ValueError("Duplicate subset UID")
    folder = root / "wds"
    shards = sorted(folder.glob("*.tar"))
    if [path.relative_to(root).as_posix() for path in shards] != summary["shards"]:
        raise ValueError("Unexpected subset shards")
    expected = {row["uid"] + suffix for row in records for suffix in (".jpg", ".txt")}
    members = {}
    for shard in shards:
        with tarfile.open(shard) as archive:
            for member in archive:
                if (
                    not member.isfile()
                    or member.name not in expected
                    or member.name in members
                ):
                    raise ValueError("Unexpected or duplicate subset member")
                members[member.name] = archive.extractfile(member).read()
    if members.keys() != expected:
        raise ValueError("Missing subset member")
    for row in records:
        blob = members[row["uid"] + ".jpg"]
        if hashlib.sha256(blob).hexdigest() != row["image_sha256"]:
            raise ValueError("Subset image hash mismatch")
        if (
            members[row["uid"] + ".txt"].decode("utf-8") != row["text"]
            or not row["text"].strip()
        ):
            raise ValueError("Subset caption mismatch")
        with Image.open(io.BytesIO(blob)) as image:
            image.convert("RGB").load()
    return folder, records


def execute_training(command, env, log):
    with subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
    ) as process:
        try:
            process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=30)
            raise
        return process


def run_case(config, output, devices, resume=False):
    world = len(devices.split(","))
    command = [sys.executable]
    if world > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
        ]
    command += [
        "-m",
        "parascale.cli",
        "train",
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    if resume:
        command.append("--resume-latest")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=devices,
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    started = time.monotonic()
    with output.with_suffix(".log").open("w") as log:
        result = execute_training(command, env, log)
    if result.returncode:
        raise RuntimeError(
            f"Training failed ({result.returncode}): {output.with_suffix('.log')}"
        )
    payload = json.loads(output.read_text())
    assert payload["workload"] == "clip_contrastive" and not payload["synthetic"]
    assert payload["optimizer"]["trainable_params"] > 100_000_000
    assert all(math.isfinite(row["loss"]) for row in payload["metrics_history"])
    return payload, time.monotonic() - started


def weight_digest(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    result = hashlib.sha256()
    for name, value in sorted(payload["backend_state"]["model_state_dict"].items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--coco", type=Path)
    source.add_argument("--datacomp-subset", type=Path)
    source.add_argument("--wds", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--find-unused-parameters", action="store_true")
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.scratch.iterdir()) or list(args.output.iterdir()):
        raise ValueError("scratch and output directories must be empty")
    import torch
    import transformers

    if args.wds:
        data = args.wds
        records = inspect_wds(data)
        dataset = f"Existing WDS at {data}, {len(records)} decoded pairs; source provenance must be reviewed separately"
    elif args.datacomp_subset:
        data, records = validate_subset(args.datacomp_subset)
        dataset = f"DataComp small URL-retrieved subset, {len(records)} pairs; NOT full track or historical-byte reproduction"
    else:
        data, records = prepare_data(args.coco, args.scratch)
        dataset = "COCO val2017, first 512 image IDs and first annotation by ID; WDS format, NOT DataComp"
    write_json(
        args.output / "inputs.json",
        {
            "pairs": records,
            "shard_sha256": {p.name: digest(p) for p in sorted(data.glob("*.tar"))},
            "model_files": {
                p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()
            },
        },
    )
    summary = {
        "status": "running",
        "dataset": dataset,
        "find_unused_parameters_override": args.find_unused_parameters,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "storage": "caller-provided scratch; see recorded filesystem evidence",
        "runs": [],
    }
    write_json(args.output / "summary.json", summary)
    for world, devices in ((1, "0"), (2, "0,1"), (4, "0,1,3,4")):
        name = "native_1gpu" if world == 1 else f"native_ddp_{world}gpu"
        config = json.loads(
            Path(f"configs/production/clip_datacomp/{name}.json").read_text()
        )
        run = args.scratch / name
        config["model"]["pretrained_model_name_or_path"] = str(args.model)
        config["data"]["data_dir"] = str(data)
        config["runtime"]["run_dir"] = str(run)
        config["training"].update(
            max_steps=40,
            checkpoint_interval=20,
            checkpoint_dir=str(run / "checkpoints"),
        )
        config["parascale"].update(
            checkpoint_save_path=str(run / "checkpoints"), checkpoint_save_interval=20
        )
        if args.find_unused_parameters and world > 1:
            config["parascale"]["ddp_find_unused_parameters"] = True
        path = args.output / f"{name}.json"
        write_json(path, config)
        print(f"Starting {name} on {devices}", flush=True)
        try:
            original, elapsed = run_case(
                path, args.output / f"{name}.baseline.json", devices
            )
            assert (
                original["global_step"] == 40
                and original["checkpoint_validation"]["ok"]
            )
            checkpoint = run / "checkpoints" / "step-00000040"
            before = weight_digest(checkpoint / "backend_state.pt")
            shutil.copyfile(
                checkpoint / "manifest.json",
                args.output / f"{name}.baseline.manifest.json",
            )
            (checkpoint / "backend_state.pt").write_bytes(b"injected corruption")
            config["training"]["max_steps"] = 20
            resume_path = args.output / f"{name}.resume.json"
            write_json(resume_path, config)
            resumed, recovery_elapsed = run_case(
                resume_path,
                args.output / f"{name}.recovered.json",
                devices,
                resume=True,
            )
            assert (
                resumed["global_step"] == 40 and resumed["checkpoint_validation"]["ok"]
            )
            assert resumed["checkpoint_recovery"]["selected_step"] == 20
            assert resumed["checkpoint_recovery"]["skipped"][0]["step"] == 40
            after = weight_digest(checkpoint / "backend_state.pt")
            shutil.copyfile(
                checkpoint / "manifest.json",
                args.output / f"{name}.recovered.manifest.json",
            )
            summary["runs"].append(
                {
                    "world_size": world,
                    "devices": devices,
                    "status": "passed" if before == after else "recovered_nonidentical",
                    "baseline_wall_seconds": elapsed,
                    "recovery_wall_seconds": recovery_elapsed,
                    "weights_bitwise_equal": before == after,
                    "baseline_weights_sha256": before,
                    "recovered_weights_sha256": after,
                    "last_metrics": original["last_metrics"],
                }
            )
        except Exception as exc:
            summary["runs"].append(
                {
                    "world_size": world,
                    "devices": devices,
                    "status": "failed",
                    "error": str(exc),
                }
            )
        write_json(args.output / "summary.json", summary)
        # Only delete this invocation's isolated scratch subtree after evidence capture.
        if (
            not args.keep_checkpoints
            and summary["runs"][-1]["status"] == "passed"
            and run.exists()
            and run.resolve().is_relative_to(args.scratch.resolve())
        ):
            shutil.rmtree(run)
    summary["status"] = (
        "failed"
        if any(run["status"] == "failed" for run in summary["runs"])
        else "partial"
        if any(run["status"] == "recovered_nonidentical" for run in summary["runs"])
        else "passed"
    )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
