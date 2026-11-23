# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded operator profile of the real, stateful CLIP input pipeline."""

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import time
from pathlib import Path

from parascale.workloads.clip import build_clip_contrastive_components
from parascale.workloads.specs.clip import ClipContrastiveSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-prefetch", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"rank-{rank}.json"
    if output.exists():
        parser.error("rank output already exists")
    spec = ClipContrastiveSpec(
        model_type="hf_clip",
        data_type="datacomp_wds",
        data_dir=args.data,
        streaming=True,
        num_workers=0,
        batch_size=32,
        num_batches=25,
        num_samples=800,
        text_length=77,
        pretrained_model_name_or_path=args.model,
    )
    model, optimizer, batches, _ = build_clip_contrastive_components(spec)
    del model, optimizer
    if args.verify_prefetch:
        import torch

        initial = batches.state_dict()
        rng = torch.get_rng_state().clone()
        digests = []
        for enabled in (False, True):
            batches.load_state_dict(initial)
            batches.prefetch = enabled
            hashes = []
            for batch in batches:
                digest = hashlib.sha256()
                for key, value in sorted(batch.items()):
                    digest.update(key.encode())
                    digest.update(str((value.dtype, value.shape)).encode())
                    digest.update(value.contiguous().numpy().tobytes())
                hashes.append(digest.hexdigest())
            digests.append(hashes)
        assert len(digests[0]) == 25 and digests[0] == digests[1]
        assert torch.equal(rng, torch.get_rng_state())
        with output.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "status": "passed",
                    "rank": rank,
                    "batch_sha256": digests[0],
                    "state": batches.state_dict(),
                    "torch_cpu_rng_unchanged": True,
                },
                stream,
                indent=2,
            )
        return
    iterator = iter(batches)
    for _ in range(5):
        next(iterator)
    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    for _ in range(20):
        next(iterator)
    profile.disable()
    elapsed = time.perf_counter() - started
    profile.dump_stats(str(args.output / f"rank-{rank}.pstats"))
    functions = [
        {
            "file": file,
            "line": line,
            "function": name,
            "calls": calls,
            "self_seconds": own,
            "cumulative_seconds": cumulative,
        }
        for (file, line, name), (_, calls, own, cumulative, _) in pstats.Stats(
            profile
        ).stats.items()
    ]
    result = {
        "rank": rank,
        "profiled_batches": 20,
        "batch_size": 32,
        "profiled_seconds": elapsed,
        "state": batches.state_dict(),
        "top_self": sorted(functions, key=lambda f: f["self_seconds"], reverse=True)[
            :20
        ],
        "top_cumulative": sorted(
            functions, key=lambda f: f["cumulative_seconds"], reverse=True
        )[:20],
        "scope": "CPU input-only diagnostic; profiler overhead included, not training throughput",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
