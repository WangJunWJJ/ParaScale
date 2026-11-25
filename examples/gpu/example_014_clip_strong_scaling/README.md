# CLIP Strong-Scaling Qualification

Compare ParaScale's actual `TrainEngine` against direct Hugging Face/PyTorch DDP
with a fixed global contrastive batch. This example reuses the numerical and
timing tools under `tests/benchmarks/scripts`; it does not implement another
trainer. For ordinary training/checkpoint commands, see the
[CLIP/DataComp configuration guide](../../../configs/production/clip_datacomp/README.md).

## Inputs and Environment

- Run from the repository root in a Linux CUDA environment with PyTorch,
  Transformers, Pillow and the project's CLIP dependencies installed.
- The bounded reference environment is PyTorch 2.9.1+cu126, Transformers 4.51.3,
  A6000, and a complete local `openai/clip-vit-base-patch32` checkpoint including
  processor/tokenizer files. Other environments require their own qualification.
- Supply existing local JPG/TXT WebDataset shards. At least 128 valid paired
  samples are required for batch128; the first caption line is used. DataComp or
  another licensed image/text dataset can be used, but they are not interchangeable
  evidence of dataset quality. These commands do not download assets.
- Use a new writable output directory outside the repository. Stop unrelated GPU
  jobs first. Select physical GPU IDs appropriate to the host topology; the
  example's four-card selection is specific to the tested A6000 host.

## Numerical Gate, Then Timing

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
MODEL=/models/clip
DATA=/datasets/image_text/train
OUT=/runs/clip-strong-scaling-$(date +%Y%m%d-%H%M%S)
BATCH=128
mkdir "$OUT"

for WORLD in 1 4; do
  if [ "$WORLD" = 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    export CUDA_VISIBLE_DEVICES=0,1,3,4
  fi
  python -m torch.distributed.run --nnodes=1 --nproc_per_node="$WORLD" \
    --master_addr=127.0.0.1 --master_port=29500 \
    --module tests.benchmarks.scripts.validate_contrastive_training \
    --model "$MODEL" --data-dir "$DATA" --global-batch "$BATCH" \
    --output "$OUT/parity-w$WORLD.json"
done

for WORLD in 1 4; do
  if [ "$WORLD" = 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    export CUDA_VISIBLE_DEVICES=0,1,3,4
  fi
  for IMPL in pytorch parascale; do
    python -m torch.distributed.run --nnodes=1 --nproc_per_node="$WORLD" \
      --master_addr=127.0.0.1 --master_port=29500 \
      --module tests.benchmarks.scripts.benchmark_contrastive_training \
      --implementation "$IMPL" --model "$MODEL" --data-dir "$DATA" \
      --global-batch "$BATCH" --output "$OUT/$IMPL-w$WORLD.json"
  done
done
```

Both tools admit global batch32 or128; never compare different global batches
as the same contrastive objective. Single-rank uses native and multi-rank uses
native DDP. Every rank starts from the same checkpoint; FP32, eager attention,
SGD(lr=1e-4, momentum=0.9), TF32 off, GAS1 and uncompressed communication are fixed.
Ordinary gradient accumulation does not enlarge the cross-example negative pool.

The numerical gate compares the global loss, every parameter gradient and one
optimizer update against an independent native HF global-batch reference with
`atol=1e-5, rtol=1e-4`. A nonzero exit stops the example before timing.

Timing uses 10 warmups and three windows of 50 updates. The denominator is the
slowest rank's wall time, and the numerator is actual global pairs processed.
Check input/initial-weight/source hashes, finite states and replica consistency
before comparing results. These are the same preprocessed pairs resident on GPU
and repeated; decoding, tokenization, H2D, setup and boundary finite scans are
excluded. This is **not** end-to-end pipeline throughput or convergence evidence.

## Interpretation and Stop Rules

- Strong scaling: compare four-card versus single-card ParaScale throughput at
  the same global batch. Report speedup and speedup/4 efficiency.
- Framework overhead: compare ParaScale versus direct PyTorch at the same card
  count, batch, checkpoint and precision. Multi-GPU speedup alone does not show
  an advantage over PyTorch.
- The first run is a screen. Before claiming a repeatable improvement, repeat
  with fresh output directories and alternate implementation order (AB/BA/AB).
  Predeclare acceptable overhead and useful-scaling thresholds before running.
- If a screen fails, preserve it and diagnose before changing parameters.
  `--profile-dir "$OUT/traces"` captures three warmed-up updates on each rank.
  Profiled reports have `performance_claim_eligible=false`; never combine their
  timings with ordinary throughput results.
- Larger batches change negative-pool size and potentially optimization behavior.
  Held-out retrieval quality, learning-rate selection, fresh-data throughput,
  checkpoint recovery and long-run stability require separate acceptance.
