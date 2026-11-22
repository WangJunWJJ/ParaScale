# CLIP Training and Held-Out Retrieval

Run a bounded real-data training/checkpoint/inference acceptance chain. This is
an engineering non-regression check, not a DataComp benchmark, SOTA claim or
production certification. The validation split is held out from this fine-tuning
run; overlap with CLIP's original pretraining data is unknown.

## Prepare Pinned Data

Use `jxie/flickr8k` revision `56f58c967835f7c508d684f36bd7897cca9d7634`.
Download these files into a local raw-data directory using the indicated names:

| Local Name | Pinned Source |
| --- | --- |
| `train-0.parquet` | [Train Part 0](https://huggingface.co/datasets/jxie/flickr8k/resolve/56f58c967835f7c508d684f36bd7897cca9d7634/data/train-00000-of-00002-2f8f6bfa852eac4b.parquet) |
| `train-1.parquet` | [Train Part 1](https://huggingface.co/datasets/jxie/flickr8k/resolve/56f58c967835f7c508d684f36bd7897cca9d7634/data/train-00001-of-00002-2173151d8cd6c7fb.parquet) |
| `validation.parquet` | [Validation](https://huggingface.co/datasets/jxie/flickr8k/resolve/56f58c967835f7c508d684f36bd7897cca9d7634/data/validation-00000-of-00001-7025a2b596f14b7b.parquet) |

The dataset card lacks complete provenance/licensing information. These inputs
are for internal engineering evaluation; obtain appropriate data rights before
commercial use. The preparation tool does not download data and verifies fixed
SHA256 values before decoding. It needs PyArrow and Pillow in the preparation
environment; the GPU training/evaluation environment does not need PyArrow.

The pinned release contains one exact duplicate inside training: source row 1359
duplicates row 1027 with different captions. The tool merges their annotations,
retains source-row lineage, and trains on the canonical first caption. All other
byte/RGB duplicates, including cross-split overlap, reject. Result: 5999 unique
training images and the unchanged 1000 validation images/5000 captions. This does
not establish absence of perceptual near-duplicates. The test split is untouched.

From the repository root:

```bash
python -m tests.benchmarks.tools.prepare_flickr8k_retrieval \
  --source /datasets/flickr8k-raw --output /datasets/flickr8k-qualified
```

The output directory must be new. Preserve failed outputs; do not disable checksum
or split checks to proceed. When transferring, verify every shard against
`manifest.json` before training. Keep datasets/checkpoints/results outside Git.

## Bounded Acceptance

Use a complete local CLIP ViT-B/32 checkpoint, four CUDA GPUs, and a new run path.
The reference environment is Torch 2.9.1+cu126 and Transformers 4.51.3 on A6000.
Adapt GPU IDs to the actual topology; do not run overlapping benchmark jobs.

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export PARASCALE_MODEL=/models/clip
export PARASCALE_DATA=/datasets/flickr8k-qualified
export PARASCALE_RUN_DIR=/runs/clip-quality-$(date +%Y%m%d-%H%M%S)
mkdir -p "$PARASCALE_RUN_DIR"

CUDA_VISIBLE_DEVICES=0 python -m tests.benchmarks.scripts.validate_clip_retrieval \
  --model "$PARASCALE_MODEL" --data "$PARASCALE_DATA" \
  --output "$PARASCALE_RUN_DIR/pretrained.json"

CUDA_VISIBLE_DEVICES=0,1,3,4 python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 \
  -m parascale.cli train \
  --config examples/gpu/example_015_clip_retrieval_quality/train.json \
  --output "$PARASCALE_RUN_DIR/train-result.json"

CUDA_VISIBLE_DEVICES=0 python -m tests.benchmarks.scripts.validate_clip_retrieval \
  --model "$PARASCALE_MODEL" --data "$PARASCALE_DATA" \
  --checkpoint-root "$PARASCALE_RUN_DIR/checkpoints" --checkpoint-step 50 \
  --baseline "$PARASCALE_RUN_DIR/pretrained.json" \
  --output "$PARASCALE_RUN_DIR/trained.json"
```

Training uses the product CLI, fresh image decoding/tokenization, FP32 native-DDP,
32 pairs/rank, GAS 1, global batch 128, AdamW lr 1e-6 and 50 optimizer updates. It is
not the cached-input speed benchmark in Example 014. The checkpoint is validated
and reloaded in a fresh process, with strict model state loading. Use checkpoints
from this trusted run only.

The evaluator uses the product `HFClipInferenceModel` and bounded `InferenceRunner`
callback, retaining only the final feature gallery. It verifies first-batch native
HF feature parity, all feature norms/finiteness, complete gallery coverage and
fixed preprocessing/data identities. It computes both directions' R@1/5/10 with
all five captions positive for their owning image and stable ordering for ties.
The pretrained baseline is recorded before training and must match the original
model weights, not another training checkpoint. Neither direction's R@1 may
drop by more than one absolute percentage point; failed gates return nonzero.

The adapter's `image_embeddings` and `text_embeddings` are normalized float lists
usable for downstream retrieval, in addition to its diagnostic shape/mean fields.
Long text is truncated to the checkpoint's text-context limit. Python callers can
use unequal image/text counts; the generic CLI's existing real-CLIP input generator
is still a demonstration, not a dataset reader. This example supplies actual
inputs through the reusable evaluator rather than claiming that demonstration
generates real-data evidence.

Passing this short check does not certify 24h stability, exact interrupted resume,
HTTP serving, distributed inference or deployment-specific quality requirements.

## Optional CPU Lookahead

`data.cpu_prefetch: true` prepares at most one following batch in a CPU thread,
overlapping original-image decoding and checkpoint preprocessing with training.
It defaults to false. This option requires pretrained CLIP/SigLIP, streaming WDS,
`num_workers: 0`, and no CUDA prefetch. It does not change the physical batch,
sample order or preprocessing algorithm. The checkpoint cursor advances only
for consumed batches, not speculative work.

The transform must be deterministic CPU-only code that does not use global RNG.
Do not add random augmentation or concurrent processor mutation to this path.
Ending a fit cancels queued work and joins running preprocessing; it cannot
interrupt a stalled decoder or storage read, so shutdown time is not bounded.

The following Linux/CUDA protocol reuses the environment variables above. Run
the input-equivalence gate before the timing comparison, with no competing jobs:

```bash
set -euo pipefail
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29500 \
  -m tests.benchmarks.scripts.profile_clip_input \
  --model "$PARASCALE_MODEL" --data "$PARASCALE_DATA/train" \
  --output "$PARASCALE_RUN_DIR/input-parity" --verify-prefetch

CUDA_VISIBLE_DEVICES=0,1,3,4 python -m tests.benchmarks.scripts.validate_clip_input_prefetch \
  --config examples/gpu/example_015_clip_retrieval_quality/train.json \
  --output "$PARASCALE_RUN_DIR/input-timing"
```

The input gate compares all tensor bytes for 25 batches/rank. Timing uses three
alternating-order pairs of fresh product CLI launches, 50 updates each, with the
first 10 excluded from steady-state metrics. It also reports whole-fit throughput.
Checkpoints are disabled only in this timing protocol. A speed gate passing does
not replace trained-checkpoint quality, distributed recovery or long-run gates.

## Fresh-Process Recovery Gate

The following Linux/CUDA gate requires at least 20 GiB free space. It trains
50 updates with CPU lookahead, saves updates 25 and 50, copies the validated
update-25 checkpoint into a separate run, and resumes 25 updates in fresh ranks.
Training shards and model assets must match the evaluation inputs. It checks
Adam moments and exact optimizer step counts, per-rank cursor/RNG, and held-out
retrieval for both trained checkpoints. This is a correctness test, not a speed
benchmark; checkpoint I/O is enabled.

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 python -m tests.benchmarks.scripts.validate_clip_prefetch_recovery \
  --config examples/gpu/example_015_clip_retrieval_quality/train.json \
  --model "$PARASCALE_MODEL" --data "$PARASCALE_DATA" \
  --output "$PARASCALE_RUN_DIR/recovery"
```

The output directory must be new. Model/moment comparisons use fixed
`atol=1e-5, rtol=1e-4`; progress and RNG must match exactly. A failed state gate
remains a failure even when retrieval passes. Preserve its checkpoints and logs
to localize the first divergent update instead of relaxing the tolerance.
`--check-only` revalidates the existing artifacts without training again, binding
retrieval reports to current checkpoint hashes and verifying the restore origin.

For a failed recovery gate, a bounded reducer diagnostic reuses the update-25
checkpoint. Run cold then warm in separate four-rank launches with a fresh
diagnostic directory and at least 10 GiB additional free space:

```bash
set -euo pipefail
for mode in cold warm; do
  CUDA_VISIBLE_DEVICES=0,1,3,4 timeout 600 python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 \
    -m tests.benchmarks.scripts.probe_clip_reducer_recovery \
    --config "$PARASCALE_RUN_DIR/recovery/continuous.config.json" \
    --checkpoints "$PARASCALE_RUN_DIR/recovery/continuous/checkpoints" \
    --output "$PARASCALE_RUN_DIR/reducer-probe" --mode "$mode"
done
```

This fixed FP32/native-DDP diagnostic performs only two updates per launch.
The warm arm first runs one backward without an optimizer update, then restores
model, optimizer, cursor and RNG without replacing DDP. It compares first-step
local gradients before reduction, reduced gradients, and resulting parameters
and Adam moments. Per-rank JSON files report differences, not a pass certificate.
Snapshots remain outside Git. The probe uses private PyTorch DDP diagnostic
state; use the validated PyTorch 2.9.1 environment. Its instrumentation overhead
must not be used as throughput evidence, and its warmup is not a product fix.
