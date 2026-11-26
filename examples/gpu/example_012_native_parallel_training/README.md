# Native Parallel TinyLlama Training

Run genuine pretrained TinyLlama 1.1B on WikiText training text through the unified
`causal_lm` CLI with `training_backend: native_parallel`. This example is not a
synthetic model, a benchmark training loop, or an acceleration claim. It targets
one Linux CUDA/NCCL node with 1, 2, or 4 GPUs, one torchrun process per GPU.

## Environment and Local Assets

From the repository root, install the existing parallel extra and the separately
pinned stage-model dependency in a dedicated environment:

```bash
python -m pip install -e '.[parallel]' 'transformers==4.51.3' sentencepiece
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__); assert torch.__version__.split("+")[0] == "2.9.1"; assert transformers.__version__ == "4.51.3"'
```

The `parallel` extra pins Torch 2.9.1 and torchvision 0.24.1; it does not pin
Transformers, hence the explicit 4.51.3 requirement above. Use a CUDA-enabled
Torch build compatible with the host driver. BF16 requires hardware support.
CPU planning does not qualify CUDA execution or establish available GPU memory.

Provide these assets before training; ParaScale does not download them:

- `/models/tinyllama`: a complete local copy of the genuine pretrained
  `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` checkpoint, including
  `config.json`, matching tokenizer assets, and all original safetensors or
  PyTorch `.bin` weights. If sharded, include the weight index and every named
  shard. Do not substitute a random tiny fixture or adapter-only checkpoint.
- `/data/wikitext/train.txt`: actual WikiText training-split text exported as
  UTF-8, one document per line. For example use the `wikitext-2-raw-v1` training
  split, retaining each nonempty text record as a line. Do not use validation or
  test data as training input; do not feed dataset JSON, CSV, or parquet directly
  to `prepare-text`. Obtain assets separately under their applicable licenses.
- Writable `/data/checkpoints` and `/data/runs`, outside the source tree.

The YAML configurations use portable example paths, not environment-variable expansion.
Edit the model, corpus, checkpoint, and run paths to match your machine. Keep the
same model/tokenizer and immutable prepared corpus accessible to every rank.
Trusted model-file symlinks, including HF cache blob links, are allowed; targets
must not change while loading or hashing. Each rank initially loads the full
model on CPU before keeping its stage, so budget host RAM as well as GPU RAM.

## Prepare the Token Corpus

```bash
python -m parascale.cli prepare-text \
  --input /data/wikitext/train.txt \
  --tokenizer /models/tinyllama \
  --output /data/wikitext-tokens
```

The output directory must not already exist. The tokenizer must be loadable
locally with `use_fast=True`, `trust_remote_code=False`, and a valid EOS token.
Use the exact tokenizer shipped with the model. The preparation command strips
each line, skips empty lines, tokenizes without automatic special tokens, and
appends one EOS per document. It rejects lines over 1 MiB by default; the actual
parser provides `--max-line-bytes` to change that bound explicitly.

Preparation writes an immutable token payload and manifest with tokenizer,
source, payload, and packing identity. Training validates the tokenizer identity
and vocabulary range. This packing differs from tokenizing an entire file as one
string. Comparisons must reuse the same token corpus and prediction-token count.
Do not edit the manifest or tokens in place; prepare a fresh directory for changed
data or tokenizers. Use enough tokens for complete fixed-size DP microbatches.
Reads are memory-mapped with no DataLoader workers. Shuffle is window-bounded,
not a full-corpus permutation; incomplete global batches are dropped equally.

## Plan, Dry Run, and Train

The two supplied configurations use BF16 computation with FP32 master parameters,
AdamW, SDPA attention, microbatch size 4, sequence length 256, and accumulation 4.
They perform 20 optimizer updates and checkpoint every 10 updates. These are
conservative starting settings, not a memory guarantee or measured optimum.
Reduce batch size or sequence length in a fresh configuration if necessary.

```bash
EXAMPLE=examples/gpu/example_012_native_parallel_training
python -m parascale.cli plan --config "$EXAMPLE/single_gpu.yaml" --json
python -m parascale.cli train --config "$EXAMPLE/single_gpu.yaml" --dry-run
python -m parascale.cli plan --config "$EXAMPLE/dp2_pp2.yaml" --json
python -m parascale.cli train --config "$EXAMPLE/dp2_pp2.yaml" --dry-run

# One GPU: DP1 TP1 PP1, 16 distinct sequences per optimizer update.
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  -m parascale.cli train --config "$EXAMPLE/single_gpu.yaml"

# Four GPUs: DP2 TP1 PP2, 32 distinct sequences per optimizer update.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m parascale.cli train --config "$EXAMPLE/dp2_pp2.yaml"
```

Planning and dry-run validate configuration, not asset availability, checkpoint
compatibility, GPU memory, or throughput. Even the one-GPU native_parallel run
must use torchrun. World size must equal DP * TP * PP. Do not use `native_ddp`
launch wrappers for this backend.

For a two-GPU run, copy `single_gpu.yaml` to a new file and edit these fields:

```yaml
# Fields to change in the copied file; this fragment is not a complete config.
parascale:
  data_parallel_size: 2
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  checkpoint_save_path: /data/checkpoints/tinyllama-dp2-tp1-pp1
runtime:
  run_dir: /data/runs/tinyllama-dp2-tp1-pp1
```

```bash
cp "$EXAMPLE/single_gpu.yaml" /data/tinyllama-dp2.yaml
# Edit /data/tinyllama-dp2.yaml with the changes above before these commands.
python -m parascale.cli plan --config /data/tinyllama-dp2.yaml --json
python -m parascale.cli train --config /data/tinyllama-dp2.yaml --dry-run
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m parascale.cli train --config /data/tinyllama-dp2.yaml
```

Topology changes are file edits, not unsupported `--dp`, `--tp`, or `--pp` CLI
overrides. Accepted (DP, TP, PP) layouts are `(1,1,1)`, `(2,1,1)`, `(4,1,1)`,
`(1,2,1)`, `(1,1,2)`, `(2,2,1)`, `(2,1,2)`, and `(1,2,2)`. The product is at
most four ranks. **DP2 TP2 PP2 is eight ranks and is not supported.** Use a new
checkpoint and run directory for each topology. The single-GPU configuration
uses GPipe; the four-GPU DP2/PP2 configuration selects `1f1b`. That schedule
requires PP > 1 and at least as many accumulated microbatches as pipeline stages;
the supplied accumulation of 4 satisfies PP2.

This bounded stage path does not enable activation checkpointing, FP16,
quantization, four-bit optimizer state, ZeRO/FSDP/DeepSpeed, or multi-node runs.
Do not copy unrelated backend knobs into these YAML files.

## Matched Single-GPU and PP2 Comparison

`matched_single.yaml` and `matched_pp2.yaml` provide a separate controlled pair:
global batch 64, physical microbatch 4, accumulation 16, sequence length 256,
BF16 compute with FP32 master parameters/AdamW state, and 16 updates. Only
pipeline topology/schedule and output directories differ. Both use the same
model and immutable token corpus paths described above. The original introductory
configurations remain unchanged and have different global batches.

Use a corpus with at least 262,144 prepared tokens (1,024 complete blocks), and
fresh run/checkpoint directories. Shuffle and repetition are disabled to compare
the same global input prefix; this is a measurement recipe, not a recommended
long-run data-order policy. Keep the two jobs sequential on an otherwise idle
node. If paths differ from your machine, edit both configurations consistently.

```bash
EXAMPLE=examples/gpu/example_012_native_parallel_training
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 \
  --master_addr=127.0.0.1 --master_port=29500 \
  -m parascale.cli train --config "$EXAMPLE/matched_single.yaml" \
  --output /data/runs/matched-single-summary.json
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 \
  -m parascale.cli train --config "$EXAMPLE/matched_pp2.yaml" \
  --output /data/runs/matched-pp2-summary.json
```

The explicit loopback rendezvous avoids relying on container hostname resolution;
it is for one node only. Ensure the port is unused. Provision persistent storage
for real checkpoints, including retained snapshots when resuming. An ephemeral
container memory filesystem is suitable only for disposable timing experiments,
not a substitute for durable recovery.

For steady training-span throughput, exclude the first four `metrics_history`
entries and divide the remaining `scored_tokens` sum by the corresponding
`training_seconds` sum: 195,840 scored tokens over 12 measured updates. This
span includes gradient validation, coordination and the optimizer, not data
transfer or checkpoint writes. Separately report `elapsed_seconds`,
`checkpoint_seconds`, peak memory, and externally measured process wall time.
Do not call this training-span number complete-job throughput. The final
checkpoint is genuinely written and can dominate a short 16-update job.

This recipe was exercised with pretrained TinyLlama on RTX A6000; results depend
on hardware, interconnect and software versions. PP2 is a useful candidate, not
a universal winner over DDP or mixed parallelism. Short training losses do not
establish held-out quality, convergence equivalence, or LLaMA-7B qualification.
For repeated measurements, use new output roots rather than resuming the previous
weights. Compare only matching input/model hashes and timing scopes.

## Deduplicated DP Checkpoints

`matched_dp2_dcp.yaml` uses two DP replicas, microbatch 4 and accumulation 8,
keeping global batch 64, sequence length 256 and 16 updates from the matched
recipes above. It opts into `parascale.checkpoint_format: dcp`; the default
`rank` format and the existing examples remain unchanged.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 \
  -m parascale.cli train --config "$EXAMPLE/matched_dp2_dcp.yaml" \
  --output /data/runs/matched-dp2-dcp-summary.json
```

Edit asset/output paths as above before running. For a matched storage baseline,
copy this configuration, set `checkpoint_format: rank`, and use separate fresh
checkpoint/run/output paths. Do not compare against a different GPU count or
change the training budget. Compare `checkpoint_seconds` and complete invocation
time separately from training-span throughput; this option optimizes checkpoint
I/O, not the model's forward/backward computation. Repeat sequentially with
alternating order on an otherwise idle node. Keep real checkpoint writes and
their independent integrity verification in both measurements.

DCP stores one copy of shared model/AdamW state across DP replicas while retaining
each rank's RNG, optimizer binding and training progress. It checks replica bytes
before deduplication. This is lossless storage deduplication, not model quantization
or optimizer-state sharding. It currently requires `native_parallel`, DP >= 2,
TP = 1 and a supported topology; single-GPU/TP paths use `rank`. Plain CPU/CUDA
tensors and standard containers are supported; custom tensor subclasses and
tensor-valued module metadata are rejected rather than silently discarded.

Use trusted checkpoints in a dedicated, shared filesystem without links or
concurrent mutations. Schema6 checks sizes and SHA256 hashes before loading into
temporary CPU targets, and requires the exact saved Torch version and topology.
Allow host memory for these targets and storage for retained and pending snapshots.
Loading dispatches by the manifest; `checkpoint_format` controls subsequent saves.
Retention never deletes snapshots from a different format/version family, so a
format switch can retain more snapshots than `checkpoint_keep_last` alone suggests.

Storage restoration and subsequent execution determinism are different contracts.
Loaded state must match exactly. With four DP ranks, fresh DDP and warmed DDP can
use different gradient buckets: tests reproduced post-update bit differences with
both `rank` and `dcp`, while cross-format restored executions matched. Do not claim
universal bitwise continuation or infer quality equivalence from matching loss.
This example does not change DDP settings to hide that boundary. Two-rank exact
recovery and four-rank cross-format equivalence do not replace held-out quality
or long-run production qualification.

## Opt-In BF16 Gradient Communication

`matched_dp4_bf16_comm.yaml` exposes PyTorch's official BF16 gradient compression
through the real training CLI. It keeps global batch 64, microbatch 4, length 256
and 16 updates, using four DP replicas and accumulation 4. FP32 master parameters
and AdamW state remain unchanged. This is lossy gradient communication, not INT8
model quantization. Native BF16 hardware and NCCL newer than 2.9.6 are required.

```bash
EXAMPLE=examples/gpu/example_012_native_parallel_training
python -m parascale.cli plan --config "$EXAMPLE/matched_dp4_bf16_comm.yaml" --json
CUDA_VISIBLE_DEVICES=0,1,3,4 torchrun --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29500 \
  -m parascale.cli train --config "$EXAMPLE/matched_dp4_bf16_comm.yaml" \
  --output /data/runs/matched-dp4-bf16-comm-summary.json
```

Edit GPU IDs and paths for your machine. For a fair control, copy the file, change
only `ddp_comm_hook` to `none` and output/checkpoint directories, then run fresh
jobs sequentially. Compare training-span throughput, complete invocation time,
checkpoint time, peak memory and held-out quality separately. Short-run numeric
screens do not establish convergence equivalence. Compression can increase peak
memory because of temporary reduction buffers; it is not a memory-fit guarantee.

Only pure DP2/DP4 supports this option. TP/PP combinations and FP16 communication
are rejected, not silently downgraded. Omitted/`auto` resolves to `none` on this
stage backend; no automatic compression recommendation is made.

The effective communication policy is part of checkpoint identity, including
`none`. A different policy or an older snapshot missing this identity is rejected
before loading payloads, even with `--resume-latest`. Do not patch old manifests
to bypass validation. Start a fresh run to change policy. Resume also requires
the same implementation/runtime identities described below; lossy communication
does not remove the warmed-versus-fresh DDP bucket determinism boundary.

## Checkpoint and Resume

Use the canonical `parascale.checkpoint_save_path`, `checkpoint_save_interval`,
and `checkpoint_keep_last` fields. Retention is 2 (minimum 2); metric history is
bounded to 20 entries. Do not replace the checkpoint path with the ordinary
backend's `training.checkpoint_dir` alias. New training requires a fresh checkpoint
root; resuming requires the same topology and compatible training identity.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  -m parascale.cli train --config "$EXAMPLE/single_gpu.yaml" --resume-latest

# Alternative: select a retained complete step explicitly, not together with latest.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m parascale.cli train --config "$EXAMPLE/dp2_pp2.yaml" --resume-step 20
```

In the introductory configurations, `training.max_steps: 20` means
**20 additional updates per invocation**, including
resume: resuming step 20 finishes at step 40. It is not a target absolute step.
Canonical checkpoint directories are `step-00000010`, `step-00000020`, etc.
Latest discovery checks complete checkpoints and reports skipped invalid
candidates, but fallback selection is not automatic recovery. If newer snapshot
directories remain above the selected checkpoint, CLI preflight rejects the run
before training to avoid immutable checkpoint-name collisions and a stale
retention anchor. Invalid snapshots are preserved, not overwritten or removed
automatically. Explicitly relocate the conflicting snapshot directories outside
the checkpoint root, preserving their contents for inspection, before retrying;
alternatively start a new run with a fresh checkpoint root. Do not rename step
numbers or edit checkpoint payloads. Retention preserves recovery candidates;
keep independent backups for long-term archival.

Resume identity covers effective model configuration and selected local weight
hashes, tokenizer/corpus, precision, causal loss, requested optimizer settings,
and execution versions/source hashes. Changing LR/betas/eps is rejected rather
than silently ignored when saved optimizer hyperparameters are restored. Keep
data seed, batching, accumulation, topology, and implementation unchanged for
same-run recovery. No scheduler is configured here; when adding one, keep its
full-run horizon stable across resumed invocations.

## Native PyTorch DDP Reference

For an executor-level comparison, run the repository benchmark against the
original Hugging Face model under native PyTorch DDP and the ParaScale stage
executor. This is a diagnostic companion to the training CLI, not a replacement
production training entry point. It writes measurements, not training checkpoints.

Provide `/data/wikitext/train.txt` and `/data/wikitext/validation.txt` as UTF-8
text from distinct dataset splits. This benchmark tokenizes each complete file
without added special tokens; it does not consume the EOS-per-document corpus
above. Keep this benchmark's results separate from that differently packed data.
It evaluates a fixed validation window, not the entire split or final test set.

From the repository root, with the same pinned environment and model assets:

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
RUN=/data/runs/native-ddp-reference-r0
mkdir -p "$RUN"
for driver in native parascale; do
  torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=29500 \
    -m tests.benchmarks.scripts.benchmark_llama_parallel \
    --model /models/tinyllama --data /data/wikitext \
    --driver "$driver" --dp 4 --tp 1 \
    --global-batch 64 --microbatch 4 --sequence-length 256 \
    --steps 16 --warmup 4 --validation-tokens 32768 \
    --precision bf16 --adam-implementation auto \
    --output "$RUN/$driver.json"
done
```

Select the actual GPU IDs for your node. For one or two GPUs, change both the
visible GPU list and `--nproc_per_node`/`--dp`, keeping the other budgets fixed.
Accumulation is 16, 8 or 4 for DP1, DP2 or DP4. Native DDP uses `no_sync()` for
all but the last microbatch; both implementations use the same AdamW settings.
The reference intentionally rejects combined DP/TP; that restriction does not
remove the product's separately validated combined topologies.

The outputs are diagnostic records, not an automatic paired acceptance verdict.
Check model/source/global-token identities, corresponding per-rank token hashes,
every step loss, and initial/final validation NLL before interpreting speed.
Validation is replicated across DP ranks: do not multiply its scored-token count
by GPU count. Use new output files for each fresh run; repeat at least three pairs
on an otherwise idle node and alternate driver order.

`scored_tokens_per_second` uses synchronized forward/backward and optimizer time,
including loss communication and executor checks, and the slowest rank per step.
It excludes setup, preprocessing, H2D, evaluation, checkpoint I/O and the full CLI
orchestration loop. Peak allocated memory includes warmup; throughput excludes
the first four updates in this command. Matching short-run NLL is not a long-run
convergence guarantee, and a smaller model or lower precision is not a fair
same-workload acceleration baseline.

For an explicit communication experiment, keep an uncompressed ParaScale run
(`--driver parascale --ddp-comm-hook none`) and repeat with
`--driver parascale --ddp-comm-hook bf16_compress` and
`--driver native --ddp-comm-hook bf16_compress`, using a fresh output path for
each arm. This diagnostic option requires pure DP with at least two GPUs,
native CUDA BF16 support and NCCL newer than 2.9.6. It uses PyTorch's official
BF16 gradient hook on the wrapper's own process group; FP32 parameters and
optimizer state are unchanged. The ParaScale arm uses the same constructor option
as the training CLI; the native arm registers the official hook directly. This
is not an automatic recommendation. Define loss/NLL tolerances before testing:
lossy gradient communication need not reproduce uncompressed updates exactly.
Compare the two compressed implementations separately from compression versus
no compression. Initial parameter fingerprints are recorded before the first
forward, outside timing; matching source filenames alone is insufficient.
This benchmark does not qualify checkpoint recovery with compression enabled.

## Measurement and Qualification

### Independent Validation

`matched_dp4_bf16_comm.yaml` evaluates a separate corpus before and after training
(after checkpoint restoration when resuming). Prepare genuine validation-split
text separately, with the same tokenizer and EOS-per-document policy:

```bash
python -m parascale.cli prepare-text \
  --input /data/wikitext/validation.txt \
  --tokenizer /models/tinyllama \
  --output /data/wikitext-validation-tokens
```

The optional root `evaluation` section requires `data_path` and `max_batches`.
Each evaluation batch is one complete accumulation window, not one microbatch.
The supplied budget evaluates the first 128 distinct blocks: 32,768 input tokens
and 32,640 next-token targets at length256, with no shuffled or repeated samples.
For matched comparisons across layouts, keep the global window size and evaluated
block prefix identical. Remove the entire section to disable evaluation.

Results appear under `evaluation.initial` and `evaluation.final`, including
global step, corpus identity, block coverage, excluded tokens, token-weighted NLL,
perplexity and separate evaluation seconds. These are disjoint-block metrics,
not standard sliding-window WikiText perplexity. Identical training/validation
payload or source hashes are rejected; this does not establish semantic
deduplication or absence of pretraining contamination. Evaluation restores the
training RNG and uses a separate cursor. Its time is excluded from update-loop
training throughput but belongs in externally measured complete CLI wall time.
Source-bound checkpoints from a previous implementation revision cannot be
resumed by editing their manifest; use the matching source or a fresh run.

FP32 masters, gradients, and AdamW state still consume substantial memory even
with BF16 computation; PP partitions layers but DP replicates their states.
Reported valid prediction tokens use `batch * (sequence_length - 1)` per
microbatch and count DP replicas, not duplicate TP/PP work. The introductory
configurations use global batch `4 * 4 * DP`; the matched pair uses `4 * 16 = 64`.
Neither is multiplied by TP or PP.

End-to-end update-loop timing includes data/transfer and checkpoint I/O; inspect
the separate timing fields before comparing compute speed. Startup model loading
and identity hashing are outside that update-loop measurement. No acceleration,
quality, memory-fit, or GPU qualification claim follows from planning or these
configurations. Real model training, resume parity, and the pinned real-stage
identity integration test require separate verification on the target environment.
