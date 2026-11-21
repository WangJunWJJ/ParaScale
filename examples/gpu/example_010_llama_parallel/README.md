# Complete Llama Parallel Validation

This example trains a complete Llama model through integer tokens, embedding,
decoder, norm, output head and next-token loss. It compares each distributed
update with an independent Hugging Face reference (unsharded by default). It is a correctness
validator, not an end-to-end throughput benchmark or a long-running training CLI.

## Environment and Assets

Use one CUDA/NCCL node with one, two or four GPUs and an isolated environment:

```bash
python -m pip install -e '.[parallel]' 'transformers==4.51.3'
```

Use trusted local safetensors weights and matching tokenizer files. Pinned asset
identities are recorded independently of the generated test logs:

- SmolLM2-135M: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- TinyLlama: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`, revision
  `59f6f375b26bde864a6ca194a9a3044570490064`. Its FP32 safetensors file is
  4,400,216,536 bytes, SHA-256
  `9dc1c1a8c6b29e153dea7aaf2ed6c4d242163489add0e7c14c23889745437be4`.
- WikiText: `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`. Build UTF-8 text by joining each
  split's original `text` rows with `\n`, without adding a final newline.

The validator loads assets offline, packs token blocks without padding and uses
alternating ignored-label regions to exercise nonuniform token normalization.
The retained reference model and optimizer increase memory beyond normal training.
Do not interpret its peak memory or execution duration as deployment performance.

## SmolLM2 Data Parallelism

From the repository root, set paths to your local assets:

```bash
MODEL=/path/to/smollm2-135m
DATA=/path/to/wikitext/train.txt
OUTPUT=$(mktemp -d /tmp/parascale-smollm-validation.XXXXXX)
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=4 \
  tests/benchmarks/scripts/validate_llama_parallel.py \
  --dp 4 --model-path "$MODEL" --data-file "$DATA" \
  --sequence-length 32 --global-batch 8 --microbatches 2 --steps 4 \
  --logits-atol 0.0002 --output "$OUTPUT/result.json"
```

For one or two GPUs, change visible devices, process count and `--dp` together.
SmolLM2 has tied embedding/head and 9 Q / 3 KV heads; PP2 and TP2 are rejected.
On the five-card 156 host, the tested four-card selection is `0,1,3,4`; that
host-specific choice is not a valid default for every four-card machine.

## TinyLlama Pipeline and Tensor Parallelism

The qualified two-stage case uses its own command and the default tight output
tolerance, not the SmolLM2 tolerance:

```bash
MODEL=/path/to/tinyllama-1.1b-59f6f375
DATA=/path/to/wikitext/train.txt
OUTPUT=$(mktemp -d /tmp/parascale-tinyllama-validation.XXXXXX)
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 \
  tests/benchmarks/scripts/validate_llama_parallel.py \
  --dp 1 --tp 1 --pp 2 --attention sdpa \
  --model-path "$MODEL" --data-file "$DATA" \
  --sequence-length 32 --global-batch 8 --microbatches 2 --steps 4 \
  --output "$OUTPUT/result.json"
```

For DP2+PP2, use four visible devices, `--nproc_per_node=4` and
`--dp 2 --tp 1 --pp 2`. To investigate TP2, use two devices and
`--dp 1 --tp 2 --pp 1 --observe-errors`; for TP2+PP2 use four devices and
`--dp 1 --tp 2 --pp 2 --observe-errors`. These are diagnostics, not equivalent
acceptance commands. Full DP2+TP2+PP2 needs eight ranks and is not admitted.

Current boundary: TinyLlama single-device and PP2 pass the update-level reference
checks. TP-containing and DP4 cases are still under numerical qualification;
the strict validator can reject their Adam parameter-update drift even when
forward/gradient checks pass. Use `--observe-errors` only for investigation, not
to turn such a run into a passing acceptance result. Native TP forward controls
are required before attributing a discrepancy to the adapter.

Omit both `--model-path` and `--data-file` to run a small randomly initialized
complete Llama fixture. That mode is explicitly synthetic, not pretrained quality.

## Numerical Gates

The reference uses the same microbatch shape to separate GEMM shape effects from
adapter behavior, while retaining one full-model global-token objective. It checks
loss, logits, gradients, absolute parameters, actual parameter updates, both Adam
moments and exact step counters. Peak and RMS bounds are separate, with small
floors for second moments. Negative-control tests reject zeroed moments, skipped
updates and incorrect DP normalization.

`--logits-atol` is explicit and recorded; default is `3e-6`. The SmolLM2 four-rank
example uses `2e-4` after matched native-forward controls showed the same rounding
drift. An independent relative logit bound remains enforced. Do not transfer this
tolerance to another model or precision without native controls and a quality gate.

`--native-forward-control` bypasses the adapter forward but still uses ParaScale's
executor: it isolates adapter error, not the entire distributed trainer. Its result
is always `diagnostic`, never adapter qualification. `--observe-errors` records
gate failures without declaring a pass. Keep generated results out of commits.

`--adam-eps` explicitly sets the same positive finite AdamW epsilon in both paths
and records it in the result; the default remains `1e-8`. Use this only for
controlled optimizer-sensitivity experiments. Changing epsilon changes training,
not the acceptance tolerance, and requires independent quality validation. A
smaller parameter drift alone does not qualify a topology or justify changing
production optimizer settings.

These correctness checks alone do not establish convergence, held-out quality,
throughput speedup, fresh-process recovery, tied cross-stage optimization or
24-hour production readiness.

## BF16 with FP32 Master Parameters

Use `--precision bf16` explicitly; the default remains FP32. CUDA BF16 autocast
applies to forward/loss operations, while parameters, gradients and Adam moments
stay FP32. This does not enable FP16, distributed QAT, or distributed distillation.

For a matched native TP reference, add `--reference-layout matched-tp`. The
reference is the complete Hugging Face model using an independent PyTorch TP
plan, not a ParaScale executor. PP ranks compare their stages with that full
model. The original unsharded reference remains available and no numerical
tolerances are relaxed by selecting BF16 or matched TP.

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 \
  tests/benchmarks/scripts/validate_llama_parallel.py \
  --dp 1 --tp 2 --pp 1 --attention sdpa --precision bf16 \
  --reference-layout matched-tp \
  --model-path "$MODEL" --data-file "$DATA" \
  --sequence-length 32 --global-batch 8 --microbatches 2 --steps 4 \
  --output "$OUTPUT/bf16-native-tp.json"
```

Here `MODEL` is the TinyLlama directory and `DATA` is `train.txt`, as in the
correctness example above. A matched-TP pass means agreement with the native TP
computation, not unsharded equivalence or pretrained quality retention.

The dedicated boundary validator exercises BF16 activation transport and rejects
shape/dtype mismatches on every rank before pipeline communication:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 \
  tests/benchmarks/scripts/validate_pipeline_precision.py \
  --dp 1 --output "$OUTPUT/bf16-boundaries.json"
```

For DP2+PP2, change visible devices and process count to four, and use `--dp 2`.
This is a small linear-stage transport/control test, not Llama quality evidence.

## Bounded Throughput Measurement

The separate benchmark measures training without a retained reference model. It
loads the same pinned weights, packs `train.txt` and `validation.txt` from `--data`,
and reports initial/final validation NLL alongside optimizer-inclusive timings.
It supports one-node FP32/BF16 TP1/TP2, not DP/PP or an end-to-end data pipeline benchmark.

```bash
MODEL=/path/to/tinyllama-1.1b-59f6f375
DATA=/path/to/wikitext
OUTPUT=$(mktemp -d /tmp/parascale-tp-throughput.XXXXXX)
COMMON=(--model "$MODEL" --data "$DATA" --tp 2
        --global-batch 16 --microbatch 8 --sequence-length 256
        --steps 32 --warmup 8 --validation-tokens 8192)
for DRIVER in native parascale; do
  CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  python -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 \
    tests/benchmarks/scripts/benchmark_llama_parallel.py \
    "${COMMON[@]}" --driver "$DRIVER" --output "$OUTPUT/$DRIVER.json"
done
```

Select GPUs using the actual `nvidia-smi topo -m` output, not their numeric order.
For example, the tested 156 host has a PIX-connected pair at physical GPU1/2;
GPU0/1 is NODE-connected. This is not a portable device-selection rule. The tool
does not select GPUs or pin CPU cores automatically; optional `--cpu-affinity`
accepts explicit allowed CPU indices.

The native baseline defaults to the same shared-input PyTorch TP optimization.
`--driver native --native-plan layerwise` is a separate ablation of independent
projection inputs, not the default baseline. For a single-device control, change
visible devices, process count and `--tp 1` together, keeping the global token
budget, optimizer, sequence length and microbatch shape matched when memory allows.

Run modes sequentially, alternate their order and repeat before claiming gains.
To compare BF16, add `--precision bf16` to `COMMON` for both drivers; do not compare
ParaScale BF16 with native FP32 and attribute the entire gain to the framework.
Compare asset/token hashes, training losses, validation block sums, device mapping
and optimizer settings as well as throughput. Validation uses a bounded prefix;
a finite or improved NLL is not a convergence or production acceptance gate.

`tokens_per_second` uses all measured steps divided by their total time;
`median_step_tokens_per_second` separately reports inverse median step latency.
`scored_tokens_per_second` excludes the first label of each causal block. TP ranks
do not multiply the token count. Startup, tokenization, H2D and validation are
outside the timed window; peak allocated memory is measured during training.
Outputs remain `diagnostic` and include per-step timings plus model, data, source
and script digests. Keep these generated files outside the repository.

## Pipeline Output Retention and Memory

Pipeline training defaults to `return_outputs=False`: it does not keep auxiliary
logit chunks or concatenate them after backward. `return_outputs=True` preserves
ordered output tensors on the last stage. Native GPipe still keeps the forward
graphs needed for backward and per-microbatch receive buffers; this change alone
does not bound activation memory or guarantee a throughput gain.

The numerical validator supports `--output-policy alternate` to switch between
retained and discarded outputs across updates while checking the same native
reference. `discard` checks gradients/updates without returning logits; `retain`
remains the default for output-level numerical comparisons.

For real model memory/timing observations, use this separate pipeline benchmark:

```bash
MODEL=/path/to/tinyllama-1.1b-59f6f375
DATA=/path/to/wikitext/train.txt
OUTPUT=$(mktemp -d /tmp/parascale-pipeline-memory.XXXXXX)
COMMON=(--model "$MODEL" --data-file "$DATA" --dp 1 --tp 1 --pp 2
        --precision bf16 --global-batch 16 --microbatches 2
        --sequence-length 256 --steps 16 --warmup 4)
for MODE in no-return-outputs return-outputs; do
  CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  python -m torch.distributed.run \
    --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 \
    tests/benchmarks/scripts/benchmark_llama_pipeline.py \
    "${COMMON[@]}" --"$MODE" --output "$OUTPUT/$MODE.json"
done
```

For DP2+PP2 or TP2+PP2, select four devices/processes and change `--dp 2` or
`--tp 2`, not both. To sweep retention at fixed microbatch shape, increase
`--microbatches` and `--global-batch` together. Respect memory limits; do not infer
four-card performance from a two-card measurement.

The result includes per-rank allocated/reserved peaks, post-backward/post-step
allocated bytes, output/backward cache counts and backing-storage bytes, losses,
and exact final model/optimizer fingerprints. Cache storage measurements exclude
autograd-saved tensors, and storage can overlap between different caches.
Fingerprints are computed only after timing/peak capture and do not write a
checkpoint. The script deletes caller-owned returned results before the optimizer
in both modes, so post-step retention is not an artificial caller-held output.

Timing sums separately synchronized forward/backward and optimizer segments,
excluding inspection, H2D, preprocessing and fingerprints. It is not end-to-end
latency and should not be compared directly with differently instrumented tools.
Compare source versions using identical benchmark/helper files, assets, topology,
precision, batch budget and affinity; alternate order and repeat. Require matching
losses and final per-rank state fingerprints before interpreting memory or speed.
The status remains `diagnostic`, not convergence or production acceptance.

## Pipeline Schedule Comparison

All validation, checkpoint and pipeline benchmark commands accept
`--pipeline-schedule gpipe|1f1b`, defaulting to GPipe. Use `--pp 2` and
`--microbatches >= 2` for 1F1B. The selected policy must be the same for saving
and resuming; schema4 checkpoints reject older schemas or schedule mismatches
before reading weight payloads. A GPipe checkpoint cannot be silently reused
for a 1F1B run.

For a fixed microbatch-size memory sweep, reuse the model/data paths above:

```bash
OUTPUT=$(mktemp -d /tmp/parascale-pipeline-schedule.XXXXXX)
for MICRO in 2 4 8; do
  for SCHEDULE in gpipe 1f1b; do
    CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
    python -m torch.distributed.run \
      --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 \
      tests/benchmarks/scripts/benchmark_llama_pipeline.py \
      --model "$MODEL" --data-file "$DATA" --pp 2 --precision bf16 \
      --pipeline-schedule "$SCHEDULE" --microbatches "$MICRO" \
      --global-batch "$((2 * MICRO))" --sequence-length 256 \
      --steps 16 --warmup 4 --output "$OUTPUT/$SCHEDULE-m$MICRO.json"
  done
done
```

Each comparison pair processes identical tokens at microbatch size 2. Repeat in
reverse order for timing; compare all ranks' allocated peaks, not only rank 0.
Different microbatch counts change the global batch and total token budget, so
do not treat the sweep as a convergence comparison. 1F1B still allocates native
per-microbatch receive buffers and is not a constant-total-memory algorithm.

## Fresh-process Checkpoint Validation

This separate validator compares a continuous four-update run with an independent
`torchrun` that resumes after update two. It uses real local weights/text, without
a retained full-model reference. The checkpoint context is constructed from the
effective model configuration, code/asset hashes, actual token stream, loss and
packing policy. Use the same source snapshot, devices, topology and paths for both
commands. Allow roughly 15 GB of free checkpoint space for untied TinyLlama TP2/PP2.

```bash
RUN=$(mktemp -d /tmp/parascale-llama-resume.XXXXXX)
COMMON=(--model /path/to/tinyllama-1.1b-59f6f375
        --data /path/to/wikitext/train.txt
        --dp 1 --tp 1 --pp 2 --steps 4 --save-step 2
        --global-batch 8 --sequence-length 32 --microbatches 2)
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 \
  tests/benchmarks/scripts/validate_llama_checkpoint.py "${COMMON[@]}" \
  --checkpoint-dir "$RUN/checkpoint" --output "$RUN/reference.json"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2 \
  tests/benchmarks/scripts/validate_llama_checkpoint.py "${COMMON[@]}" \
  --resume-from "$RUN/checkpoint" --reference-result "$RUN/reference.json" \
  --output "$RUN/resumed.json"
```

The first result is `reference`, not a recovery pass. The second reports `passed`
only when every rank has identical final parameter/Adam-state/RNG fingerprints and
an exact matching loss suffix. Nonfinite tensor state is rejected. To test TP2,
use `--tp 2 --pp 1`; for DP2+TP2 or TP2+PP2, use four visible devices/processes and
adjust the three topology options together.

For BF16 continuation, add `--precision bf16` to `COMMON` for both commands and
use a fresh checkpoint/result directory. The result explicitly records execution
precision, and the checkpoint context rejects a different precision on resume.

This verifies bounded same-topology continuation, not equivalence to unsharded
training, data-loader/scheduler restoration, convergence, crash injection or long-run
readiness. Input position is reconstructed from the checkpoint step in a fixed,
unshuffled packed token stream. Keep the result JSON and checkpoints outside Git.
