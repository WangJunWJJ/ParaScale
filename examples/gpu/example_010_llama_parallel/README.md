# Complete Llama Parallel Validation

This example trains a complete Llama model through integer tokens, embedding,
decoder, norm, output head and next-token loss. It compares each distributed
update with an independent unsharded Hugging Face reference. It is a correctness
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

This example does not establish convergence, held-out quality, throughput speedup,
fresh-process recovery, tied cross-stage optimization or 24-hour production readiness.
