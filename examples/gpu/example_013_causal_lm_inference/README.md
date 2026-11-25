# Local Llama Generation Through the Product CLI

Run local Llama-family weights or canonical ParaScale PTQ/QAT artifacts through
`parascale infer`. This is single-device offline generation, not an HTTP service
or a continuous-batching engine. The included prompts are a functional example,
not a model-quality evaluation dataset.

## Floating-Point Generation

Run from the repository root with a matching local model and fast tokenizer:

```bash
export PARASCALE_MODEL=/data/models/tinyllama-1.1b
export PARASCALE_PROMPTS="$PWD/examples/gpu/example_013_causal_lm_inference/prompts.jsonl"
export PARASCALE_RUN_DIR=/data/runs/tinyllama-infer-float
CUDA_VISIBLE_DEVICES=1 python -m parascale.cli infer \
  --config examples/gpu/example_013_causal_lm_inference/float.json \
  --output "$PARASCALE_RUN_DIR/summary.json"
```

Use a dedicated fresh run directory outside model, prompt and source-config
locations. No implicit downloads or remote model code are used. The JSONL input
contains one nonempty `prompt` and optional string `id` per line. Oversized or
malformed inputs fail explicitly; prompts are never silently truncated.

Generation is greedy and stops at EOS or `max_new_tokens`. Warmup replays the
first request and does not consume it. Every request is emitted once to
`generations.jsonl`; `max_outputs: 0` avoids retaining responses in the summary.
Failed/interrupted output keeps a `.partial` suffix and must not be treated as
complete. Existing output files are rejected, not overwritten.

## Saved PTQ/QAT Artifacts

Set `PARASCALE_ARTIFACT` to an artifact exported by the current product's
`training.workload: causal_lm` PTQ or QAT commands, then select `reference.json`
or `fused.json` and a fresh run directory. See the
[causal-LM training example](../example_012_causal_lm_training/README.md).

To generate a new product artifact, prepare an immutable **training** token corpus
as in that example, then use the supplied compression config:

```bash
export PARASCALE_TOKEN_CORPUS=/data/tokens/wiki-train
export PARASCALE_RUN_DIR=/data/runs/tinyllama-ptq
export PARASCALE_ARTIFACT=/data/artifacts/tinyllama-product-ptq.pt
export PARASCALE_COMPRESSION_MODE=ptq
python -m parascale.cli quantize \
  --config examples/gpu/example_013_causal_lm_inference/compression.json \
  --output "$PARASCALE_ARTIFACT"
```

PTQ currently calibrates the CPU FP32 workload. This weight-only example uses one
calibration batch; it is not a quality recommendation for activation quantization.
For QAT, use fresh artifact/run paths, set `PARASCALE_COMPRESSION_MODE=qat`, then run:

```bash
CUDA_VISIBLE_DEVICES=1 python -m parascale.cli train \
  --config examples/gpu/example_013_causal_lm_inference/compression.json \
  --output "$PARASCALE_RUN_DIR/train-summary.json"
```

QAT uses 16 native FP32 updates at batch1/length128 and LR1e-5. It is a bounded
recipe, not proof that QAT improves model quality. Allow space for the full FP32
training checkpoint and optimizer state as well as the INT8 artifact. Evaluate
against independent validation/test data and a matched float fine-tuning control
before attributing gains to QAT. Use another fresh run directory for inference.

The stable `CausalLmModel` signature binds the model structure, effective config
and tokenizer. Old nested-wrapper artifacts and bare-HF benchmark artifacts are
not compatible: regenerate them with the current product commands. Artifact
inference needs matching configuration/tokenizer files but not original float
weight files. Loading is strict; there is no automatic signature conversion or
floating-point fallback.

`reference` is a correctness path with per-forward dequantization. It is **not
an appropriate acceleration baseline**. `gemlite_int8` consumes the exact saved
per-channel INT8 weights/scales without requantizing from float; it currently
requires weight-only quantization and CUDA BF16.

## Optimized Policy and Measurement Limits

`fused.json` opts into math SDPA, static cache and max-autotune compilation. This
bounded path requires batch size 1 and Transformers 4.51.3. Use the existing
[pinned INT8 environment](../../../docs/quantized_inference.md): PyTorch 2.9.1,
GemLite 0.5.1 and Triton 3.5.1. The float/reference examples intentionally use
dynamic eager execution for functional validation, so comparing these example
summaries directly is **not a matched performance experiment**.

`float_compiled.json` uses the same optimized policy with original floating-point
weights. It can isolate cache/compile changes against `float.json`; it is not
the dequantized-once artifact control required to isolate fused-kernel benefits.

For qualification, compare fused execution against the same artifact
dequantized once to BF16 with identical attention/cache/compile policy, prompts,
lengths and device. Compare compression quality against the original or matched
trained float model separately. Historical benchmark gains are not proof that
this new product path meets the same gates; product CUDA qualification remains
required before deployment.

The summary separates setup, warmup and synchronized prediction/postprocessing
time from inclusive job wall time. Token throughput counts actual generated
continuations, including the first EOS but excluding padding. It is neither
time-to-first-token nor a serving latency SLA. Compilation and warmup costs must
be included when assessing short-lived jobs; sustained decode gains alone do
not establish end-to-end acceleration.

For batched generation, `latency_ms_avg` is request-weighted batch completion
time within that prediction scope: all requests return together after the
batch finishes. `batch_latency_ms_avg` averages over batches, while
`amortized_request_time_ms` divides total batch time by request count and is a
cost/throughput measure, not individual response latency.
