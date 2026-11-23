# Saved PTQ/QAT Artifact Inference

This experimental CUDA example loads a real canonical ParaScale INT8 artifact,
then compares its fused inference with the same stored weights dequantized once
to BF16. The comparison does not use repeated reference dequantization as a slow
baseline and does not requantize a QAT model from its original floating weights.
Passing layer tests is not production or model-quality qualification.

## Prerequisites

- Linux CUDA SM80+ and the pinned [INT8 environment](../../../docs/quantized_inference.md).
- Local Hugging Face Llama architecture configuration and tokenizer matching the
  exported artifact. No remote model code or implicit download is used.
- A per-channel, weight-only INT8 artifact from [PTQ](../example_005_llama_ptq/README.md)
  or [QAT](../example_006_llama_qat/README.md). INT4 and activation quantization are
  not supported by this execution path.
- Independent evaluation text, not the calibration or QAT training corpus.

## Run the Comparison

Use the fixed SmolLM2/WikiText assets described in the PTQ/QAT examples, or replace
all paths with a matching local model and dataset. Set `ARTIFACT` to the actual
exported `ptq-8bit.pt` or `qat-8bit.pt`, and use a fresh output filename:

```bash
ASSET_DIR=/data/production-assets
ARTIFACT=/data/quantized-model/qat-8bit.pt
CUDA_VISIBLE_DEVICES=0 OUTPUT_FILE=/data/results/qat8-validation.json \
bash examples/gpu/example_011_quantized_artifact_inference/run.sh \
  --model "$ASSET_DIR/smollm2-135m" --reference-artifact "$ARTIFACT" \
  --data "$ASSET_DIR/wikitext/validation.txt" \
  --data-description WikiText2-validation \
  --sequence-length 128 --batch-size 1 --new-tokens 128 \
  --eval-tokens 8192 --warmup 3 --iterations 5 --repeats 4
```

The launcher resolves the repository root and supports `PYTHON` and `--help`.
Both variants run in separate processes, with identical static-cache/compile
settings, prompts and token budgets. Hashes identify model assets, source files,
evaluation text and the canonical quantized artifact. Keep outputs outside Git.

## Interpret the Result

- `summary.engineering_gate_passed` requires the configured throughput and
  quality gates, actual packed-kernel execution and compiled-region evidence.
- Artifact comparisons require at least 1.2x generation speedup and no more than
  1% perplexity increase against the matched BF16 artifact control. A valid run
  can still fail these gates; inspect the summary, not only process exit status.
- `runs.int8.conversion` must report preserved codes/scales and `requantized: false`.
- This comparison isolates execution changes, not quality loss from PTQ/QAT.
  Also inspect the PTQ/QAT validator's comparison against the original float
  model and, for QAT attribution, a matched unquantized fine-tuning control.
- Use validation for configuration selection. Freeze the settings before a
  separate test-split evaluation. Small-model, short-sequence or high-batch
  results do not imply acceleration on different workloads.

## Application API

```python
from parascale.quantization import load_quantized_model, prepare_quantized_inference_model

# template is the matching unquantized architecture, not a quantized runtime.
reference, config, metadata = load_quantized_model(artifact_path, template)
reference = reference.eval().cuda().bfloat16()
runtime, report = prepare_quantized_inference_model(reference)
```

Execute under `torch.inference_mode()`. Compile only after preparation. Do not
cast the fused runtime to BF16 again: scales must remain FP32. Persist the
canonical artifact and reconstruct the runtime; a standalone GemLite checkpoint
is not a portable deployment format. Backend configuration caches are checked
at preparation; do not mutate them after preparing or compiling the model.
