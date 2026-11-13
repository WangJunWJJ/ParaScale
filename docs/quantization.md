<!--
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
-->

# Reference PTQ and QAT validation

ParaScale's quantization path targets **reference floating-point computation**.
Converted weights may use packed storage, but this is not an optimized integer
Llama execution engine. Smaller checkpoints do not imply lower latency or lower
runtime memory. Dequantization, metadata, embeddings and the excluded output head
affect the actual result. Measure checkpoint size and runtime separately.

Converted quantized layers expose integer storage as `packed_weight`.
Their `.weight` accessor returns floating-point dequantized weights, not the
integer storage. Inspect `packed_weight` for storage representation; using
`.weight` can materialize a floating tensor and does not establish memory savings.

## Supported validation boundary

- Weight-only 4-bit and 8-bit PTQ/QAT, per-channel weights, no module fusion,
  `quantize_activations=False`, and `exclude_modules=["lm_head"]` for Llama.
- `backend="reference"` is the only supported quantization backend; other
  values are rejected. `fuse_modules=False` is the default, and setting it to
  `True` is rejected rather than silently ignored.
- Single-device CPU/CUDA, float32 reference execution. QAT uses genuine optimizer
  updates through fake quantization before conversion and export.
- FSDP, DeepSpeed, distributed quantization integration and mixed-precision/AMP
  QAT are **unsupported until explicitly validated**. No distributed or AMP
  support or performance claim is made here.
- The offline random two-layer Llama test establishes API and serialization
  correctness only. It is not evidence of pretrained model quality, TinyLlama
  accuracy, GPU performance, or production stability.

### Validation status (2026-11-13)

The coordinating validation run reports 18 remote Llama/runtime tests passed,
plus GPU 4-bit and 8-bit PTQ/QAT smoke runs with 16 QAT optimizer steps passed.
These use a tiny randomly initialized, genuine Llama architecture, not a
pretrained quality benchmark. Latest template revalidation remains pending
the quantization core review fixes. The unified validation report maintained
by the coordinating task supplies the detailed evidence and links.

**Production pretrained-quality validation remains blocked:** Hugging Face,
mirror and ModelScope model downloads were inaccessible or stalled. No stable
pretrained quality, TinyLlama quality retention, or production readiness claim
can be inferred from these smoke results.

## Native workload CLI

PTQ builds the model and calibration dataloader through the existing workload's
`build_training_components` path. Supply an existing workload configuration, not
an arbitrary Hugging Face repository identifier:

```bash
python -m parascale.cli quantize --config EXISTING_WORKLOAD_CONFIG --output artifact.pt
```

Configure `parascale.quantization` with `enabled: true`, `mode: ptq`, `bits: 4`
or `8`, `backend: reference`, `fuse_modules: false`, `per_channel: true`,
`quantize_activations: false`, and workload-appropriate `exclude_modules`.
For Llama, exclude `lm_head`.

Native training QAT uses `parascale.quantization.enabled=true` and
`parascale.quantization.mode=qat`, float32, and `training_backend=native` in the
existing training configuration. Continue through the existing native training
entrypoint. Set `training.quantized_output` to choose the exported checkpoint;
the default is `checkpoint_dir/quantized.pt`. QAT CLI training rejects
`adapter_only_checkpoint`: full-model checkpoints are required. Do not enable
adapter-only checkpointing for this path. Unsupported backends must produce
explicit errors. The Llama validation below uses the Python API directly and
does not certify arbitrary workload CLI integrations.

## Python API

`model` is a float Llama model. `calibration_batches` is a list of dictionaries
containing `input_ids` and `attention_mask` on the prepared model's device.
Training and held-out evaluation data must be separate.

```python
from parascale.quantization import (
    QuantizationConfig, PostTrainingQuantization,
    QuantizationAwareTraining, load_quantized_model,
)

config = QuantizationConfig(
    enabled=True, mode="ptq", bits=4, backend="reference", fuse_modules=False,
    per_channel=True, quantize_activations=False, exclude_modules=["lm_head"],
)
ptq = PostTrainingQuantization(model, config)
prepared = ptq.prepare()
ptq.calibrate(calibration_batches)
converted = ptq.convert()
ptq.export("llama-ptq.pt")
loaded, restored_config, metadata = load_quantized_model(
    "llama-ptq.pt", model=float_template,
)
```

For QAT, construct the same configuration with `mode="qat"`, call
`QuantizationAwareTraining(model, config).prepare()`, and train the **returned
copy**, not the original float model. Build the optimizer after preparation.
Use causal next-token cross-entropy, backpropagate and step the optimizer;
verify finite nonzero gradients on quantized projection weights and that these
weights actually changed. Then call the controller's `freeze_observer()`,
`convert()` and `export(path)`. The benchmark script supplies the complete loop.
Reload with the same float architecture and check held-out output parity.

## Offline smoke validation

Install compatible PyTorch and Transformers in the active environment. `psutil`
is optional for sampled process memory. No model or tokenizer download is needed:

```bash
python tests/benchmarks/scripts/validate_llama_quantization.py --tiny --mode both --bits 4 --device cpu --seqlen 8 --batches 2 --qat-steps 2 --output tests/benchmarks/reports/llama-tiny-4
python tests/benchmarks/scripts/validate_llama_quantization.py --tiny --mode both --bits 8 --device cpu --output tests/benchmarks/reports/llama-tiny-8
python -m pytest tests/test_llama_quantization.py -q
```

Tests use `pytest.importorskip` for PyTorch/Transformers; the repository's
no-PyTorch collection hook can skip the entire module. Skipped tests are not
passing Llama validation evidence.

## Pretrained Llama validation

Use a local Llama checkpoint or an explicitly requested Hugging Face repository,
for example `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Supply UTF-8 local files for both
calibration/QAT training and held-out evaluation. There is **no synthetic data
fallback** in pretrained mode. The tokenizer comes from the selected model.

```bash
python tests/benchmarks/scripts/validate_llama_quantization.py --model /models/TinyLlama-1.1B-Chat-v1.0 --local-files-only --calibration-text data/calibration.txt --evaluation-text data/evaluation.txt --mode ptq --bits 4 --device cuda:0 --seed 42 --seqlen 128 --batch-size 1 --batches 16 --warmup 3 --repeats 10 --output tests/benchmarks/reports/tinyllama-ptq-4
python tests/benchmarks/scripts/validate_llama_quantization.py --model /models/TinyLlama-1.1B-Chat-v1.0 --local-files-only --calibration-text data/calibration.txt --evaluation-text data/evaluation.txt --mode qat --bits 8 --device cuda:0 --seed 42 --seqlen 128 --batch-size 1 --batches 16 --qat-steps 16 --learning-rate 0.0001 --output tests/benchmarks/reports/tinyllama-qat-8
```

For a repository identifier, omit `--local-files-only`; use `--revision` with an
immutable commit for reproducibility. Repository access requires network access
and any applicable model permissions. The script does not execute remote code.

File SHA-256, file sizes, paths and consumed token-batch hashes appear in the
report. Identical files and identical full batches across splits are rejected.
This detects obvious leakage, not semantic duplication or partial overlap:
users remain responsible for genuinely independent corpora. QAT trains on the
calibration split and cycles it when more steps than batches are requested.

Text is tokenized in fixed 8192-character chunks, concatenated into full fixed
length batches; incomplete trailing tokens are discarded. Chunk boundaries can
change tokenization compared with tokenizing a whole document. Loss is summed
next-token cross-entropy divided by the exact predicted token count, excluding
the first token of each sequence. Perplexity is `exp(loss)`. There is no padding
and no sliding-window context overlap. Each split must contain enough tokens
for the requested full batches or the run fails.

## Reading the report

Each output directory contains `report.json` and converted checkpoints. Use a
fresh directory for independent experiments. Results include baseline/PTQ/QAT
token-weighted loss and perplexity, logits mean/max absolute errors against the
baseline, state tensor bytes, actual serialized checkpoint bytes, synchronized
warmup/measured forward latency, and prefill input tokens per second.
Reports also identify the GPU name, CUDA runtime, requested revision and resolved
model/tokenizer commit when Transformers provides one. The baseline serialized
state and each quantized checkpoint have SHA-256 evidence. Local checkpoints may
have no repository commit; a null commit is not a claimed pinned revision.
The temporary baseline checkpoint is deleted after hashing and sizing.

Timing uses one fixed resident batch, excluding loading, tokenization and device
transfers. It is not autoregressive generation throughput.
`timing.inference_memory` measures only the synchronized timed forwards, resetting
CUDA peak statistics **after warmup**, with only the evaluated model resident.
It reports total peak CUDA allocated bytes (including model and resident batch),
starting allocation, and incremental peak above that starting allocation. These
are allocator measurements, not reserved memory or total device usage.

PTQ/QAT `procedure_memory` is separate: it includes float model loading,
preparation copies, calibration or optimizer training, conversion and export,
but excludes evaluation and timed inference. Baseline loading and serialization
have their own memory fields. Do not compare procedure peaks to inference peaks.
Optional process RSS is sampled every 10 ms and may miss short peaks or retain
allocator memory from earlier procedures; it is not isolated inference memory.
Without
`psutil`, RSS is `null`, not a fabricated CPU memory measurement. State bytes sum
state-dict tensors and can count aliases; they are not physical process memory.

Models are discarded between baseline/PTQ/QAT/reload phases and the float model
is reloaded for each phase. Evaluation logits are temporarily stored on disk
one batch at a time. Disk needs scale with evaluation tokens times vocabulary
size. Preparation/conversion may temporarily copy models, and full-model QAT
requires gradients and optimizer state: this is not a guarantee that a 1.1B
model fits on a particular device. Run `--mode ptq` and `--mode qat` separately
when appropriate, and reduce sequence/batch budgets for memory constraints.

Success requires finite losses, perplexities and logits, real finite nonzero
QAT gradients, changed quantized projection weights, matching restored config,
and export/load output parity (`rtol=1e-5`, `atol=1e-5`). Failures return a nonzero
exit code and write a failure report. There is no universal accuracy regression
threshold: compare held-out quality against your own acceptance criteria.
Passing a roundtrip does not prove model quality or acceleration.

## Verified snapshot (2026-11-13)

- Local no-PyTorch suite: 319 passed, 3 dependency skips. Local CPU quantization
  and Llama focused suite: 94 passed, 1 CUDA-only skip.
- A6000 full suite: 483 passed, 1 DeepSpeed dependency skip. PyTorch
  `2.5.0a0+872d972e41.nv24.08`, Transformers `4.51.3`, Python `3.10.12`.
- The dedicated `parascale-quantization:a6000-torch25-transformers451` image is
  derived from the existing CI image with DeepSpeed removed to avoid its
  Transformers circular import. The original CI image is unchanged.
- Both 4/8-bit GPU smoke runs used sequence length 128, batch size 1, eight
  separate calibration/evaluation batches, 16 QAT steps, five warmup forwards
  and 20 measured forwards. All 16 QAT steps had finite nonzero projection
  gradients; 14 projection weights changed. Reload maximum logit error was 0.
- These are random two-layer Llama tests, not pretrained TinyLlama results.
  Pretrained weight downloads failed in the available network environment;
  real-corpus accuracy acceptance remains pending.

See the [unified benchmark report](../tests/benchmarks/reports/BENCHMARK_REPORT.md#llama-reference-ptqqat)
for actual latency/storage results and complete measurement evidence. The
reference implementation was slower than the float baseline in these smoke
runs; no low-bit inference acceleration is claimed.
