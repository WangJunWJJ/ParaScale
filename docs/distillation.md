# Online Distillation

Online distillation trains a student using ground-truth targets and a frozen,
evaluation-mode teacher. The teacher runs under `torch.no_grad()` on every
training batch. It is not registered as part of the student: student parameters,
optimizer, exported weights, and inference remain independent of the teacher.
Online teacher forward computation and resident weights add training cost.
Distillation does not guarantee improved accuracy or lower held-out perplexity.

## Objective

For aligned student and teacher logits, temperature `T`, and soft-loss weight
`alpha`, the objective is:

```text
hard_loss = CE(student_logits, labels)
soft_loss = T^2 * mean_valid KL(softmax(teacher_logits / T)
                              || softmax(student_logits / T))
total = (1 - alpha) * hard_loss + alpha * soft_loss
```

KL direction is teacher to student, not the reverse. The `T^2` multiplier
compensates for the temperature-dependent gradient scale. `alpha=0` selects
hard-label CE; `alpha=1` selects soft-target KL. Intermediate values mix them.
Loss computation uses FP32 logits and valid-token means, not a mean over all
padded positions or over vocabulary entries. For `task="causal_lm"`, logits at
position `t` predict labels at `t+1`; ignored labels (`ignore_index=-100`) and
masked source or target positions are excluded. Classification requires matching class
semantics and class order. Causal LM requires matching token identities and
token-to-ID mappings, not merely equal vocabulary size. The generic loss API
cannot infer semantic alignment from tensors; the caller must establish it.

For classification batches, `attention_mask` remains a teacher model input;
it does not filter the loss. Use the optional sample-level `loss_mask[B]` for
classification loss filtering. Causal LM instead uses `attention_mask[B, L]`
to exclude prediction pairs with a masked source or target position.

## Python API

Given independent loaded `student` and `teacher` models, a prepared `batch`, and
a student-only `optimizer`:

```python
from parascale.distillation import DistillationConfig, OnlineDistillationLoss

config = DistillationConfig(
    enabled=True,
    task="causal_lm",
    temperature=2.0,
    alpha=0.5,
    ignore_index=-100,
    input_keys=["input_ids", "attention_mask"],
)
objective = OnlineDistillationLoss(student, teacher, config)
optimizer.zero_grad(set_to_none=True)
output = student(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    use_cache=False,
)
loss = objective(output, batch)
loss.backward()
optimizer.step()
print(objective.last_metrics)  # hard_loss, soft_loss, total, valid_count
```

The adapter mutates the teacher to frozen/eval mode, clears teacher gradients,
and evaluates it without gradient recording. It does not wrap the student or
attach teacher weights to its state dictionary. Use separate model objects and
storage. `distillation_loss(student_logits, teacher_logits, labels=...,
attention_mask=..., config=config)` exposes `total`, `hard_loss`, `soft_loss`,
and `valid_count` for custom callers.

The existing runtime can consume this callable without a second training engine:

```python
engine.fit(
    dataloader=train_loader,
    model=student,
    optimizer=optimizer,
    loss_fn=objective,
    max_steps=10,
)
```

This snippet assumes a configured native FP32 `engine` and aligned batches.
The tensor loss API is not a runtime-policy gate: direct callers are responsible
for placement, precision, accumulation semantics, checkpoint metadata, and
unsupported combinations. Passing a callable alone does not automatically
establish the CLI's objective-manifest or resume validation contract.

## Native Train CLI

Use the normal `python -m parascale.cli train --config <config.json>` command.
Add a top-level `distillation` object to the existing full student workload
configuration. The following is a structural fragment, not a complete runnable
workload configuration:

```yaml
distillation:
  enabled: true
  task: causal_lm
  temperature: 2.0
  alpha: 0.5
  teacher:
    config: <nested full existing workload config for the teacher>
    checkpoint: /path/to/backend_state.pt
```

`teacher.config` is an actual nested configuration mapping, not a configuration
filename or just a model name. `teacher.checkpoint` is required and must contain
a full tensor raw `state_dict` or a native `backend_state.pt` checkpoint.
The teacher must load successfully with the configured architecture. There is
no silent fallback to randomly initialized teacher weights. The benchmark's
explicit `--tiny` random teacher is a mechanical test, not CLI training policy.

Student checkpoints preserve the objective metadata/manifest, including teacher
and configuration fingerprints. Resume rejects a different teacher or objective
configuration instead of silently changing the learning objective. Source
teacher weights are not included in the student's state dictionary. A standalone
student state export is an inference artifact, not an optimizer/resume checkpoint.

The initial native CLI integration supports one-process FP32 training only.
It rejects distributed execution, non-FP32 precision, adapter-only checkpoints,
quantization, and `gradient_accumulation_steps != 1`. Correct token-weighted
normalization across variable-length accumulated microbatches is not yet
validated. QAT plus KD is not an available combined CLI workflow until validated.
These are CLI/runtime guards, not guarantees enforced by the standalone loss
API. Feature distillation, reverse distillation, offline cached logits, top-k
targets, reinforcement learning, and detection distillation are outside scope.

## Llama Verification

Run the offline random-model smoke check with a torch/transformers environment:

```bash
python tests/benchmarks/scripts/validate_llama_distillation.py \
  --tiny --device cpu --seed 42 --seqlen 32 --batch-size 1 \
  --batches 4 --steps 4 --output /tmp/llama-kd-new-run
```

The teacher has hidden size 64 and two layers; the student has hidden size 32
and one layer; both use 64 token IDs. Initialization, training data, held-out
data, and training randomness use explicit separated seeds. Training and
evaluation token batches must be distinct. No model download is needed.

For pretrained Llama pairs, provide both model paths or Hugging Face model IDs
and separate calibration/training and evaluation text files:

```bash
python tests/benchmarks/scripts/validate_llama_distillation.py \
  --teacher-model /models/teacher --student-model /models/student \
  --local-files-only --calibration-text /data/train.txt \
  --evaluation-text /data/heldout.txt --device cpu \
  --seqlen 128 --batch-size 1 --batches 4 --steps 10 \
  --temperature 2 --alpha 0.5 --output /tmp/llama-kd-pretrained-new-run
```

`--training-text` aliases `--calibration-text`. Text must supply the requested
number of full batches; incomplete batches are not padded. File paths and file
content hashes must differ between splits; this does not prove the corpora have
no semantic overlap. The script additionally rejects identical token batches.
Exact vocabulary and special-token mappings are checked; equal vocabulary sizes
alone are insufficient. Both models receive the same student-tokenized inputs.
Remote runs can pin `--teacher-revision` and `--student-revision`; use immutable
revisions for reproducibility. The benchmark permits `cpu` or `cuda[:index]` but
does not launch distributed jobs or schedule shared GPUs.

Use a fresh, empty output directory. The script writes `report.json` and a
student-only `student_state_dict.pt`. The report includes:

- Arguments, package versions, hardware, source model configurations, initial
  student hash, parameter counts, corpus hashes, and exact token-batch hashes.
- Held-out student CE, perplexity, and FP32 valid-token-mean
  `KL(P_teacher || P_student)` at temperature 1, before and after training.
- A CE-only student baseline restored from exactly the same initialization,
  using the same training batches, step budget, optimizer, and training seed.
- Per-step synchronized timings, including forward, objective, backward,
  validation checks, and optimizer update, excluding batch transfer. There is
  no warmup; first-step initialization is included.
- CUDA allocator peaks and optional 10 ms sampled whole-process RSS during the
  training loop. CPU allocator peaks are not claimed. Teacher and active student
  remain resident even during the CE baseline; baseline memory therefore is not
  a teacher-free deployment footprint. Initial weights are retained on CPU.
- Teacher state hashes before/after, frozen/eval/no-gradient checks, finite
  losses and gradients, nonzero student gradients and changed parameters,
  and student-only reload logit parity on all held-out batches.

The benchmark deliberately uses a small standard torch loop for controlled
measurement; it is not another production training engine. Native CLI save and
resume tests cover runtime integration separately. Failed correctness checks
produce a nonzero exit status. Finite measurements, real updates, immutable
teacher state, and export parity are required; no arbitrary quality-improvement
threshold is imposed. Random tiny results are correctness evidence only, not
evidence that distillation improves a trained model.

## Recorded Validation

The A6000 FP32 smoke run uses 32 steps, batch size 2, sequence length 128,
eight training batches and eight independent evaluation batches. The teacher
has 82,240 parameters and the student has 13,408. Both distillation and CE-only
training produced nonzero gradients on every step; teacher hashes were unchanged
and student-only reload maximum logit error was zero.

The remote regression suite passed 561 tests with one skipped; the focused CPU
suite passed 80 tests. These are functional results for small random models,
not evidence of pretrained Llama quality, distributed scalability, or production
speed. Pretrained evaluation remains pending. See the
[unified benchmark report](../tests/benchmarks/reports/BENCHMARK_REPORT.md#llama-online-distillation)
for the CE comparison, measured timing scope and full evidence.
