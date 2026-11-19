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

## Cross-model applicability

The loss is architecture-independent within two explicit output contracts:
classification `[B, C]` and causal next-token prediction `[B, L, V]`. Teacher
and student may have different depths and hidden widths, but output positions
and class/token identities must align. A smaller student reduces inference
cost only if its architecture and deployment actually do less work; the loss
alone does not compress an existing model.

`tests/test_compression_portability.py` validates genuine student updates and
unchanged frozen teachers for small random Transformers BERT classifiers, ViT,
ResNet and Qwen2 models, with different teacher/student widths. It does not
measure held-out quality, pretrained accuracy retention, or CLI workload support.

- BERT/ViT/ResNet classification: align label meanings and order; provide the
  correct teacher `input_keys`. Tensor outputs, `logits` mappings and `.logits`
  model outputs are accepted. Raw tuple outputs require an adapter.
- Llama/Qwen-style causal LM: use the same token-to-ID mapping and aligned
  sequences, not merely equal vocabulary sizes. Different-tokenizer distillation
  needs explicit alignment and is not implemented.
- Encoder-decoder and masked LM losses cannot reuse the causal shift blindly.
  Token classification, segmentation, general detection, CLIP contrastive outputs and
  continuous-action policies do not match the current objective contract.
- Standard same-family YOLO detection has a separate feature adapter described
  below. Different class spaces and general cross-family feature matching are
  not supported.
- The existing single-process native FP32 CLI restrictions still apply. Python
  API model coverage does not lift distributed, AMP, accumulation or QAT+KD guards.

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
API. Reverse distillation, offline cached logits, top-k targets and reinforcement
learning are outside scope. Detection uses the separate contract below.

## YOLO Detection Distillation

Use `training.workload: yolo_detection` and `distillation.task: detection`.
This is a standard closed-set detector adapter, separate from `yolo_world`.
It reuses ParaScale's image/label pipeline and Ultralytics' official detection
loss. It does not replace that objective with classification KL or a proxy loss.

Supported: same-family standard Ultralytics `DetectionModel` pairs with exactly
three `Detect` input scales, identical class IDs/names and strides. Different
student/teacher widths are handled by trainable 1x1 projections. YOLOv8 and
YOLO11 have integration tests. Segmentation, pose, YOLO-World, end-to-end heads,
GroundingDINO and cross-family pairs are rejected. The validated upstream version
is Ultralytics 8.3.161; other versions are not implicitly certified.

The feature objective is intentionally simple, not an FGD implementation:

```text
normalize(F) = F / sqrt(mean_CHW(F^2) + 1e-6)  # separately per image
feature_loss = mean_scales MSE(normalize(project(student_neck)), normalize(teacher_neck))
total = official_detection_loss + feature_weight * batch_size * feature_loss
```

The batch-size factor follows Ultralytics' batch-summed detection objective.
Foreground and background have equal weight; classification-logit and box/DFL
distribution distillation are not implemented. The teacher sees exactly the
student's preprocessed images in eval/no-grad mode. Teacher weights and BatchNorm
statistics stay fixed. DFL's fixed integral convolution is not trained.
`feature_weight: 0` bypasses feature capture, projection and teacher inference.

Install the detection dependency with `pip install ultralytics==8.3.161`.
The following complete YAML example assumes trusted local pretrained model files,
a YOLO-format directory (`images/*.jpg`, `labels/*.txt`), and an existing teacher
tensor state dictionary. Empty label files represent background-only images;
images without a matching label file are skipped by the workload.

```yaml
backend: &backend
  training_backend: native
  precision: fp32
model:
  path: /models/yolov8n.pt
data: &data
  data_dir: /datasets/detection/train
  image_size: 320
  batch_size: 4
  num_samples: 96
  num_workers: 0
optimizer:
  type: adamw
  lr: 0.00001
  weight_decay: 0.0005
training:
  workload: yolo_detection
  max_steps: 40
  checkpoint_interval: 20
  checkpoint_dir: /outputs/yolo-kd
  seed: 42
distillation:
  enabled: true
  task: detection
  feature_weight: 0.1
  teacher:
    checkpoint: /models/teacher_state.pt
    config:
      backend: *backend
      model:
        path: /models/yolov8s.pt
      data: *data
      training:
        workload: yolo_detection
        max_steps: 1
```

Prepare the tensor-only teacher state from a **trusted** Ultralytics checkpoint:

```python
import torch
from ultralytics import YOLO

torch.save(YOLO("/models/yolov8s.pt").model.state_dict(), "/models/teacher_state.pt")
```

Teacher loading also accepts ParaScale YOLO proxy state dictionaries and native
`backend_state.pt` files. Strict shape/key checks and teacher/configuration hashes
protect resume. Class-name agreement does not prove dataset label semantics;
ensure labels use the same mapping before training.

```bash
python -m parascale.cli train --config detection-kd.yaml --output result.json
```

Full checkpoints include student, projections, optimizer and runtime state;
they exclude teacher tensors. The default `checkpoint_dir/student.pt` export
(overridable via `training.student_output`) contains **only raw student weights**,
without projections or teacher. It is for inference, not resuming training:

```python
student = YOLO("/models/yolov8n.pt")
student.model.load_state_dict(torch.load("/outputs/yolo-kd/student.pt", weights_only=True))
student.predict("test.jpg")
```

Native FP32, one process, accumulation=1 and full checkpoints are required.
AMP, distributed KD and QAT+KD remain guarded. Exact parameter comparisons on
CUDA require deterministic kernels (`CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(True)`) plus repeatable input order.

Validation uses `tests/test_detection_distillation.py` (ownership, gradients,
shape/class/family guards, hook cleanup, empty targets and zero weight) and
`tests/test_detection_training.py` (actual YOLOv8/YOLO11 CLI, resume, export and
changed-teacher rejection). The reusable real-data benchmark is
`tests/benchmarks/scripts/validate_detection_distillation.py`; it compares paired
initializations, hashes model/data inputs, records step and end-to-end throughput,
and optionally invokes official mAP evaluation with `--evaluation-data`.
See the [unified report](../tests/benchmarks/reports/BENCHMARK_REPORT.md).

This workload uses square resizing and the configured ParaScale optimizer, not
Ultralytics' complete augmentation/EMA/scheduler recipe. A short smoke run and
finite gradients are not evidence of quality improvement. Use an explicit
fine-tuning learning rate and compare against a same-budget supervised student
on a sufficiently large held-out set before deployment.

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
