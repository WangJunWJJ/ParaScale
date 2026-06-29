#!/usr/bin/env bash
set -u

ROOT_DIR=${ROOT_DIR:-/workspace}
OUTPUT_DIR=${OUTPUT_DIR:-runs/benchmarks/datacomp_clip_boundary}
CONFIG_DIR=${CONFIG_DIR:-runs/benchmarks/datacomp_clip_boundary_configs}
mkdir -p "${ROOT_DIR}/${OUTPUT_DIR}" "${ROOT_DIR}/${CONFIG_DIR}"
cd "${ROOT_DIR}" || exit 1

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

rm -f "${OUTPUT_DIR}"/*.json

run_and_capture() {
  local name="$1"
  shift
  local output="${OUTPUT_DIR}/${name}.json"
  local error_output="${OUTPUT_DIR}/${name}.error.json"
  echo "[ParaScale] running ${name}: $*"
  if "$@" --output "${output}"; then
    rm -f "${error_output}"
  else
    local code=$?
    python3 - "$name" "$code" "$error_output" "$*" <<'PY'
import json
import sys

name, code, path, command = sys.argv[1:5]
payload = {
    "backend": name,
    "status": "error",
    "returncode": int(code),
    "command": command,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
    return "${code}"
  fi
}

make_config() {
  local base="$1"
  local output="$2"
  local backend="$3"
  local batch_size="$4"
  local grad_accum="$5"
  local hook="$6"
  python3 - "$base" "$output" "$backend" "$batch_size" "$grad_accum" "$hook" "$CONFIG_DIR" <<'PY'
import json
import sys

base, output, backend, batch, grad_accum, hook, config_dir = sys.argv[1:8]
batch = int(batch)
grad_accum = int(grad_accum)
with open(base, "r", encoding="utf-8") as handle:
    config = json.load(handle)
config["parascale"]["training_backend"] = backend
config["parascale"]["batch_size"] = batch
config["parascale"]["gradient_accumulation_steps"] = grad_accum
config["data"]["batch_size"] = batch
config["data"]["num_samples"] = max(4096, batch * grad_accum * 2 * 96)
config["training"]["max_steps"] = 80
config["training"]["benchmark_steps"] = 80
config["training"]["warmup_steps"] = 10
config["training"]["checkpoint_interval"] = 999999
config["training"]["skip_final_checkpoint"] = True
config["training"]["checkpoint_dir"] = f"{config_dir}/{output}_ckpt"
config["parascale"]["checkpoint_save_path"] = f"{config_dir}/{output}_ckpt"
if backend == "native_ddp":
    config["parascale"]["data_parallel_size"] = 2
    config["parascale"]["ddp_gradient_as_bucket_view"] = True
    config["parascale"]["ddp_static_graph"] = grad_accum <= 1
    config["parascale"]["ddp_comm_hook"] = hook
elif backend == "fsdp":
    config["parascale"]["data_parallel_size"] = 2
    config["parascale"]["fsdp_sharding_strategy"] = "full_shard"
    config["parascale"]["fsdp_state_dict_type"] = "full"
elif backend == "deepspeed":
    config["parascale"]["data_parallel_size"] = 2
    config["parascale"]["zero_optimization"] = True
    config["parascale"]["zero_stage"] = 2
out_path = f"{config_dir}/{output}.json"
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
print(out_path)
PY
}

run_torchrun_config() {
  local name="$1"
  local config="$2"
  run_and_capture "${name}" \
    torchrun --standalone --nproc_per_node=2 -m parascale.cli benchmark \
    --config "${config}"
}

run_deepspeed_config() {
  local name="$1"
  local config="$2"
  run_and_capture "${name}" \
    deepspeed --num_gpus=2 --module parascale.cli benchmark \
    --config "${config}"
}

for batch in 2 4 8; do
  native_ddp_config=$(make_config tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json "native_ddp_b${batch}_ga1_none" native_ddp "${batch}" 1 none)
  fsdp_config=$(make_config tests/benchmarks/configs/benchmark_datacomp_medium_fsdp.json "fsdp_b${batch}" fsdp "${batch}" 1 none)
  deepspeed_config=$(make_config tests/benchmarks/configs/benchmark_datacomp_medium_deepspeed.json "deepspeed_b${batch}" deepspeed "${batch}" 1 none)
  run_torchrun_config "native_ddp_b${batch}_ga1_none" "${native_ddp_config}"
  run_torchrun_config "fsdp_b${batch}" "${fsdp_config}"
  run_deepspeed_config "deepspeed_b${batch}" "${deepspeed_config}"
done

for grad_accum in 2 4; do
  config=$(make_config tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json "native_ddp_b2_ga${grad_accum}_none" native_ddp 2 "${grad_accum}" none)
  run_torchrun_config "native_ddp_b2_ga${grad_accum}_none" "${config}"
done

for hook in fp16_compress bf16_compress; do
  config=$(make_config tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json "native_ddp_b2_ga1_${hook}" native_ddp 2 1 "${hook}")
  run_torchrun_config "native_ddp_b2_ga1_${hook}" "${config}"
done

for batch in 4 8; do
  config=$(make_config tests/benchmarks/configs/benchmark_datacomp_medium_native_ddp.json "native_ddp_b${batch}_ga1_bf16_compress" native_ddp "${batch}" 1 bf16_compress)
  run_torchrun_config "native_ddp_b${batch}_ga1_bf16_compress" "${config}"
done

python3 - "${OUTPUT_DIR}" "tests/reports/archive/datacomp_clip_boundary_sweep_report.md" <<'PY'
import json
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
markdown_path = Path(sys.argv[2])
rows = []
for path in sorted(output_dir.glob("*.json")):
    if path.name.endswith(".error.json") or path.name in {"summary.json"}:
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    name = path.stem
    match = re.match(r"(?P<backend>native_ddp|fsdp|deepspeed)_b(?P<batch>\d+)(?:_ga(?P<ga>\d+)_(?P<hook>.+))?", name)
    if not match:
        continue
    rows.append(
        {
            "name": name,
            "backend": match.group("backend"),
            "batch_size": int(match.group("batch")),
            "gradient_accumulation_steps": int(match.group("ga") or 1),
            "ddp_comm_hook": match.group("hook") or "none",
            "stable_end_to_end_pairs_per_second": float(metrics.get("stable_end_to_end_image_text_pairs_per_second", 0.0) or 0.0),
            "stable_compute_pairs_per_second": float(metrics.get("stable_image_text_pairs_per_second", 0.0) or 0.0),
            "stable_dataloader_wait_ms": float(metrics.get("stable_dataloader_wait_ms", 0.0) or 0.0),
            "peak_memory_bytes": float(metrics.get("peak_memory_bytes", 0.0) or 0.0),
        }
    )

def by_name(name):
    for row in rows:
        if row["name"] == name:
            return row
    return None

comparisons = []
for batch in [2, 4, 8]:
    target = by_name(f"native_ddp_b{batch}_ga1_none")
    for baseline_name in [f"fsdp_b{batch}", f"deepspeed_b{batch}"]:
        baseline = by_name(baseline_name)
        if not target or not baseline:
            continue
        base_value = baseline["stable_end_to_end_pairs_per_second"]
        comparisons.append(
            {
                "target": target["name"],
                "baseline": baseline["name"],
                "speedup": target["stable_end_to_end_pairs_per_second"] / base_value if base_value > 0 else 0.0,
                "passed": target["stable_end_to_end_pairs_per_second"] > base_value,
            }
        )

summary = {
    "benchmark_id": "datacomp_clip_native_ddp_boundary_sweep",
    "primary_metric": "stable_end_to_end_image_text_pairs_per_second",
    "rows": rows,
    "comparisons": comparisons,
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    "# DataComp CLIP Native-DDP Boundary Sweep",
    "",
    "## Results",
    "",
    "| Run | Backend | Batch/rank | Grad accum | Hook | End-to-end pairs/s | Compute pairs/s | Wait ms | Peak memory GB |",
    "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        "| {name} | {backend} | {batch_size} | {gradient_accumulation_steps} | {ddp_comm_hook} | {e2e:.3f} | {compute:.3f} | {wait:.3f} | {mem:.3f} |".format(
            name=row["name"],
            backend=row["backend"],
            batch_size=row["batch_size"],
            gradient_accumulation_steps=row["gradient_accumulation_steps"],
            ddp_comm_hook=row["ddp_comm_hook"],
            e2e=row["stable_end_to_end_pairs_per_second"],
            compute=row["stable_compute_pairs_per_second"],
            wait=row["stable_dataloader_wait_ms"],
            mem=row["peak_memory_bytes"] / (1024 ** 3),
        )
    )
lines.extend(["", "## Backend Comparisons", ""])
for item in comparisons:
    lines.append(
        "- {target} vs {baseline}: speedup={speedup:.4f}, passed={passed}".format(**item)
    )
lines.append("")
markdown_path.parent.mkdir(parents=True, exist_ok=True)
markdown_path.write_text("\n".join(lines), encoding="utf-8")
PY
