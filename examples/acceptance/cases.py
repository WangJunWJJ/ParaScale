# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded product commands and explicit acceptance coverage."""

import sys
from dataclasses import dataclass
from pathlib import Path

GROUPS = (
    "installation", "parallel_training", "llama7b_finetuning", "quantization",
    "distillation", "pruning", "generative_inference", "stability",
)
ASSETS = ("llama_model", "token_corpus", "llama_prompts", "diffusion_model", "diffusion_prompts")
DEFAULT_BUDGETS = {
    "train_steps": 2, "microbatch": 1, "global_batch": 8, "sequence_length": 32,
    "precision": "bf16", "max_new_tokens": 16, "requests": 2,
    "diffusion_steps": 2, "diffusion_size": 256,
}


@dataclass(frozen=True)
class Case:
    """A fixed capability probe, never an arbitrary shell command."""

    id: str
    kind: str
    requirements: tuple[str, ...]
    assets: tuple[str, ...] = ()
    layout: tuple[int, int, int] = (1, 1, 1)
    blocked: str = ""

    @property
    def group(self):
        return self.id.split(".")[0]

    @property
    def gpus(self):
        return 0 if self.kind == "installation" else self.layout[0] * self.layout[1] * self.layout[2]


CASES = {"installation.entries": Case("installation.entries", "installation", ("REL-01",))}
for name, layout in {
    "single": (1, 1, 1), "dp2": (2, 1, 1), "dp4": (4, 1, 1),
    "tp2": (1, 2, 1), "pp2": (1, 1, 2), "dp2_tp2": (2, 2, 1),
    "dp2_pp2": (2, 1, 2), "tp2_pp2": (1, 2, 2), "dp2_tp2_pp2": (2, 2, 2),
}.items():
    key = f"parallel_training.{name}"
    requirements = ["REL-05"]
    if layout[0] > 1:
        requirements.append("PAR-02")
    if layout[1] > 1 or layout[2] > 1:
        requirements.append("PAR-03")
    if sum(size > 1 for size in layout) > 1:
        requirements.append("PAR-04")
    CASES[key] = Case(key, "parallel", tuple(requirements),
                      ("llama_model", "token_corpus"), layout)
for name, assets, requirement in (
    ("transformer", ("llama_model", "llama_prompts"), "INF-06"),
    ("diffusion", ("diffusion_model", "diffusion_prompts"), "INF-07"),
):
    key = f"generative_inference.{name}"
    CASES[key] = Case(key, name, (requirement,), assets)
for group, requirements, reason in (
    ("llama7b_finetuning", ("TRN-01", "TRN-02", "TRN-03"), "Authorized Llama-2-7B, matched AdamW state-memory/quality protocol and runner required"),
    ("quantization", ("INF-02", "INF-03"), "Calibration, export/reload, integer-kernel and held-out quality gates are not integrated yet"),
    ("distillation", ("INF-04",), "Matched student CE control and independent quality comparison are not integrated yet"),
    ("pruning", ("INF-05",), "Original/pruned/recovery quality and speed comparison are not integrated yet"),
    ("stability", ("OPS-01", "REL-05"), "Final-candidate 24-hour run and fault-injection protocol are not integrated yet"),
):
    key = f"{group}.quality"
    CASES[key] = Case(key, "pending", requirements, blocked=reason)


def validate_environment(raw: dict) -> dict:
    """Reject ambiguous settings before launching any process."""
    allowed = {"schema_version", "gpu_ids", "assets", "budgets", "timeout_seconds"}
    if not isinstance(raw, dict) or set(raw) - allowed or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("environment requires schema_version=1 and known keys only")
    ids = raw.get("gpu_ids", [])
    if not isinstance(ids, list) or len(ids) > 4 or any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("gpu_ids must contain at most four distinct nonnegative integers")
    assets = raw.get("assets", {})
    budgets = raw.get("budgets", {})
    if not isinstance(assets, dict) or set(assets) - set(ASSETS) or any(not isinstance(v, str) for v in assets.values()):
        raise ValueError("assets must map known names to local path strings")
    if not isinstance(budgets, dict) or set(budgets) - set(DEFAULT_BUDGETS):
        raise ValueError("unknown budget setting")
    budgets = {**DEFAULT_BUDGETS, **budgets}
    limits = {"train_steps": 100, "microbatch": 64, "global_batch": 1024,
              "sequence_length": 4096, "max_new_tokens": 2048, "requests": 64,
              "diffusion_steps": 100, "diffusion_size": 1024}
    for name, maximum in limits.items():
        value = budgets[name]
        minimum = 2 if name == "sequence_length" else 1
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    if budgets["precision"] not in ("fp32", "bf16") or budgets["diffusion_size"] % 8:
        raise ValueError("precision must be fp32/bf16 and diffusion_size divisible by 8")
    timeout = raw.get("timeout_seconds", 900)
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError("timeout_seconds must be in 1..86400")
    return {"schema_version": 1, "gpu_ids": ids, "assets": assets,
            "budgets": budgets, "timeout_seconds": timeout}


def build_plan(case: Case, env: dict, directory: Path, train: str, infer: str) -> dict:
    """Generate product configs and argv lists without executing anything."""
    budgets, assets = env["budgets"], env["assets"]
    plan = {"commands": [], "configs": {}, "acceptance_gaps": []}
    if case.kind == "installation":
        plan["commands"] = [[tool, flag] for tool in (train, infer) for flag in ("--version", "--help")]
        plan["acceptance_gaps"] = ["Fresh isolated dependency installation and offline/image reproduction not checked"]
    elif case.kind == "parallel":
        dp, tp, pp = case.layout
        divisor = dp * budgets["microbatch"]
        if budgets["global_batch"] % divisor:
            raise ValueError("global_batch must be divisible by dp * microbatch")
        steps = budgets["train_steps"]
        config = {
            "parascale": {"training_backend": "native_parallel", "precision": budgets["precision"],
                          "data_parallel_size": dp, "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                          "pipeline_schedule": "gpipe", "checkpoint_save_path": str(directory / "checkpoints"),
                          "checkpoint_save_interval": steps, "checkpoint_keep_last": 2, "metrics_history_limit": steps},
            "model": {"pretrained_model_name_or_path": assets.get("llama_model", ""), "attention_backend": "sdpa"},
            "data": {"type": "token_corpus", "path": assets.get("token_corpus", ""),
                     "sequence_length": budgets["sequence_length"], "seed": 42, "shuffle": False,
                     "repeat": False, "num_workers": 0},
            "optimizer": {"type": "adamw", "lr": 1e-5, "weight_decay": 0.01},
            "training": {"workload": "causal_lm", "max_steps": steps, "seed": 42,
                         "batch_size": budgets["microbatch"], "gradient_accumulation_steps": budgets["global_batch"] // divisor},
            "runtime": {"run_dir": str(directory / "runtime"), "distributed_timeout_seconds": 120},
        }
        plan["configs"]["train.json"] = config
        prefix = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
                  "--master-addr=127.0.0.1", "--master-port=29507", f"--nproc-per-node={case.gpus}",
                  "--no-python", train, "run", "--config", str(directory / "train.json")]
        plan["commands"] = [prefix + ["--output", str(directory / "train-result.json")],
                            prefix + ["--resume-latest", "--output", str(directory / "resume-result.json")]]
        plan["acceptance_gaps"] = ["Matched native baseline, three paired repeats and held-out quality not evaluated",
                                   "Resume progress is checked; exact continuation equivalence is not checked"]
    elif case.kind in ("transformer", "diffusion"):
        transformer = case.kind == "transformer"
        config = {
            "schema_version": 1,
            "model": {"pretrained_model_name_or_path": assets.get("llama_model" if transformer else "diffusion_model", "")},
            "data": {"type": "prompts_jsonl", "path": assets.get("llama_prompts" if transformer else "diffusion_prompts", "")},
            "runtime": {"device": "cuda:0", "run_dir": str(directory / "runtime")},
            "inference": {"workload": "causal_lm" if transformer else "diffusion", "precision": "fp16",
                          "batch_size": 1, "warmup_steps": 1, "max_requests": budgets["requests"], "compile_mode": "off"},
        }
        if transformer:
            config["inference"].update(execution="float", max_input_tokens=128,
                                       max_new_tokens=budgets["max_new_tokens"], max_outputs=0,
                                       output_jsonl=str(directory / "generations.jsonl"),
                                       attention_policy="math", cache_implementation="dynamic")
        else:
            config["inference"].update(output_dir=str(directory / "images"), height=budgets["diffusion_size"],
                                       width=budgets["diffusion_size"], num_inference_steps=budgets["diffusion_steps"],
                                       guidance_scale=7.5, seed=42)
        plan["configs"]["inference.json"] = config
        plan["commands"] = [[infer, "run", "--config", str(directory / "inference.json"),
                             "--output", str(directory / "inference-result.json")]]
        plan["acceptance_gaps"] = ["Matched native baseline, three paired repeats and independent quality gates not evaluated"]
    return plan
