# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Conservative output-space reservation without loading model tensors."""

import json
import math
import re
from pathlib import Path

RESERVE_BYTES = 1024 ** 3
SHARDED_WEIGHT = re.compile(
    r"model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight"
)


def parameter_bound(model_path, tp_size=1):
    """Count stored elements across TP ranks, including replicated tensors."""
    if type(tp_size) is not int or tp_size not in (1, 2):
        raise ValueError("storage estimation supports tp_size one or two")
    files = sorted(Path(model_path).glob("*.safetensors"))
    if not files:
        raise ValueError("storage estimation requires local floating safetensors weights")
    elements = 0
    for path in files:
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 2 <= size <= 64 * 1024 * 1024:
                raise ValueError("invalid safetensors header length")
            header = json.loads(stream.read(size))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if entry.get("dtype") not in ("F16", "BF16", "F32", "F64"):
                raise ValueError("storage estimation requires floating model tensors")
            shape = entry["shape"]
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("invalid safetensors shape")
            # LlamaStage shards only these projection weights. Embeddings,
            # heads, norms and conservatively all biases count on every TP rank.
            replicas = 1 if SHARDED_WEIGHT.fullmatch(name) else tp_size
            elements += math.prod(shape) * replicas
    if elements <= 0:
        raise ValueError("empty model tensor inventory")
    return elements


def storage_plan(case, env):
    """Budget two retained rank checkpoints plus a temporary full payload."""
    payload = 0
    retained = 1024 ** 2
    if case.kind == "parallel":
        payload = parameter_bound(env["assets"]["llama_model"], case.layout[1]) * 12 * case.layout[0]
        retained += 2 * payload
    elif case.kind == "diffusion":
        retained += env["budgets"]["requests"] * env["budgets"]["diffusion_size"] ** 2 * 8
    elif case.kind == "transformer":
        retained += env["budgets"]["requests"] * env["budgets"]["max_new_tokens"] * 128
    return {"retained_bytes": retained, "temporary_bytes": payload,
            "checkpoint_payload_bytes": payload, "reserve_bytes": RESERVE_BYTES}


def required_free(plan, command_index):
    """After the initial train command, one checkpoint is already on disk."""
    return plan["reserve_bytes"] + plan["retained_bytes"] + plan["temporary_bytes"] - command_index * plan["checkpoint_payload_bytes"]
