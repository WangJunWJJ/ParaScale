# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午8:48
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded teacher-forced likelihood checks through autoregressive generation."""

import hashlib
import json
import math
from typing import Callable

import torch


def _metric(nll_sum: float, tokens: int) -> dict:
    if tokens <= 0 or not math.isfinite(nll_sum):
        raise ValueError("Invalid generation quality statistics")
    perplexity = math.exp(nll_sum / tokens)
    if not math.isfinite(perplexity):
        raise ValueError("nonfinite generation perplexity")
    return {"nll_sum": nll_sum, "tokens": tokens, "perplexity": perplexity}


class TeacherForcedScores:
    """Score original FP32 logits, then force targets without retaining logits.

    The caller must disable other logits transformations. Input targets are
    equal-length, unpadded int64 sequences. Position, shape and numerical errors
    raise ValueError. Call finish after generation to validate complete output.
    """

    def __init__(
        self,
        targets: torch.Tensor,
        *,
        prompt_length: int,
        on_decode_start: Callable | None = None,
    ):
        if (
            targets.dtype != torch.long
            or targets.ndim != 2
            or targets.shape[0] < 1
            or targets.shape[1] < 2
            or type(prompt_length) is not int
            or prompt_length < 1
            or targets.min().item() < 0
        ):
            raise ValueError("Invalid teacher-forcing targets or prompt length")
        self.targets = targets.detach().clone()
        self.max_target = int(targets.max().item())
        self.prompt_length = prompt_length
        self.step = 0
        self.nll = torch.empty_like(targets, dtype=torch.float32)
        self.nonfinite = torch.zeros((), dtype=torch.bool, device=targets.device)
        self.on_decode_start = on_decode_start

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.step >= self.targets.shape[1] or input_ids.shape != (
            self.targets.shape[0],
            self.prompt_length + self.step,
        ):
            raise ValueError("Invalid teacher-forcing position")
        if (
            scores.ndim != 2
            or scores.shape[0] != self.targets.shape[0]
            or scores.shape[1] <= self.max_target
            or not scores.is_floating_point()
            or scores.device != self.targets.device
        ):
            raise ValueError("Invalid teacher-forcing scores")
        values = scores.detach().float()
        self.nonfinite.logical_or_(~torch.isfinite(values).all())
        target = self.targets[:, self.step : self.step + 1]
        self.nll[:, self.step] = values.logsumexp(-1) - values.gather(
            1, target
        ).squeeze(1)
        self.step += 1
        forced = torch.full_like(scores, -float("inf")).scatter_(1, target, 0.0)
        if self.step == 1 and self.on_decode_start is not None:
            self.on_decode_start()
        return forced

    def finish(self, generated: torch.Tensor, prompt: torch.Tensor) -> dict:
        """Validate complete teacher forcing and return separate prefill/decode NLL."""
        if self.step != self.targets.shape[1]:
            raise ValueError("Teacher-forcing generation is incomplete")
        if self.nonfinite.item() or not torch.isfinite(self.nll).all().item():
            raise ValueError("nonfinite original generation scores")
        if (
            generated.shape != (self.targets.shape[0], self.prompt_length + self.step)
            or not torch.equal(generated[:, : self.prompt_length], prompt)
            or not torch.equal(generated[:, self.prompt_length :], self.targets)
        ):
            raise ValueError("Generated tokens differ from teacher-forcing targets")
        nll = self.nll.double()
        return {
            "first_token": _metric(nll[:, 0].sum().item(), nll.shape[0]),
            "decode": _metric(nll[:, 1:].sum().item(), nll[:, 1:].numel()),
            "all": _metric(nll.sum().item(), nll.numel()),
        }


def _profile_decode(generate, prompt, processor, backend):
    from tests.benchmarks.scripts.quantized_inference_report import (
        compiled_generation_regions,
        quantized_cuda_kernels,
    )

    trace = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    )
    started = False

    def start_decode():
        nonlocal started
        # Drain queued prefill work before enabling CUDA activity collection.
        torch.cuda.synchronize()
        trace.start()
        started = True

    processor.on_decode_start = start_decode
    try:
        generated = generate(prompt, processor)
    finally:
        if started:
            torch.cuda.synchronize()
            trace.stop()
    if not started:
        raise ValueError("Generation did not reach decode profiling")
    events = trace.key_averages()
    return generated, {
        "profiled_segment": 0,
        "prefill_excluded": True,
        "quantized_cuda_kernels": quantized_cuda_kernels(events, backend=backend),
        "compiled_regions": compiled_generation_regions(events),
    }


def evaluate_generation(
    generate: Callable,
    tokens: torch.Tensor,
    *,
    prompt_length: int,
    new_tokens: int,
    batch_size: int,
    device: str,
    profile_cuda: bool = False,
    backend: str = "gemlite_int8",
) -> dict:
    """Score non-overlapping real corpus blocks using the supplied generation path.

    generate(prompt, processor) must disable logits transforms and return full
    token sequences. Complete blocks are used without wrapping/padding; the tail
    is reported as discarded. This diagnostic is never throughput timing.
    """
    if (
        any(
            type(value) is not int or value < 1 for value in (prompt_length, batch_size)
        )
        or type(new_tokens) is not int
        or new_tokens < 2
        or tokens.dtype != torch.long
        or tokens.ndim != 1
    ):
        raise ValueError("Invalid generation quality shape")
    width = prompt_length + new_tokens
    block_size = batch_size * width
    segments = tokens.numel() // block_size
    if segments < 1:
        raise ValueError("Insufficient complete generation quality blocks")
    totals = {key: [0.0, 0] for key in ("first_token", "decode", "all")}
    prompt_hash, target_hash = hashlib.sha256(), hashlib.sha256()
    evidence = None
    with torch.inference_mode():
        for segment in range(segments):
            block = tokens[segment * block_size : (segment + 1) * block_size].reshape(
                batch_size, width
            )
            prompt, targets = block[:, :prompt_length], block[:, prompt_length:]
            for digest, value in ((prompt_hash, prompt), (target_hash, targets)):
                digest.update(
                    json.dumps(value.tolist(), separators=(",", ":")).encode()
                )
            prompt, targets = prompt.to(device), targets.to(device)
            processor = TeacherForcedScores(targets, prompt_length=prompt_length)
            if profile_cuda and segment == 0:
                generated, evidence = _profile_decode(
                    generate, prompt, processor, backend
                )
            else:
                generated = generate(prompt, processor)
            summary = processor.finish(generated, prompt)
            for key, values in totals.items():
                values[0] += summary[key]["nll_sum"]
                values[1] += summary[key]["tokens"]
            del generated, processor
    return {
        "scope": "teacher-forced generation likelihood; not free-running quality",
        "timing_used_for_throughput": False,
        "segments": segments,
        "discarded_tokens": tokens.numel() - segments * block_size,
        "prompt_sha256": prompt_hash.hexdigest(),
        "target_sha256": target_hash.hexdigest(),
        "execution_evidence": evidence,
        **{key: _metric(*values) for key, values in totals.items()},
    }
