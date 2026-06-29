# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:55
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""VLM processor, prompt and pipeline cache helpers."""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from parascale.runtime.specs import VlmLoraSpec

from .common import _require_pil_image

_VLM_PROMPT_CACHE: Dict[str, str] = {}


def _vlm_processor_cache_key(
    processor: Any,
    samples: Sequence[Mapping[str, Any]],
    prompts: Sequence[str],
    spec: VlmLoraSpec,
) -> str | None:
    if not spec.pipeline_cache:
        return None
    hasher = hashlib.sha256()
    hasher.update(_vlm_processor_config_hash(processor, spec).encode("utf-8"))
    for sample, prompt in zip(samples, prompts):
        hasher.update(_vlm_sample_fingerprint(sample).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(prompt.encode("utf-8", errors="replace"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _vlm_processor_config_hash(processor: Any, spec: VlmLoraSpec) -> str:
    payload = {
        "processor": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
        "tokenizer": _component_name(getattr(processor, "tokenizer", None)),
        "image_processor": _component_name(getattr(processor, "image_processor", None)),
        "model_path": spec.pretrained_model_name_or_path,
        "image_size": spec.image_size,
        "text_length": spec.text_length,
        "conversation_template": spec.conversation_template,
        "response_template": spec.response_template,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _component_name(component: Any) -> str:
    if component is None:
        return "none"
    return f"{component.__class__.__module__}.{component.__class__.__name__}"


def _vlm_sample_fingerprint(sample: Mapping[str, Any]) -> str:
    metadata = sample.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    image_value = sample.get("image", sample.get("pixel_values", sample.get("images")))
    candidates = []
    if isinstance(image_value, (str, Path)):
        candidates.append(Path(image_value))
    source_tar = metadata.get("source_tar")
    if source_tar:
        candidates.append(Path(str(source_tar)))
    parts = [
        str(metadata.get("sample_id", "")),
        str(sample.get("text", "")),
    ]
    for path in candidates:
        try:
            stat = path.stat()
            parts.extend([str(path), str(stat.st_mtime_ns), str(stat.st_size)])
            return "|".join(parts)
        except OSError:
            parts.append(str(path))
    if isinstance(image_value, (bytes, bytearray)):
        parts.append(hashlib.sha256(bytes(image_value)).hexdigest())
    elif hasattr(image_value, "shape"):
        parts.append(str(tuple(int(dim) for dim in image_value.shape)))
    return "|".join(parts)


def _vlm_cache_dir(spec: VlmLoraSpec) -> Path:
    root = spec.pipeline_cache_dir or ".parascale_cache/vlm_processor"
    return Path(root) / "processor"


def _vlm_prompt_cache_dir(spec: VlmLoraSpec) -> Path:
    if spec.prompt_template_cache_dir:
        return Path(spec.prompt_template_cache_dir)
    root = spec.pipeline_cache_dir or ".parascale_cache/vlm_processor"
    return Path(root) / "prompts"


def _cache_file_is_fresh(path: Path, ttl_seconds: float) -> bool:
    if ttl_seconds <= 0:
        return True
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age <= ttl_seconds


def _prune_cache_dir(
    root: Path,
    *,
    max_entries: int,
    max_bytes: int,
    ttl_seconds: float,
) -> None:
    try:
        files = [path for path in root.glob("*") if path.is_file()]
    except OSError:
        return
    now = time.time()
    kept: list[tuple[float, int, Path]] = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        if ttl_seconds > 0 and now - stat.st_mtime > ttl_seconds:
            try:
                path.unlink()
            except OSError:
                pass
            continue
        kept.append((stat.st_mtime, stat.st_size, path))
    kept.sort(key=lambda item: item[0])
    total_bytes = sum(size for _mtime, size, _path in kept)
    while len(kept) > max_entries or total_bytes > max_bytes:
        _mtime, size, path = kept.pop(0)
        try:
            path.unlink()
        except OSError:
            pass
        total_bytes -= size


def _load_vlm_processor_cache(torch: Any, spec: VlmLoraSpec, key: str | None) -> Any:
    if not key:
        return None
    path = _vlm_cache_dir(spec) / f"{key}.pt"
    if not path.exists():
        return None
    if not _cache_file_is_fresh(path, spec.pipeline_cache_ttl_seconds):
        try:
            path.unlink()
        except OSError:
            pass
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _save_vlm_processor_cache(
    torch: Any, spec: VlmLoraSpec, key: str | None, encoded: Any
) -> None:
    if not key:
        return
    path = _vlm_cache_dir(spec) / f"{key}.pt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            item_key: (
                item_value.detach().cpu()
                if hasattr(item_value, "detach")
                else item_value
            )
            for item_key, item_value in dict(encoded).items()
        }
        torch.save(payload, path)
        _prune_cache_dir(
            path.parent,
            max_entries=max(1, int(spec.pipeline_cache_max_entries)),
            max_bytes=max(1, int(spec.pipeline_cache_max_bytes)),
            ttl_seconds=max(0.0, float(spec.pipeline_cache_ttl_seconds)),
        )
    except Exception:
        return


class _TimedCallableProxy:
    def __init__(self, wrapped: Any, metric_name: str, profile: Dict[str, float]):
        self._wrapped = wrapped
        self._metric_name = metric_name
        self._profile = profile

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return self._wrapped(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self._profile[self._metric_name] = (
                self._profile.get(self._metric_name, 0.0) + elapsed
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _timed_vlm_processor_call(
    processor: Any, **kwargs: Any
) -> tuple[Any, Dict[str, float]]:
    profile: Dict[str, float] = {}
    patched: list[tuple[str, Any]] = []
    for attr, metric_name in [
        ("tokenizer", "tokenizer_ms"),
        ("image_processor", "image_processor_ms"),
    ]:
        component = getattr(processor, attr, None)
        if component is None or not callable(component):
            continue
        try:
            setattr(
                processor,
                attr,
                _TimedCallableProxy(component, metric_name, profile),
            )
            patched.append((attr, component))
        except Exception:
            continue
    try:
        return processor(**kwargs), profile
    finally:
        for attr, component in patched:
            try:
                setattr(processor, attr, component)
            except Exception:
                pass


def _normalize_pipeline_profile(profile: Mapping[str, Any]) -> Dict[str, float]:
    keys = [
        "shard_read_ms",
        "tar_open_ms",
        "sample_decode_ms",
        "sample_tensor_build_ms",
        "sample_build_ms",
        "collate_ms",
        "image_decode_ms",
        "prompt_template_ms",
        "processor_ms",
        "tokenizer_ms",
        "image_processor_ms",
        "host_to_device_ms",
        "label_build_ms",
        "cache_hit",
    ]
    return {key: max(0.0, float(profile.get(key, 0.0) or 0.0)) for key in keys}


def _pipeline_profile_from_sample_metadata(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    profile = {
        "shard_read_ms": 0.0,
        "tar_open_ms": 0.0,
        "sample_decode_ms": 0.0,
        "sample_tensor_build_ms": 0.0,
        "sample_build_ms": 0.0,
    }
    for sample in samples:
        metadata = sample.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        profile["shard_read_ms"] += float(metadata.get("wds_shard_read_ms", 0.0) or 0.0)
        profile["tar_open_ms"] += float(metadata.get("wds_tar_open_ms", 0.0) or 0.0)
        profile["sample_decode_ms"] += float(
            metadata.get("wds_image_decode_ms", 0.0) or 0.0
        )
        profile["sample_tensor_build_ms"] += float(
            metadata.get("wds_tensor_build_ms", 0.0) or 0.0
        )
        profile["sample_build_ms"] += float(
            metadata.get("wds_sample_build_ms", 0.0) or 0.0
        )
    return profile


def _vlm_prompt(processor: Any, sample: Mapping[str, Any], spec: VlmLoraSpec) -> str:
    text = str(sample.get(spec.prompt_field, sample.get("text", ""))).strip()
    if not text:
        text = spec.response_template
    cache_key = _vlm_prompt_cache_key(processor, text, spec)
    cached = _load_vlm_prompt_cache(spec, cache_key)
    if cached is not None:
        return cached
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        },
        {"role": "assistant", "content": spec.response_template},
    ]
    apply_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_template):
        try:
            prompt = apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            _save_vlm_prompt_cache(spec, cache_key, prompt)
            return prompt
        except Exception:
            pass
    template = spec.conversation_template.lower()
    if "llava" in template:
        prompt = f"<image>\nUSER: {text}\nASSISTANT: {spec.response_template}"
        _save_vlm_prompt_cache(spec, cache_key, prompt)
        return prompt
    if "internvl" in template:
        prompt = f"<image>\n{text}\n{spec.response_template}"
        _save_vlm_prompt_cache(spec, cache_key, prompt)
        return prompt
    prompt = (
        f"<|vision_start|><|image_pad|><|vision_end|>{text}\n"
        f"{spec.response_template}"
    )
    _save_vlm_prompt_cache(spec, cache_key, prompt)
    return prompt


def _vlm_prompt_cache_key(processor: Any, text: str, spec: VlmLoraSpec) -> str:
    payload = {
        "processor": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
        "conversation_template": spec.conversation_template,
        "response_template": spec.response_template,
        "text": text,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_vlm_prompt_cache(spec: VlmLoraSpec, cache_key: str) -> str | None:
    cached = _VLM_PROMPT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not spec.prompt_template_cache:
        return None
    path = _vlm_prompt_cache_dir(spec) / f"{cache_key}.txt"
    if not path.exists():
        return None
    if not _cache_file_is_fresh(path, spec.pipeline_cache_ttl_seconds):
        try:
            path.unlink()
        except OSError:
            pass
        return None
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError:
        return None
    _VLM_PROMPT_CACHE[cache_key] = prompt
    return prompt


def _save_vlm_prompt_cache(spec: VlmLoraSpec, cache_key: str, prompt: str) -> None:
    _VLM_PROMPT_CACHE[cache_key] = prompt
    if not spec.prompt_template_cache:
        return
    path = _vlm_prompt_cache_dir(spec) / f"{cache_key}.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt, encoding="utf-8")
        _prune_cache_dir(
            path.parent,
            max_entries=max(1, int(spec.pipeline_cache_max_entries)),
            max_bytes=max(1, int(spec.pipeline_cache_max_bytes)),
            ttl_seconds=max(0.0, float(spec.pipeline_cache_ttl_seconds)),
        )
    except OSError:
        return


def _sample_to_pil_image(sample: Mapping[str, Any], spec: VlmLoraSpec):
    Image = _require_pil_image()
    value = sample.get("image", sample.get("pixel_values", sample.get("images")))
    if value is None:
        return Image.new("RGB", (spec.image_size, spec.image_size), color=(0, 0, 0))
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.clamp(0, 1).mul(255).byte().numpy()
        return Image.fromarray(array).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        try:
            return (
                Image.open(io.BytesIO(value))
                .convert("RGB")
                .resize((spec.image_size, spec.image_size))
            )
        except Exception:
            return Image.new("RGB", (spec.image_size, spec.image_size), color=(0, 0, 0))
    if isinstance(value, Image.Image):
        return value.convert("RGB").resize((spec.image_size, spec.image_size))
    return Image.open(value).convert("RGB").resize((spec.image_size, spec.image_size))
