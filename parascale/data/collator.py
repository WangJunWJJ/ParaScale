# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午12:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Collators for multimodal training batches."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .multimodal import estimate_multimodal_token_cost, normalize_multimodal_sample
from .schema import MultiModalBatchSchema


class MultiModalCollator:
    """Collate common multimodal dictionaries with optional text padding/packing."""

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        max_length: Optional[int] = None,
        pack_text: bool = False,
        schema: Optional[MultiModalBatchSchema] = None,
        processors: Optional[Mapping[str, Callable[[Any], Any]]] = None,
        return_tensors: Optional[str] = None,
        include_token_cost: bool = True,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length
        self.pack_text = pack_text
        self.schema = schema or MultiModalBatchSchema()
        self.processors = dict(processors or {})
        self.return_tensors = return_tensors
        self.include_token_cost = include_token_cost

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        collate_start = time.perf_counter()
        batch: Dict[str, Any] = {}
        if not samples:
            return batch

        samples = [
            normalize_multimodal_sample(self._process_sample(sample), self.schema)
            for sample in samples
        ]

        text_keys = {
            self.schema.input_ids,
            self.schema.attention_mask,
            self.schema.labels,
        }
        for key in samples[0].keys():
            values = [sample.get(key) for sample in samples]
            if key in text_keys and all(
                isinstance(value, Sequence) for value in values
            ):
                pad_value = self._pad_value_for_key(key)
                padded = self._pad_sequences(values, pad_value)
                batch[key] = self._maybe_tensor(padded, dtype=self._dtype_for_key(key))
            else:
                batch[key] = self._stack_or_list(values)

        if self.pack_text and "input_ids" in batch:
            batch = self._pack_text_batch(batch)
        if self.include_token_cost:
            costs = [
                estimate_multimodal_token_cost(sample).to_dict() for sample in samples
            ]
            batch["token_cost"] = costs
            batch["tokens"] = sum(cost["total_tokens"] for cost in costs)
            batch["image_tokens"] = sum(cost["image_tokens"] for cost in costs)
            if all(
                sample.get(self.schema.pixel_values) is not None for sample in samples
            ):
                batch["num_images"] = len(samples)
                batch["num_pairs"] = len(samples)
            if costs and batch["tokens"]:
                max_tokens = max(cost["total_tokens"] for cost in costs)
                batch["padding_ratio"] = 1.0 - (
                    float(batch["tokens"]) / max(float(max_tokens * len(costs)), 1.0)
                )
        profile = self._pipeline_profile_from_metadata(samples)
        profile["collate_ms"] = (time.perf_counter() - collate_start) * 1000.0
        if any(value > 0.0 for value in profile.values()):
            batch["pipeline_profile"] = profile
        return batch

    def _pipeline_profile_from_metadata(
        self, samples: Sequence[Mapping[str, Any]]
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
            profile["shard_read_ms"] += float(
                metadata.get("wds_shard_read_ms", 0.0) or 0.0
            )
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

    def _process_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        processed = dict(sample)
        for key, processor in self.processors.items():
            if key in processed:
                processed[key] = processor(processed[key])
        return processed

    def _pad_value_for_key(self, key: str) -> int:
        if key == self.schema.labels:
            return self.label_pad_token_id
        if key == self.schema.attention_mask:
            return 0
        return self.pad_token_id

    def _dtype_for_key(self, key: str) -> Any:
        if self.return_tensors != "pt":
            return None
        try:
            import torch
        except Exception:
            return None
        if key in {
            self.schema.input_ids,
            self.schema.attention_mask,
            self.schema.labels,
        }:
            return torch.long
        return None

    def _pad_sequences(
        self, values: Sequence[Sequence[Any]], pad_value: int
    ) -> List[List[Any]]:
        max_len = max(len(value) for value in values)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        padded = []
        for value in values:
            trimmed = list(value[:max_len])
            padded.append(trimmed + [pad_value] * (max_len - len(trimmed)))
        return padded

    def _stack_or_list(self, values: Sequence[Any]) -> Any:
        first = values[0]
        if hasattr(first, "shape"):
            try:
                import torch

                return torch.stack(list(values), dim=0)
            except Exception:
                return list(values)
        return list(values)

    def _maybe_tensor(self, value: Any, dtype: Any = None) -> Any:
        if self.return_tensors != "pt":
            return value
        try:
            import torch

            return torch.tensor(value, dtype=dtype)
        except Exception:
            return value

    def _pack_text_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        max_len = self.max_length
        if max_len is None:
            return batch

        packed_ids: List[List[int]] = []
        packed_labels: List[List[int]] = []
        current_ids: List[int] = []
        current_labels: List[int] = []

        input_key = self.schema.input_ids
        label_key = self.schema.labels
        mask_key = self.schema.attention_mask
        labels = batch.get(
            label_key,
            [[self.label_pad_token_id] * len(ids) for ids in batch[input_key]],
        )
        for ids, label_seq in zip(batch[input_key], labels):
            valid_pairs = [
                (token, label)
                for token, label in zip(ids, label_seq)
                if token != self.pad_token_id
            ]
            for token, label in valid_pairs:
                if len(current_ids) >= max_len:
                    packed_ids.append(current_ids)
                    packed_labels.append(current_labels)
                    current_ids, current_labels = [], []
                current_ids.append(token)
                current_labels.append(label)

        if current_ids:
            packed_ids.append(current_ids)
            packed_labels.append(current_labels)

        batch[input_key] = self._pad_sequences(packed_ids, self.pad_token_id)
        batch[label_key] = self._pad_sequences(packed_labels, self.label_pad_token_id)
        batch[mask_key] = [
            [1 if token != self.pad_token_id else 0 for token in row]
            for row in batch[input_key]
        ]
        return batch
