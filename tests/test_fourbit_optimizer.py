# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Tests for four-bit optimizer helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import unittest

import pytest
import torch
import torch.nn as nn

from parascale.optimizers import FourBitAdamW, FourBitSGD, QuantizedState


def _assert_tensor_state_tree(value):
    if (
        torch.is_tensor(value)
        or value is None
        or isinstance(value, (bool, int, float, str))
    ):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_tensor_state_tree(key)
            _assert_tensor_state_tree(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_tensor_state_tree(item)
        return
    raise AssertionError(f"non-portable optimizer state value: {type(value)!r}")


def test_four_bit_optimizer_state_dict_uses_portable_tensor_tree():
    model = nn.Linear(4, 2)
    optimizer = FourBitAdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state_dict()

    _assert_tensor_state_tree(state)
    assert state["parascale_optimizer"]["state_schema_version"] == 1


def test_quantized_state_pack_and_unpack_do_not_use_python_range(monkeypatch):
    import parascale.optimizers.optimizers as optimizer_module

    def forbidden_range(*_args, **_kwargs):
        raise AssertionError("4-bit pack/unpack must be vectorized")

    monkeypatch.setattr(optimizer_module, "range", forbidden_range, raising=False)

    state = optimizer_module.QuantizedState(torch.randn(256), group_size=128)
    restored = state.dequantize()

    assert restored.shape == (256,)


def test_four_bit_adamw_reuses_persistent_quantized_state_buffers():
    model = nn.Linear(4, 2)
    optimizer = FourBitAdamW(model.parameters(), lr=0.01)
    parameter = next(model.parameters())

    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    state = optimizer.state[parameter]["exp_avg"]
    data_ptr = state.quantized_data.data_ptr()

    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()

    assert optimizer.state[parameter]["exp_avg"] is state
    assert state.quantized_data.data_ptr() == data_ptr


def test_quantized_state_update_returns_error_without_redecode():
    original = torch.randn(256)
    updated = torch.randn(256)
    state = QuantizedState(original, group_size=128)

    error = state.update_and_error(updated)

    torch.testing.assert_close(error, updated - state.dequantize())


def test_block_scaled_fp16_residual_preserves_tiny_second_moment_error():
    torch.manual_seed(7)
    updated = torch.rand(256) * 1e-7
    state = QuantizedState(torch.zeros_like(updated), group_size=128)

    normalized = state.update_and_normalized_error(updated).half()
    restored_error = state.denormalize_error(normalized.float())
    absolute_error = updated - state.dequantize()

    assert torch.count_nonzero(absolute_error.half()) == 0
    assert torch.count_nonzero(normalized) > 0
    torch.testing.assert_close(restored_error, absolute_error, rtol=1e-3, atol=1e-12)


@pytest.mark.parametrize("optimizer_type,slots", [(FourBitAdamW, 2), (FourBitSGD, 1)])
@pytest.mark.parametrize("size", [1, 127, 129, 32768])
@pytest.mark.parametrize(
    "compensate,dtype", [(False, None), (True, None), (True, "fp16"), (True, "fp32")]
)
def test_memory_stats_include_all_resident_compensation(
    optimizer_type, slots, size, compensate, dtype
):
    parameter = nn.Parameter(torch.ones(size))
    unused = nn.Parameter(torch.ones(65536))
    optimizer = optimizer_type(
        [parameter, unused],
        compensate_quant_error=compensate,
        error_compensation_dtype=dtype,
    )
    empty = optimizer.get_memory_stats()
    assert empty["state_bytes"] == empty["standard_bytes"] == 0
    assert empty["savings_percent"] == 0
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    packed = slots * ((size + 127) // 128) * (64 + 8)
    residual = slots * size * (2 if dtype == "fp16" else 4) if compensate else 0
    stats = optimizer.get_memory_stats()
    assert stats["total_params"] == size + 65536
    assert stats["initialized_params"] == size
    assert stats["quantized_bytes"] == packed
    assert stats["state_bytes"] == packed + residual
    assert stats["state_bytes_by_device"] == {"cpu": packed + residual}
    assert stats["standard_bytes"] == slots * size * 4
    assert stats["savings_percent"] == pytest.approx(
        100 * (1 - (packed + residual) / (slots * size * 4))
    )
    optimizer.state[parameter]["alias"] = optimizer.state[parameter][
        "exp_avg" if slots == 2 else "momentum_buffer"
    ].quantized_data[1:2]
    assert optimizer.get_memory_stats()["state_bytes"] == packed + residual
    optimizer.state[parameter]["backing_view"] = torch.zeros(100)[17:18]
    assert optimizer.get_memory_stats()["state_bytes"] == packed + residual + 400


def test_quantized_memory_usage_uses_actual_storage_dtype():
    state = QuantizedState(torch.ones(128, dtype=torch.float16))
    assert state.memory_usage() == 64 + 2 + 2


def test_memory_stats_use_resident_dtype_after_checkpoint_reload():
    parameter = nn.Parameter(torch.ones(129))
    source = FourBitAdamW([parameter], error_compensation_dtype="fp16")
    parameter.grad = torch.ones_like(parameter)
    source.step()
    restored_parameter = nn.Parameter(parameter.detach().clone())
    target = FourBitAdamW([restored_parameter], error_compensation_dtype="fp16")
    target.load_state_dict(source.state_dict())
    state = target.state[restored_parameter]
    expected = sum(state[key].memory_usage() for key in ("exp_avg", "exp_avg_sq"))
    expected += sum(
        state[key].numel() * state[key].element_size()
        for key in ("exp_avg_error", "exp_avg_sq_error")
    )
    assert target.get_memory_stats()["state_bytes"] == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_memory_stats_separate_cpu_and_cuda_storage():
    parameter = nn.Parameter(torch.ones(128, device="cuda"))
    optimizer = FourBitAdamW([parameter], compensate_quant_error=False)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["host_state"] = torch.zeros(10)
    stats = optimizer.get_memory_stats()
    assert stats["state_bytes_by_device"] == {str(parameter.device): 144, "cpu": 40}
    assert stats["state_bytes"] == 184


class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestQuantizedState(unittest.TestCase):
    def test_quantize_dequantize(self):
        original = torch.randn(100, 50)
        qs = QuantizedState(tensor=original, group_size=64)
        reconstructed = qs.dequantize()
        self.assertEqual(reconstructed.shape, original.shape)
        relative_error = torch.norm(reconstructed - original) / torch.norm(original)
        self.assertLess(relative_error.item(), 0.15)

    def test_memory_savings(self):
        original = torch.randn(1000, 1000)
        fp32_bytes = original.numel() * 4
        qs = QuantizedState(tensor=original, group_size=128)
        quantized_bytes = qs.memory_usage()
        savings_ratio = 1 - quantized_bytes / fp32_bytes
        self.assertGreater(savings_ratio, 0.5)

    def test_update(self):
        original = torch.randn(100, 50)
        qs = QuantizedState(tensor=original, group_size=64)
        new_tensor = torch.randn(100, 50)
        qs.update(new_tensor)
        reconstructed = qs.dequantize()
        self.assertEqual(reconstructed.shape, new_tensor.shape)

    def test_device_transfer(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        original = torch.randn(100, 50)
        qs = QuantizedState(tensor=original, group_size=64)
        qs.to(torch.device("cuda"))
        self.assertEqual(qs.quantized_data.device.type, "cuda")
        self.assertEqual(qs.scale.device.type, "cuda")

    def test_sparseify(self):
        original = torch.randn(100, 50)
        original[int(original.numel() * 0.7) :] = 0
        qs = QuantizedState(tensor=original, group_size=64)
        sparse_qs = qs.sparseify(threshold=1e-05)
        self.assertTrue(hasattr(sparse_qs, "is_sparse"))
        self.assertTrue(sparse_qs.is_sparse)
        reconstructed = sparse_qs.dequantize()
        self.assertEqual(reconstructed.shape, original.shape)

    def test_memory_usage_sparse(self):
        original = torch.randn(100, 50)
        qs = QuantizedState(tensor=original, group_size=64)
        dense_memory = qs.memory_usage()
        self.assertGreater(dense_memory, 0)
        sparse_qs = qs.sparseify(threshold=1e-05)
        sparse_memory = sparse_qs.memory_usage()
        self.assertGreater(sparse_memory, 0)


class TestFourBitAdamW(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SimpleModel()
        self.criterion = nn.MSELoss()

    def test_initialization(self):
        optimizer = FourBitAdamW(self.model.parameters(), lr=0.001)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)

    def test_step(self):
        optimizer = FourBitAdamW(self.model.parameters(), lr=0.001)
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

    def test_memory_stats(self):
        optimizer = FourBitAdamW(self.model.parameters(), lr=0.001)
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats = optimizer.get_memory_stats()
        self.assertIn("total_params", stats)
        self.assertIn("savings_percent", stats)
        self.assertGreater(stats["state_bytes"], stats["standard_bytes"])
        self.assertLess(stats["savings_percent"], 0)

    def test_state_dict(self):
        optimizer = FourBitAdamW(self.model.parameters(), lr=0.001)
        for _ in range(3):
            x = torch.randn(4, 10)
            target = torch.randn(4, 5)
            output = self.model(x)
            loss = self.criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        state_dict = optimizer.state_dict()
        new_model = SimpleModel()
        new_optimizer = FourBitAdamW(new_model.parameters(), lr=0.001)
        new_optimizer.load_state_dict(state_dict)
        self.assertEqual(
            new_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"]
        )

    def test_training_convergence(self):
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)
        model = SimpleModel()
        optimizer = FourBitAdamW(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        initial_loss = None
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if epoch == 0:
                initial_loss = loss.item()
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss)

    def test_error_compensation(self):
        optimizer_with_comp = FourBitAdamW(
            self.model.parameters(), lr=0.001, compensate_quant_error=True
        )
        optimizer_without_comp = FourBitAdamW(
            self.model.parameters(), lr=0.001, compensate_quant_error=False
        )
        for optimizer in [optimizer_with_comp, optimizer_without_comp]:
            x = torch.randn(4, 10)
            target = torch.randn(4, 5)
            output = self.model(x)
            loss = self.criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_fp16_error_compensation(self):
        optimizer = FourBitAdamW(
            self.model.parameters(),
            lr=0.001,
            compensate_quant_error=True,
            error_compensation_dtype="fp16",
        )
        self.assertEqual(optimizer.error_compensation_dtype, "fp16")
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if "exp_avg_error" in state:
                    self.assertEqual(state["exp_avg_error"].dtype, torch.float16)
                    self.assertEqual(state["exp_avg_sq_error"].dtype, torch.float16)

    def test_invalid_error_compensation_dtype(self):
        with self.assertRaises(ValueError):
            FourBitAdamW(
                self.model.parameters(), lr=0.001, error_compensation_dtype="invalid"
            )

    def test_group_size_config(self):
        for group_size in [64, 128, 256]:
            optimizer = FourBitAdamW(
                self.model.parameters(), lr=0.001, group_size=group_size
            )
            x = torch.randn(4, 10)
            target = torch.randn(4, 5)
            output = self.model(x)
            loss = self.criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.assertEqual(optimizer.group_size, group_size)


class TestFourBitSGD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SimpleModel()
        self.criterion = nn.MSELoss()

    def test_initialization(self):
        optimizer = FourBitSGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)

    def test_step(self):
        optimizer = FourBitSGD(self.model.parameters(), lr=0.01, momentum=0.9)
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

    def test_training_convergence(self):
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)
        model = SimpleModel()
        optimizer = FourBitSGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.MSELoss()
        initial_loss = None
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if epoch == 0:
                initial_loss = loss.item()
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss)

    def test_nesterov(self):
        optimizer = FourBitSGD(
            self.model.parameters(), lr=0.01, momentum=0.9, nesterov=True
        )
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(True)

    def test_memory_stats(self):
        optimizer = FourBitSGD(self.model.parameters(), lr=0.01, momentum=0.9)
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats = optimizer.get_memory_stats()
        self.assertIn("total_params", stats)
        self.assertIn("savings_percent", stats)

    def test_fp16_error_compensation(self):
        optimizer = FourBitSGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            compensate_quant_error=True,
            error_compensation_dtype="fp16",
        )
        self.assertEqual(optimizer.error_compensation_dtype, "fp16")
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        output = self.model(x)
        loss = self.criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if "momentum_error" in state:
                    self.assertEqual(state["momentum_error"].dtype, torch.float16)


class TestComparisonWithStandardOptimizers(unittest.TestCase):
    def test_adamw_comparison(self):
        torch.manual_seed(42)
        model1 = SimpleModel()
        model2 = SimpleModel()
        model2.load_state_dict(model1.state_dict())
        optimizer1 = FourBitAdamW(model1.parameters(), lr=0.001)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        X = torch.randn(32, 10)
        y = torch.randn(32, 5)
        losses1 = []
        losses2 = []
        for epoch in range(20):
            optimizer1.zero_grad()
            output1 = model1(X)
            loss1 = criterion(output1, y)
            loss1.backward()
            optimizer1.step()
            losses1.append(loss1.item())
            optimizer2.zero_grad()
            output2 = model2(X)
            loss2 = criterion(output2, y)
            loss2.backward()
            optimizer2.step()
            losses2.append(loss2.item())
        self.assertLess(losses1[-1], losses1[0])
        self.assertLess(losses2[-1], losses2[0])
        loss_diff = abs(losses1[-1] - losses2[-1]) / max(
            abs(losses1[-1]), abs(losses2[-1]), 1e-08
        )
        self.assertLess(loss_diff, 0.5)


def run_simple_demo():
    print("\n" + "=" * 60)
    print("CUDA is not available")
    print("=" * 60)
    model = SimpleModel(input_dim=100, hidden_dim=200, output_dim=10)
    total_params = sum((p.numel() for p in model.parameters()))
    print(f"CUDA is not available{total_params:,}")
    print("\n--- 4bit AdamW ---")
    optimizer = FourBitAdamW(model.parameters(), lr=0.001, group_size=128)
    criterion = nn.MSELoss()
    for i in range(5):
        x = torch.randn(8, 100)
        target = torch.randn(8, 10)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Step {i + 1}, Loss: {loss.item():.4f}")
    optimizer.print_memory_stats()
    print("\n--- 4bit SGD ---")
    model2 = SimpleModel(input_dim=100, hidden_dim=200, output_dim=10)
    optimizer2 = FourBitSGD(model2.parameters(), lr=0.01, momentum=0.9)
    for i in range(5):
        x = torch.randn(8, 100)
        target = torch.randn(8, 10)
        optimizer2.zero_grad()
        output = model2(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer2.step()
        print(f"Step {i + 1}, Loss: {loss.item():.4f}")
    optimizer2.print_memory_stats()
    print("\n" + "=" * 60)
    print("CUDA is not available")
    print("=" * 60)


if __name__ == "__main__":
    run_simple_demo()
    print("CUDA is not available")
    unittest.main(argv=[""], verbosity=2, exit=False)
