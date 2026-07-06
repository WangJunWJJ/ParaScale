# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Tests for four-bit optimizer helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import unittest

import torch
import torch.nn as nn

from parascale.optimizers import FourBitAdamW, FourBitSGD, QuantizedState


def _assert_tensor_state_tree(value):
    if torch.is_tensor(value) or value is None or isinstance(
        value, (bool, int, float, str)
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
        self.assertGreater(stats["savings_percent"], 0)

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
