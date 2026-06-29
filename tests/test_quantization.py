# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest
import torch
import torch.nn as nn

from parascale.quantization import (
    FakeQuantize,
    FakeQuantizedLinear,
    MinMaxObserver,
    MovingAverageObserver,
    QuantizationAwareTraining,
    QuantizationConfig,
    QuantizedConv2d,
    QuantizedLinear,
    calculate_scale_zero_point,
    dequantize_tensor,
    get_quantizable_layers,
    quantize_tensor,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


def test_quantization_config_validation():
    default_config = QuantizationConfig()
    assert default_config.enabled is False
    assert default_config.bits == 8
    assert default_config.scheme == "symmetric"

    config = QuantizationConfig(
        enabled=True,
        bits=8,
        scheme="asymmetric",
        per_channel=True,
        observer_type="moving_average",
    )
    assert config.get_qmin_qmax() == (-128, 127)

    with pytest.raises(ValueError):
        QuantizationConfig(bits=16)


def test_observers_calculate_qparams():
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")

    minmax = MinMaxObserver(config)
    minmax.update(torch.randn(8, 3, 16, 16))
    scale, zero_point = minmax.calculate_qparams()
    assert torch.all(scale > 0)
    assert torch.all(zero_point == 0)

    moving = MovingAverageObserver(config)
    for _ in range(3):
        moving.update(torch.randn(8, 3, 16, 16))
    scale, zero_point = moving.calculate_qparams()
    assert torch.all(scale > 0)
    assert torch.all(zero_point == 0)


def test_minmax_observer_window_returns_scalar_qparams():
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")
    observer = MinMaxObserver(config, window_size=5)

    for i in range(10):
        observer.update(torch.randn(32, 128) * (i + 1))

    assert len(observer.history_min) <= 5
    assert len(observer.history_max) <= 5
    scale, zero_point = observer.calculate_qparams()
    assert scale.numel() == 1
    assert zero_point.numel() == 1

    observer.reset()
    assert observer.history_min == []
    assert observer.history_max == []


def test_fake_quantize_can_be_enabled_and_disabled():
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")
    fake_quant = FakeQuantize(config)
    x = torch.randn(4, 3, 16, 16)

    fake_quant.train()
    y = fake_quant(x)
    assert y.shape == x.shape

    fake_quant.enable_fake_quant(False)
    assert torch.allclose(fake_quant(x), x)


def test_fake_quantized_linear_and_qat_flow():
    config = QuantizationConfig(
        enabled=True, bits=8, scheme="symmetric", fuse_modules=False
    )
    linear = FakeQuantizedLinear(784, 128, bias=True, config=config)
    assert linear(torch.randn(4, 784)).shape == (4, 128)
    assert hasattr(linear, "activation_fake_quant")
    assert hasattr(linear, "weight_fake_quant")

    model = SimpleModel()
    assert get_quantizable_layers(model) == ["fc1", "fc2"]

    qat = QuantizationAwareTraining(model, config)
    prepared = qat.prepare()
    assert hasattr(prepared.fc1, "activation_fake_quant")
    assert hasattr(prepared.fc2, "activation_fake_quant")

    optimizer = torch.optim.SGD(prepared.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(8, 1, 28, 28)
    target = torch.randint(0, 10, (8,))
    optimizer.zero_grad()
    loss = criterion(prepared(x), target)
    loss.backward()
    optimizer.step()

    qat.freeze_observer()
    assert qat.get_quantization_params()


def test_quantize_dequantize_tensor_helpers():
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")
    x = torch.randn(4, 3, 16, 16)
    scale, zero_point = calculate_scale_zero_point(torch.min(x), torch.max(x), config)

    x_quant = quantize_tensor(x, scale, zero_point, config)
    x_dequant = dequantize_tensor(x_quant, scale, zero_point)

    assert x_quant.shape == x.shape
    assert x_dequant.shape == x.shape
    assert torch.abs(x - x_dequant).mean() >= 0


def test_quantized_linear_and_conv2d_layers():
    config = QuantizationConfig(enabled=True, bits=8, scheme="symmetric")

    linear = QuantizedLinear(128, 64, bias=True, quant_config=config)
    linear_weight = torch.randint(-128, 127, (64, 128), dtype=torch.int8)
    linear.set_quantized_weight(linear_weight, torch.tensor(0.01), torch.tensor(0.0))
    assert linear(torch.randn(8, 128)).shape == (8, 64)

    conv = QuantizedConv2d(
        3, 16, kernel_size=(3, 3), padding=1, bias=True, quant_config=config
    )
    conv_weight = torch.randint(-128, 127, (16, 3, 3, 3), dtype=torch.int8)
    conv.set_quantized_weight(conv_weight, torch.tensor(0.01), torch.tensor(0.0))
    assert conv(torch.randn(4, 3, 32, 32)).shape == (4, 16, 32, 32)
