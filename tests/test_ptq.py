# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from parascale.quantization import (
    PostTrainingQuantization,
    QuantizationConfig,
    load_quantized_model,
    ptq_quantize,
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


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.fc(x.view(-1, 64 * 8 * 8))


def create_calibration_data(num_samples=320, input_shape=(784,)):
    x = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)


def create_cnn_calibration_data(num_samples=320):
    return create_calibration_data(num_samples=num_samples, input_shape=(3, 32, 32))


def _run_ptq(model, config, calib_loader):
    ptq = PostTrainingQuantization(model, config)
    prepared = ptq.prepare()
    assert prepared is ptq.calibrated_model
    ptq.calibrate(calib_loader)
    quantized_model = ptq.convert()
    assert ptq.is_converted
    assert ptq.quantized_model is quantized_model
    return ptq, quantized_model


def test_ptq_basic_flow():
    config = QuantizationConfig(
        mode="ptq", bits=8, scheme="symmetric", per_channel=True, calib_batches=10
    )
    _ptq, quantized_model = _run_ptq(SimpleModel(), config, create_calibration_data())

    with torch.no_grad():
        output = quantized_model(torch.randn(32, 784))

    assert output.shape == (32, 10)


def test_ptq_with_module_fusion():
    config = QuantizationConfig(
        mode="ptq",
        bits=8,
        scheme="symmetric",
        per_channel=True,
        calib_batches=10,
        fuse_modules=True,
    )
    _ptq, quantized_model = _run_ptq(CNNModel(), config, create_cnn_calibration_data())

    with torch.no_grad():
        output = quantized_model(torch.randn(32, 3, 32, 32))

    assert output.shape == (32, 10)


def test_ptq_int4_and_asymmetric_variants():
    variants = [
        QuantizationConfig(
            mode="ptq", bits=4, scheme="symmetric", per_channel=True, calib_batches=10
        ),
        QuantizationConfig(
            mode="ptq", bits=8, scheme="asymmetric", per_channel=True, calib_batches=10
        ),
    ]
    for config in variants:
        _ptq, quantized_model = _run_ptq(
            SimpleModel(), config, create_calibration_data()
        )
        with torch.no_grad():
            output = quantized_model(torch.randn(32, 784))
        assert output.shape == (32, 10)


def test_ptq_quantization_params_and_info():
    config = QuantizationConfig(mode="ptq", bits=8, calib_batches=10)
    ptq, _quantized_model = _run_ptq(SimpleModel(), config, create_calibration_data())

    quant_params = ptq.get_quantization_params()
    info = ptq.get_quantization_info()

    assert quant_params
    for params in quant_params.values():
        assert "scale" in params
        assert "zero_point" in params
    assert info["bits"] == 8
    assert info["scheme"] == config.scheme
    assert 0 <= info["quantization_ratio"] <= 1


def test_ptq_export_load_restores_quantized_model_outputs():
    config = QuantizationConfig(mode="ptq", bits=8, calib_batches=10)
    ptq, quantized_model = _run_ptq(SimpleModel(), config, create_calibration_data())

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ptq.export(tmp_path)
        loaded_model, loaded_config, loaded_params = load_quantized_model(
            tmp_path, model=SimpleModel()
        )

        x = torch.randn(32, 784)
        with torch.no_grad():
            expected = quantized_model(x)
            actual = loaded_model(x)

        assert loaded_config.bits == config.bits
        assert loaded_params
        assert torch.allclose(expected, actual, rtol=1e-4)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ptq_evaluate_and_convenience_function():
    config = QuantizationConfig(mode="ptq", bits=8, calib_batches=10)
    calib_loader = create_calibration_data()
    ptq, _quantized_model = _run_ptq(SimpleModel(), config, calib_loader)

    loss, accuracy = ptq.evaluate(
        create_calibration_data(num_samples=64),
        nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )
    assert loss >= 0
    assert 0 <= accuracy <= 100

    convenience_model = ptq_quantize(SimpleModel(), config, calib_loader)
    with torch.no_grad():
        output = convenience_model(torch.randn(32, 784))
    assert output.shape == (32, 10)
