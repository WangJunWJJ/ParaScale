# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Post-training quantization helpers."""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import QuantizationConfig
from .fake_quantize import FakeQuantize, FakeQuantizedConv2d, FakeQuantizedLinear
from .quantized_layers import QuantizedConv2d, QuantizedLinear
from .utils import copy_model_weights, fuse_modules, get_quantizable_layers

logger = logging.getLogger(__name__)


class PostTrainingQuantization:

    def __init__(self, model: nn.Module, config: QuantizationConfig):
        if config.mode != "ptq":
            logger.warning(
                f"PTQ operation cannot continue{config.mode}PTQ operation cannot continue"
            )
        self.model = model
        self.config = config
        self.calibrated_model: Optional[nn.Module] = None
        self.quantized_model: Optional[nn.Module] = None
        self.is_calibrated = False
        self.is_converted = False

    def prepare(self) -> nn.Module:
        logger.info("PTQ operation cannot continue")
        self.calibrated_model = self._copy_model(self.model)
        if self.config.fuse_modules:
            logger.info("PTQ operation cannot continue")
            fuse_modules(self.calibrated_model)
        logger.info("PTQ operation cannot continue")
        self._insert_fake_quantize(self.calibrated_model)
        logger.info("PTQ operation cannot continue")
        copy_model_weights(self.model, self.calibrated_model)
        logger.info(
            f"PTQ operation cannot continue{len(get_quantizable_layers(self.calibrated_model))}PTQ operation cannot continue"
        )
        return self.calibrated_model

    def calibrate(
        self,
        calib_loader,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if self.calibrated_model is None:
            raise RuntimeError("PTQ operation cannot continue")
        logger.info(
            f"PTQ operation cannot continue{self.config.calib_batches}PTQ operation cannot continue"
        )
        self.calibrated_model.eval()
        bn_training_state = {}
        for name, module in self.calibrated_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                bn_training_state[name] = module.training
                module.eval()
        total_batches = min(self.config.calib_batches, len(calib_loader))
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(calib_loader):
                if batch_idx >= total_batches:
                    break
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0]
                else:
                    inputs = batch_data
                _ = self.calibrated_model(inputs)
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches)
                elif (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"PTQ operation cannot continue{batch_idx + 1}/{total_batches}"
                    )
        for name, module in self.calibrated_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                if name in bn_training_state:
                    if bn_training_state[name]:
                        module.train()
        logger.info("PTQ operation cannot continue")
        self.is_calibrated = True
        logger.info("PTQ operation cannot continue")
        self.freeze_observer()

    def freeze_observer(self) -> None:
        if self.calibrated_model is None:
            raise RuntimeError("PTQ operation cannot continue")
        count = 0
        for module in self.calibrated_model.modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer(False)
                count += 1
        logger.info(
            f"PTQ operation cannot continue{count}PTQ operation cannot continue"
        )

    def quantize_weights(self) -> nn.Module:
        if not self.is_calibrated:
            raise RuntimeError("PTQ operation cannot continue")
        logger.info("PTQ operation cannot continue")
        count = 0
        for name, module in self.calibrated_model.named_modules():
            if isinstance(module, (FakeQuantizedLinear, FakeQuantizedConv2d)):
                weight_quant = module.weight_fake_quant
                with torch.no_grad():
                    weight_quant.observer.update(module.weight.detach())
                    scale, zero_point = weight_quant.observer.calculate_qparams()
                    weight_quant.scale = scale
                    weight_quant.zero_point = zero_point
                    module.weight.data = weight_quant(module.weight.data)
                    count += 1
        logger.info(
            f"PTQ operation cannot continue{count}PTQ operation cannot continue"
        )
        return self.calibrated_model

    def convert(self) -> nn.Module:
        if not self.is_calibrated:
            raise RuntimeError("PTQ operation cannot continue")
        logger.info("PTQ operation cannot continue")
        self.quantize_weights()
        quantized_model = self._create_quantized_model()
        self.quantized_model = quantized_model
        self.is_converted = True
        logger.info("PTQ operation cannot continue")
        return quantized_model

    def _create_quantized_model(self) -> nn.Module:
        if self.calibrated_model is None:
            raise RuntimeError("PTQ operation cannot continue")
        quantized_model = self._copy_model(self.calibrated_model)
        for name, module in quantized_model.named_modules():
            if isinstance(module, FakeQuantizedLinear):
                quantized_module = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    quant_config=self.config,
                )
                quantized_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    quantized_module.bias.data = module.bias.data.clone()
                if (
                    hasattr(module, "weight_fake_quant")
                    and module.weight_fake_quant.scale is not None
                ):
                    quantized_module.weight_scale = (
                        module.weight_fake_quant.scale.clone()
                    )
                    quantized_module.weight_zero_point = (
                        module.weight_fake_quant.zero_point.clone()
                    )
                parent_name, layer_name = (
                    name.rsplit(".", 1) if "." in name else ("", name)
                )
                if parent_name:
                    parent = quantized_model
                    for p in parent_name.split("."):
                        parent = getattr(parent, p)
                    setattr(parent, layer_name, quantized_module)
                else:
                    setattr(quantized_model, layer_name, quantized_module)
            elif isinstance(module, FakeQuantizedConv2d):
                quantized_module = QuantizedConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    quant_config=self.config,
                )
                quantized_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    quantized_module.bias.data = module.bias.data.clone()
                if (
                    hasattr(module, "weight_fake_quant")
                    and module.weight_fake_quant.scale is not None
                ):
                    quantized_module.weight_scale = (
                        module.weight_fake_quant.scale.clone()
                    )
                    quantized_module.weight_zero_point = (
                        module.weight_fake_quant.zero_point.clone()
                    )
                parent_name, layer_name = (
                    name.rsplit(".", 1) if "." in name else ("", name)
                )
                if parent_name:
                    parent = quantized_model
                    for p in parent_name.split("."):
                        parent = getattr(parent, p)
                    setattr(parent, layer_name, quantized_module)
                else:
                    setattr(quantized_model, layer_name, quantized_module)
        return quantized_model

    def _copy_model(self, model: nn.Module) -> nn.Module:
        import copy

        return copy.deepcopy(model)

    def _insert_fake_quantize(self, model: nn.Module) -> None:
        quantizable_layers = get_quantizable_layers(model)
        for name in quantizable_layers:
            *parent_path, layer_name = name.split(".")
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            original_layer = getattr(parent, layer_name)
            if isinstance(original_layer, nn.Linear):
                quantized_layer = FakeQuantizedLinear(
                    original_layer.in_features,
                    original_layer.out_features,
                    bias=original_layer.bias is not None,
                    config=self.config,
                )
                quantized_layer.weight.data = original_layer.weight.data.clone()
                if original_layer.bias is not None:
                    quantized_layer.bias.data = original_layer.bias.data.clone()
                setattr(parent, layer_name, quantized_layer)
            elif isinstance(original_layer, nn.Conv2d):
                quantized_layer = FakeQuantizedConv2d(
                    original_layer.in_channels,
                    original_layer.out_channels,
                    original_layer.kernel_size,
                    stride=original_layer.stride,
                    padding=original_layer.padding,
                    dilation=original_layer.dilation,
                    groups=original_layer.groups,
                    bias=original_layer.bias is not None,
                    padding_mode=original_layer.padding_mode,
                    config=self.config,
                )
                quantized_layer.weight.data = original_layer.weight.data.clone()
                if original_layer.bias is not None:
                    quantized_layer.bias.data = original_layer.bias.data.clone()
                setattr(parent, layer_name, quantized_layer)

    def get_quantization_params(self) -> Dict[str, Any]:
        if not self.is_calibrated:
            raise RuntimeError("PTQ operation cannot continue")
        params = {}
        for name, module in self.calibrated_model.named_modules():
            if isinstance(module, FakeQuantize):
                scale = module.scale
                zero_point = module.zero_point
                params[name] = {
                    "scale": scale.cpu().numpy().tolist(),
                    "zero_point": zero_point.cpu().numpy().tolist(),
                    "min_val": (
                        module.observer.min_val.cpu().numpy().tolist()
                        if module.observer.min_val is not None
                        else None
                    ),
                    "max_val": (
                        module.observer.max_val.cpu().numpy().tolist()
                        if module.observer.max_val is not None
                        else None
                    ),
                }
        return params

    def export(self, save_path: str) -> None:
        if not self.is_converted:
            raise RuntimeError("PTQ operation cannot continue")
        quant_params = self.get_quantization_params()
        model_to_save = (
            self.quantized_model
            if self.quantized_model is not None
            else self.calibrated_model
        )
        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "quantization_params": quant_params,
            "quantized_layer_params": self._get_quantized_layer_params(model_to_save),
            "config": self.config.to_dict(),
            "is_quantized": True,
            "quantization_type": "ptq",
        }
        torch.save(checkpoint, save_path)
        logger.info(f"PTQ operation cannot continue{save_path}")

    def _get_quantized_layer_params(
        self, model: nn.Module
    ) -> Dict[str, Dict[str, Any]]:
        params: Dict[str, Dict[str, Any]] = {}
        for name, module in model.named_modules():
            if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
                params[name] = {
                    "weight_scale": (
                        module.weight_scale.detach().cpu()
                        if module.weight_scale is not None
                        else None
                    ),
                    "weight_zero_point": (
                        module.weight_zero_point.detach().cpu()
                        if module.weight_zero_point is not None
                        else None
                    ),
                }
        return params

    def evaluate(
        self, test_loader, criterion: nn.Module, device: Optional[torch.device] = None
    ) -> Tuple[float, float]:
        if self.calibrated_model is None:
            raise RuntimeError("PTQ operation cannot continue")
        self.calibrated_model.eval()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, (list, tuple)):
                    inputs, targets = (batch_data[0], batch_data[1])
                else:
                    inputs = batch_data
                    targets = None
                inputs = inputs.to(device)
                outputs = self.calibrated_model(inputs)
                if targets is not None:
                    targets = targets.to(device)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        if targets is not None:
            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
            logger.info(
                f"PTQ operation cannot continue{avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )
            return (avg_loss, accuracy)
        else:
            return (0.0, 0.0)

    def get_quantization_info(self) -> Dict[str, Any]:
        if self.calibrated_model is None:
            return {}
        total_params = 0
        quantized_params = 0
        for name, module in self.calibrated_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                num_params = sum((p.numel() for p in module.parameters()))
                total_params += num_params
                if hasattr(module, "weight_fake_quant"):
                    quantized_params += num_params
        return {
            "total_params": total_params,
            "quantized_params": quantized_params,
            "quantization_ratio": (
                quantized_params / total_params if total_params > 0 else 0
            ),
            "bits": self.config.bits,
            "scheme": self.config.scheme,
            "per_channel": self.config.per_channel,
        }

    def print_quantization_info(self) -> None:
        info = self.get_quantization_info()
        print("=" * 60)
        print("PTQ operation cannot continue")
        print("=" * 60)
        print(f"PTQ operation cannot continue{info['total_params']:,}")
        print(f"PTQ operation cannot continue{info['quantized_params']:,}")
        print(f"PTQ operation cannot continue{info['quantization_ratio'] * 100:.2f}%")
        print(f"PTQ operation cannot continue{info['bits']} bit")
        print(f"PTQ operation cannot continue{info['scheme']}")
        print(f"PTQ operation cannot continue{info['per_channel']}")
        print("=" * 60)


def ptq_quantize(
    model: nn.Module,
    config: QuantizationConfig,
    calib_loader,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> nn.Module:
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    ptq.calibrate(calib_loader, progress_callback)
    return ptq.convert()


def load_quantized_model(
    checkpoint_path: str, model: Optional[nn.Module] = None
) -> Tuple[nn.Module, QuantizationConfig, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = QuantizationConfig.from_dict(checkpoint["config"])
    quant_params = checkpoint["quantization_params"]
    if model is None:
        raise ValueError("PTQ operation cannot continue")
    ptq = PostTrainingQuantization(model, config)
    ptq.prepare()
    loaded_model = ptq._create_quantized_model()
    loaded_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    for name, params in checkpoint.get("quantized_layer_params", {}).items():
        module = loaded_model
        for part in name.split("."):
            if not part:
                continue
            module = getattr(module, part)
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            module.weight_scale = params.get("weight_scale")
            module.weight_zero_point = params.get("weight_zero_point")
    logger.info(f"PTQ operation cannot continue{checkpoint_path}")
    logger.info(
        f"PTQ operation cannot continue{checkpoint.get('quantization_type', 'unknown')}"
    )
    logger.info(f"PTQ operation cannot continue{config.bits} bit")
    return (loaded_model, config, quant_params)
