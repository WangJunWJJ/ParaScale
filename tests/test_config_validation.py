# -*- coding: utf-8 -*-
# @Time    : 2026/3/21
# @Author  : Jun Wang
# @File    : test_config_validation.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
配置跨参数一致性检查测试模块

测试配置参数的跨参数一致性验证功能。
"""

import pytest
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale.config import ParaScaleConfig, QuantizationConfig, ConfigValidationError


class TestQuantizationConfigCrossValidation:
    """测试量化配置跨参数一致性检查"""
    
    def test_observer_type_with_moving_average_ratio_warning(self):
        """测试 observer_type='minmax' 但设置了 moving_average_ratio 时发出警告"""
        with pytest.warns(UserWarning, match="observer_type is 'minmax' but moving_average_ratio is set"):
            config = QuantizationConfig(
                observer_type="minmax",
                moving_average_ratio=0.5  # 非默认值
            )
            assert config.observer_type == "minmax"
    
    def test_ptq_mode_with_qat_epochs_warning(self):
        """测试 PTQ 模式但设置了 qat_epochs 时发出警告"""
        with pytest.warns(UserWarning, match="mode is 'ptq' but qat_epochs"):
            config = QuantizationConfig(
                mode="ptq",
                qat_epochs=20  # 非默认值
            )
            assert config.mode == "ptq"
    
    def test_qat_mode_with_calib_batches_warning(self):
        """测试 QAT 模式但设置了 calib_batches 时发出警告"""
        with pytest.warns(UserWarning, match="mode is 'qat' but calib_batches"):
            config = QuantizationConfig(
                mode="qat",
                calib_batches=50  # 非默认值
            )
            assert config.mode == "qat"
    
    def test_4bit_asymmetric_warning(self):
        """测试 4-bit 非对称量化时发出警告"""
        with pytest.warns(UserWarning, match="4-bit quantization with asymmetric scheme"):
            config = QuantizationConfig(
                bits=4,
                scheme="asymmetric"
            )
            assert config.bits == 4
            assert config.scheme == "asymmetric"
    
    def test_valid_quantization_config_no_warnings(self):
        """测试有效配置不发出警告"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 将警告转为错误
            config = QuantizationConfig(
                enabled=True,
                mode="qat",
                bits=8,
                observer_type="moving_average",
                moving_average_ratio=0.95
            )
            assert config.enabled is True


class TestParaScaleConfigParallelStrategyValidation:
    """测试并行策略配置一致性检查"""
    
    def test_too_many_parallel_strategies_warning(self):
        """测试使用过多并行策略时发出警告"""
        with pytest.warns(UserWarning, match="Using 3 parallel strategies simultaneously"):
            config = ParaScaleConfig(
                data_parallel_size=2,
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                pipeline_parallel_chunks=4  # 必须 >= 2
            )
            assert config.data_parallel_size == 2
    
    def test_pipeline_without_enough_chunks_error(self):
        """测试流水线并行但 chunks < 2 时报错"""
        with pytest.raises(ConfigValidationError, match="pipeline_parallel_chunks must be >= 2"):
            ParaScaleConfig(
                pipeline_parallel_size=2,
                pipeline_parallel_chunks=1
            )
    
    def test_invalid_tensor_parallel_mode_error(self):
        """测试无效的张量并行模式时报错"""
        with pytest.raises(ConfigValidationError, match="tensor_parallel_mode must be 'row' or 'column'"):
            ParaScaleConfig(
                tensor_parallel_size=2,
                tensor_parallel_mode="invalid"
            )


class TestParaScaleConfigZeROValidation:
    """测试 ZeRO 配置一致性检查"""
    
    def test_zero_stage_without_optimization_warning(self):
        """测试设置了 zero_stage 但未启用 zero_optimization 时发出警告"""
        with pytest.warns(UserWarning, match="zero_stage=1 but zero_optimization=False"):
            config = ParaScaleConfig(
                zero_stage=1,
                zero_optimization=False
            )
            assert config.zero_stage == 1
    
    def test_zero_optimization_with_stage_zero_warning(self):
        """测试启用了 zero_optimization 但 stage=0 时发出警告"""
        with pytest.warns(UserWarning, match="zero_optimization=True but zero_stage=0"):
            config = ParaScaleConfig(
                zero_optimization=True,
                zero_stage=0
            )
            assert config.zero_optimization is True
    
    def test_zero_offload_without_zero_error(self):
        """测试启用 zero_offload 但未启用 ZeRO 时报错"""
        with pytest.raises(ConfigValidationError, match="zero_offload=True requires zero_stage >= 1"):
            ParaScaleConfig(
                zero_offload=True,
                zero_stage=0
            )
    
    def test_zero_stage2_without_data_parallel_warning(self):
        """测试 Stage 2+ 但没有数据并行时发出警告"""
        with pytest.warns(UserWarning, match="Using ZeRO Stage 2 with data_parallel_size=1"):
            config = ParaScaleConfig(
                zero_stage=2,
                data_parallel_size=1
            )
            assert config.zero_stage == 2


class TestParaScaleConfigQuantizationValidation:
    """测试量化配置一致性检查"""
    
    def test_quantization_with_tensor_parallel_warning(self):
        """测试量化与张量并行同时使用时发出警告"""
        with pytest.warns(UserWarning, match="Quantization with tensor parallelism may have precision issues"):
            config = ParaScaleConfig(
                tensor_parallel_size=2,
                quantization=QuantizationConfig(enabled=True)
            )
            assert config.quantization.enabled is True
    
    def test_qat_with_short_checkpoint_interval_warning(self):
        """测试 QAT 但检查点间隔太短时发出警告"""
        with pytest.warns(UserWarning, match="checkpoint_save_interval .* is less than qat_epochs"):
            config = ParaScaleConfig(
                checkpoint_save_interval=5,
                quantization=QuantizationConfig(
                    enabled=True,
                    mode="qat",
                    qat_epochs=10
                )
            )
            assert config.quantization.mode == "qat"


class TestParaScaleConfigTrainingValidation:
    """测试训练配置一致性检查"""
    
    def test_large_batch_without_accumulation_warning(self):
        """测试大批次但没有梯度累积时发出警告"""
        with pytest.warns(UserWarning, match="Large effective batch size .* without gradient accumulation"):
            config = ParaScaleConfig(
                batch_size=2048,  # 需要 > 1024 才能触发警告
                gradient_accumulation_steps=1
            )
            assert config.batch_size == 2048
    
    def test_high_lr_with_large_batch_warning(self):
        """测试高学习率与大批次同时使用时发出警告"""
        with pytest.warns(UserWarning, match="High learning rate .* with large batch size"):
            config = ParaScaleConfig(
                learning_rate=0.1,
                batch_size=256
            )
            assert config.learning_rate == 0.1


class TestConfigValidationReport:
    """测试配置验证报告功能"""
    
    def test_validation_report_structure(self):
        """测试验证报告结构"""
        config = ParaScaleConfig()
        report = config.get_validation_report()
        
        assert "valid" in report
        assert "warnings" in report
        assert "suggestions" in report
        assert "config_summary" in report
        assert report["valid"] is True
    
    def test_validation_report_with_dp_tp(self):
        """测试 DP+TP 配置的建议"""
        config = ParaScaleConfig(
            data_parallel_size=2,
            tensor_parallel_size=2
        )
        report = config.get_validation_report()
        
        # 应该建议使用 3D 并行
        assert any("3D parallelism" in s for s in report["suggestions"])
    
    def test_validation_report_suggests_zero(self):
        """测试大 DP 时建议使用 ZeRO"""
        config = ParaScaleConfig(
            data_parallel_size=4,
            zero_optimization=False
        )
        report = config.get_validation_report()
        
        # 应该建议启用 ZeRO
        assert any("ZeRO optimization" in s for s in report["suggestions"])
    
    def test_validation_report_suggests_quantization(self):
        """测试 MP 时建议使用量化"""
        config = ParaScaleConfig(
            model_parallel_size=2,
            quantization=QuantizationConfig(enabled=False)
        )
        report = config.get_validation_report()
        
        # 应该建议启用量化
        assert any("quantization" in s for s in report["suggestions"])
    
    def test_config_summary(self):
        """测试配置摘要"""
        config = ParaScaleConfig(
            data_parallel_size=2,
            tensor_parallel_size=2,
            zero_optimization=True,
            zero_stage=1
        )
        report = config.get_validation_report()
        
        assert report["config_summary"]["parallel_strategy"] == "DP=2 + TP=2"
        assert "ZeRO Stage 1" in report["config_summary"]["memory_optimization"]


class TestConfigUpdateValidation:
    """测试配置更新时的验证"""
    
    def test_update_triggers_validation(self):
        """测试更新配置时触发验证"""
        config = ParaScaleConfig()
        
        # 更新为无效配置应该报错
        with pytest.raises(ConfigValidationError):
            config.update({"pipeline_parallel_size": 2, "pipeline_parallel_chunks": 1})
    
    def test_update_with_valid_config(self):
        """测试有效配置更新"""
        config = ParaScaleConfig()
        
        # 更新为有效配置
        config.update({
            "batch_size": 64,
            "learning_rate": 1e-4
        })
        
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4


class TestConfigFromDict:
    """测试从字典创建配置的验证"""
    
    def test_from_dict_with_invalid_config(self):
        """测试从无效字典创建配置时报错"""
        with pytest.raises(ConfigValidationError):
            ParaScaleConfig.from_dict({
                "zero_offload": True,
                "zero_stage": 0
            })
    
    def test_from_dict_with_quantization(self):
        """测试从包含量化配置的字典创建"""
        config = ParaScaleConfig.from_dict({
            "quantization": {
                "enabled": True,
                "bits": 8
            }
        })
        
        assert config.quantization.enabled is True
        assert config.quantization.bits == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
