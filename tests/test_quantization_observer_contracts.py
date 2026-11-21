# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午11:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Observer range reduction and invalid-input contracts."""

import pytest

torch = pytest.importorskip("torch")

from parascale.quantization import QuantizationConfig  # noqa: E402
from parascale.quantization.observers import MinMaxObserver  # noqa: E402


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("shape", [(7,), (8, 12), (4, 3, 5, 5)])
def test_observer_uses_one_range_reduction_with_identical_values(
    per_channel, shape, device
):
    torch.manual_seed(42)
    values = torch.randn(shape, requires_grad=True, device=device)
    observer = MinMaxObserver(QuantizationConfig(per_channel=per_channel)).to(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as trace:
        observer.update(values)
    counts = {event.key: event.count for event in trace.key_averages()}
    assert counts.get("aten::aminmax", 0) == 1
    assert counts.get("aten::amin", 0) == counts.get("aten::amax", 0) == 0
    dims = tuple(range(1, values.ndim)) if per_channel and values.ndim > 1 else None
    expected_min = values.detach().amin(dims)
    expected_max = values.detach().amax(dims)
    if dims is None:
        expected_min, expected_max = expected_min.reshape(1), expected_max.reshape(1)
    torch.testing.assert_close(observer.min_val, expected_min, rtol=0, atol=0)
    torch.testing.assert_close(observer.max_val, expected_max, rtol=0, atol=0)
    assert not observer.min_val.requires_grad


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("per_channel", [False, True])
def test_nonfinite_observation_does_not_change_existing_ranges(
    value, per_channel, device
):
    observer = MinMaxObserver(QuantizationConfig(per_channel=per_channel)).to(device)
    observer.update(torch.ones(3, 4, device=device))
    before = {k: v.clone() for k, v in observer.state_dict().items()}
    values = torch.ones(3, 4, device=device)
    values[1, 2] = value
    with pytest.raises(ValueError, match="nonempty and finite"):
        observer.update(values)
    for key, expected in before.items():
        torch.testing.assert_close(observer.state_dict()[key], expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_noncontiguous_observation_and_empty_input(dtype, device):
    values = (
        torch.arange(120, dtype=dtype, device=device).reshape(3, 4, 10).transpose(1, 2)
    )
    observer = MinMaxObserver(QuantizationConfig(per_channel=True)).to(device)
    observer.update(values)
    torch.testing.assert_close(observer.min_val, values.float().amin((1, 2)))
    torch.testing.assert_close(observer.max_val, values.float().amax((1, 2)))
    with pytest.raises(ValueError, match="nonempty and finite"):
        observer.update(torch.empty(3, 0, device=device))


@pytest.mark.parametrize("imaginary", [float("nan"), float("inf")])
def test_nonfinite_imaginary_values_cannot_mutate_observer(imaginary, device):
    observer = MinMaxObserver(QuantizationConfig(per_channel=False)).to(device)
    observer.update(torch.ones(1, device=device))
    with pytest.raises(ValueError, match="nonempty and finite"):
        observer.update(torch.tensor([complex(2, imaginary)], device=device))
    torch.testing.assert_close(observer.min_val, torch.ones(1, device=device))
    torch.testing.assert_close(observer.max_val, torch.ones(1, device=device))
