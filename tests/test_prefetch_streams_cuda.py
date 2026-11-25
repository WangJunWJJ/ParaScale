# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real stream dependency and allocation-lifetime stress checks."""

import pytest

torch = pytest.importorskip("torch")

from parascale.runtime.training.prefetch import CudaStreamPrefetchIterator  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA allocation lifetime requires a GPU"
)


@pytest.mark.parametrize("early_exit", [False, True])
def test_prefetch_output_survives_consumer_delay_and_allocation_churn(early_exit):
    device = torch.device("cuda", torch.cuda.current_device())
    consumer = torch.cuda.Stream(device=device)
    shape = (256, 1024)
    output = torch.empty((2, *shape), device=device)
    torch.cuda.synchronize(device)
    for index in range(12):
        cpu = torch.full(shape, float(index + 1), pin_memory=True)
        iterator = CudaStreamPrefetchIterator(
            torch, iter([{"x": [cpu, (cpu,)]}, {"x": cpu}]), device
        )
        copy_stream = iterator.stream
        with torch.cuda.stream(consumer):
            batch = next(iterator)
            torch.cuda._sleep(20_000_000)
            output[0].copy_(batch["x"][0])
            output[1].copy_(batch["x"][1][0])
        del batch
        if early_exit:
            del iterator
        # Churn the allocation stream while the consumer still has queued reads.
        with torch.cuda.stream(copy_stream):
            scratches = [torch.empty(shape, device=device) for _ in range(16)]
            for scratch in scratches:
                scratch.fill_(-99)
            del scratch, scratches
        consumer.synchronize()
        assert torch.equal(output, torch.full_like(output, float(index + 1)))
        copy_stream.synchronize()


def test_cuda_input_observes_producer_current_stream():
    device = torch.device("cuda", torch.cuda.current_device())
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    value = torch.zeros(256 * 1024, device=device)
    output = torch.empty_like(value)
    torch.cuda.synchronize(device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(20_000_000)
        value.fill_(17)
        iterator = CudaStreamPrefetchIterator(torch, iter([value]), device)
    with torch.cuda.stream(consumer):
        batch = next(iterator)
        output.copy_(batch)
    del batch, iterator, value
    consumer.synchronize()
    assert torch.equal(output, torch.full_like(output, 17))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible GPUs")
def test_cross_device_copy_retains_source_until_transfer_finishes():
    source_device = torch.device("cuda:0")
    target_device = torch.device("cuda:1")
    producer = torch.cuda.Stream(device=source_device)
    consumer = torch.cuda.Stream(device=target_device)
    allocation_stream = torch.cuda.current_stream(source_device)
    value = torch.zeros(256 * 1024, device=source_device)
    output = torch.empty_like(value, device=target_device)
    torch.cuda.synchronize(source_device)
    torch.cuda.synchronize(target_device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(20_000_000)
        value.fill_(23)
        iterator = CudaStreamPrefetchIterator(torch, iter([value]), target_device)
    with torch.cuda.stream(consumer):
        batch = next(iterator)
        output.copy_(batch)
    del value, batch, iterator
    with torch.cuda.stream(allocation_stream):
        scratches = [torch.empty(256 * 1024, device=source_device) for _ in range(16)]
        for scratch in scratches:
            scratch.fill_(-99)
        del scratch, scratches
    consumer.synchronize()
    assert torch.equal(output, torch.full_like(output, 23))
    producer.synchronize()
    allocation_stream.synchronize()
