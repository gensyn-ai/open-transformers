"""Offloaded accumulation must match MPS autograd, including special bits."""

import pytest
import torch

from pretrain.train.mps_gradient_accumulation import accumulate_mps_fp32_gradients


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("kind", ["random", "zeros", "subnormal", "nonfinite"])
@pytest.mark.parametrize("shape", [(257,), (17, 31)])
def test_offloaded_gradients_match_autograd(kind, shape):
    models = [torch.nn.Linear(1, 1, bias=False, device="mps") for _ in range(2)]
    for model in models:
        model.weight = torch.nn.Parameter(torch.zeros(shape, device="mps"))
    count = models[0].weight.numel()
    gen = torch.Generator().manual_seed(761)
    patterns = {
        "zeros": [0, 0x80000000],
        "subnormal": [1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000],
        "nonfinite": [0x7FC12345, 0xFFC54321, 0x7F800000, 0xFF800000, 0],
    }
    accumulator = {}
    for step in range(3):
        if kind == "random":
            grad = torch.randn(shape, generator=gen) * 1e-6
        else:
            bits = torch.tensor(patterns[kind], dtype=torch.int64).to(torch.int32)
            grad = bits.repeat(count // len(bits) + 1)[:count].roll(step)
            grad = grad.view(torch.float32).reshape(shape)
        for model in models:
            model.weight.backward(grad.to("mps"))
        accumulate_mps_fp32_gradients(models[1], accumulator)
        assert models[1].weight.grad is None
        expected = models[0].weight.grad.cpu().contiguous().view(torch.uint8)
        assert torch.equal(expected, accumulator["weight"].view(torch.uint8))
        # A parameter unused in the next microbatch must retain its total.
        accumulate_mps_fp32_gradients(models[1], accumulator)
        assert torch.equal(expected, accumulator["weight"].view(torch.uint8))


def test_offloaded_gradients_reject_cpu_inputs():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(ValueError, match="FP32 MPS"):
        accumulate_mps_fp32_gradients(model, {})
