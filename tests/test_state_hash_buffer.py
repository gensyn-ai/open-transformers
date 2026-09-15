"""Buffer ownership and compatibility with the original hash byte stream."""

import gc
import hashlib

import pytest
import torch

from pretrain.train import state_hash


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64,
     torch.int32, torch.int8, torch.uint8, torch.bool, torch.complex64],
)
@pytest.mark.parametrize("layout", ["scalar", "empty", "contiguous", "transpose"])
def test_buffer_matches_bytes_and_retains_storage(dtype, layout):
    tensor = torch.arange(12).to(dtype).reshape(3, 4)
    if layout == "scalar":
        tensor = tensor[0, 0]
    elif layout == "empty":
        tensor = tensor[:0]
    elif layout == "transpose":
        tensor = tensor.t()
    expected = bytes(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy())
    buffer = state_hash.tensor_bytes(tensor)
    assert isinstance(buffer, memoryview)
    if layout == "contiguous":
        assert buffer.obj.__array_interface__["data"][0] == tensor.data_ptr()
    del tensor
    gc.collect()
    assert bytes(buffer) == expected
    assert hashlib.blake2b(buffer).digest() == hashlib.blake2b(expected).digest()


def test_special_float_bits_preserved():
    tensor = torch.tensor([0, -2147483648, 2139095040, -8388608, 2143289345], dtype=torch.int32)
    assert bytes(state_hash.tensor_bytes(tensor.view(torch.float32))) == bytes(tensor.view(torch.uint8).numpy())


def test_requires_grad_buffer():
    tensor = torch.tensor([1.0, -0.0], requires_grad=True)
    buffer = state_hash.tensor_bytes(tensor)
    assert buffer.obj.__array_interface__["data"][0] == tensor.data_ptr()
    assert bytes(buffer) == bytes(tensor.detach().view(torch.uint8).numpy())


@pytest.mark.parametrize("shards,ranks", [(1, 1), (2, 2), (4, 4), (1, 2), (2, 4)])
def test_full_and_sharded_hashes_match_copy_implementation(monkeypatch, shards, ranks):
    torch.manual_seed(42)
    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.randn(3, 8)).sum().backward()
    optimizer.step()

    def hashes():
        return (
            state_hash.compute_state_hash(model, optimizer=optimizer, include_grads=True),
            state_hash.audit_shard_state_digest(
                model, optimizer=optimizer, include_grads=True,
                num_shards=shards, num_ranks=ranks,
            ),
        )

    candidate = hashes()
    monkeypatch.setattr(state_hash, "tensor_bytes", lambda tensor: bytes(
        tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()))
    assert hashes() == candidate
