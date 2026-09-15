"""CPU attention caching is an explicit checkpoint-memory tradeoff."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from pretrain.parallel.parallelize_llama3_repop import _apply_ac


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_attention_cache_requires_opt_in_and_cpu(monkeypatch, enabled, device):
    from torch.distributed.algorithms._checkpoint import \
        checkpoint_wrapper as wrappers

    if enabled:
        monkeypatch.setenv("REPOP_CPU_ATTENTION_CHECKPOINT_CACHE", "1")
    else:
        monkeypatch.delenv("REPOP_CPU_ATTENTION_CHECKPOINT_CACHE", raising=False)
    calls = []
    def context_factory():
        return None

    monkeypatch.setattr(
        import_module("repop.nn.flash_attention"),
        "cpu_attention_checkpoint_contexts",
        context_factory,
        raising=False,
    )

    def wrapper(block, **kwargs):
        calls.append(kwargs)
        return block

    monkeypatch.setattr(wrappers, "checkpoint_wrapper", wrapper)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, device=device)])
    cfg = SimpleNamespace(
        run=SimpleNamespace(activation_checkpoint=True, ac_every_other_block=False)
    )
    _apply_ac(model, cfg)
    assert len(calls) == 1
    assert ("context_fn" in calls[0]) == (enabled and device == "cpu")
    if enabled and device == "cpu":
        assert calls[0]["context_fn"] is context_factory
    assert calls[0]["preserve_rng_state"] is False
