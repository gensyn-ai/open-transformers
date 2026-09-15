"""The native build must preserve the state hash's byte delimiters."""

from types import SimpleNamespace

import pytest
import torch

from pretrain.train import state_hash


@pytest.mark.parametrize("local", [False, True])
def test_optimizer_tags_end_with_nul(monkeypatch, local):
    model = torch.nn.Linear(1, 1, bias=False)
    parameter = model.weight
    optimizer = SimpleNamespace(
        state={
            parameter: {
                "step": 1,
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
        },
        param_groups=[],
    )
    tags = []
    monkeypatch.setattr(
        state_hash,
        "_feed_local" if local else "feed_tensor",
        lambda h, tag, tensor: tags.append(tag),
    )
    if local:
        state_hash.local_shard_state_digest(model, optimizer=optimizer)
    else:
        state_hash.compute_state_hash(model, optimizer=optimizer)
    assert tags == [b"weight\x00", b"optim_exp_avg\x00", b"optim_exp_avg_sq\x00"]
