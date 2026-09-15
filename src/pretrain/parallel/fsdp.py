"""DDP wrapping path.

The FSDP / TP / AC / compile pipeline lives in
``pretrain.parallel.parallelize_llama3_repop``; this module only handles
the (less interesting) ``cfg.run.parallel == "ddp"`` branch: full model
replicated, gradients all-reduced. Viable only when params + grads +
optim state + activations fit on a single device.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from pretrain.config.schema import RootConfig
from pretrain.model.llama3 import Llama3

LOG = logging.getLogger(__name__)


def wrap_model_ddp(model: Llama3, cfg: RootConfig) -> nn.Module:
    """Apply AC + per-block compile + DDP. On CPU / world_size == 1
    returns the model with only AC/compile applied (the DDP wrap would
    be a no-op).
    """
    if not torch.cuda.is_available():
        LOG.warning("CUDA unavailable — running unwrapped (single-process dev mode)")
        return _maybe_apply_ac(model, cfg)

    _maybe_apply_ac(model, cfg)

    ws = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    if ws <= 1:
        LOG.info("world_size=1 — skipping DDP wrap")
        return model

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        gradient_as_bucket_view=True,
    )
    LOG.info(
        "DDP wrap applied (local_rank=%d, ws=%d); ac=%s",
        local_rank, ws, cfg.run.activation_checkpoint,
    )
    return ddp_model


def _maybe_apply_ac(model: Llama3, cfg: RootConfig) -> Llama3:
    if not cfg.run.activation_checkpoint:
        return model
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    every_other = cfg.run.ac_every_other_block
    for i, block in enumerate(model.blocks):
        if every_other and i % 2 == 1:
            continue
        model.blocks[i] = checkpoint_wrapper(block, preserve_rng_state=False)
    return model


