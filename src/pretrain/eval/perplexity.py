"""Held-out perplexity (plan/07 §3).

The held-out set is built during data prep — same indexed-binary format
as training shards, hash recorded in the data manifest, and explicitly
absent from training mixes. We compute PPL by streaming the held-out
tokens through the (eval) model in fixed-length chunks.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pretrain.data.indexed_dataset import IndexedDatasetReader


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    held_out_prefix: str | Path,
    seq_len: int,
    device: str | torch.device = "cuda",
    micro_batch_size: int = 1,
) -> float:
    """Stream documents, concatenate with EOS, sliding-window evaluate.

    The simplest correct implementation: pack tokens contiguously (with
    EOS separators), chunk into ``seq_len + 1`` windows, average loss.

    For a serious eval we'd shard across ranks; this version runs on a
    single rank. The training loop's eval cadence path is async on
    rank-0 (plan/07 §2), which is enough for the run.
    """
    reader = IndexedDatasetReader(held_out_prefix)
    eos = 0   # set by the manifest in the real path
    buffer: list[int] = []
    total_loss = 0.0
    total_count = 0

    model.eval()
    chunks_per_micro: list[torch.Tensor] = []
    for doc in reader:
        buffer.extend(int(t) for t in doc.tolist())
        buffer.append(eos)
        while len(buffer) >= seq_len + 1:
            window = torch.tensor(buffer[: seq_len + 1], dtype=torch.long)
            buffer = buffer[seq_len:]
            chunks_per_micro.append(window)
            if len(chunks_per_micro) == micro_batch_size:
                total_loss, total_count = _flush(
                    chunks_per_micro, model, device, total_loss, total_count
                )
                chunks_per_micro = []

    if chunks_per_micro:
        total_loss, total_count = _flush(
            chunks_per_micro, model, device, total_loss, total_count
        )

    if total_count == 0:
        return float("nan")
    avg_nll = total_loss / total_count
    return math.exp(avg_nll)


def _flush(
    chunks: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device | str,
    total_loss: float,
    total_count: int,
) -> tuple[float, int]:
    batch = torch.stack(chunks, dim=0).to(device)
    inp = batch[:, :-1].contiguous()
    lab = batch[:, 1:].contiguous()
    out = model(inp)
    logits = out.logits if hasattr(out, "logits") else out
    loss = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        lab.view(-1),
        reduction="sum",
    )
    n = lab.numel()
    return total_loss + float(loss.item()), total_count + n
