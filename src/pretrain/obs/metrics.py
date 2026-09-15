"""Cheap, always-on metrics + an MFU helper.

Metric values are also written to a local JSONL so we can reconstruct a
W&B view post-hoc if W&B has an outage (plan/08 §7).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


def tokens_per_sec(tokens: int, elapsed_s: float) -> float:
    return tokens / max(elapsed_s, 1e-9)


def transformer_flops_per_token(
    n_layers: int,
    d_model: int,
    seq_len: int,
    vocab_size: int,
) -> float:
    """Forward+backward model FLOPs per token — the MFU numerator.

    Three terms, each already ×3 over the forward pass (fwd + 2× bwd):

      * dense blocks  ``72 · L · d²`` — attention QKV+out-proj and the MLP;
        equals ``6 · (12 · L · d²)``, i.e. the classic 6N over the dense params.
      * attention     ``12 · L · s · d`` — the score (QKᵀ) and context (·V)
        matmuls. This is the O(seq) per-token term the bare 6N drops; it grows
        with ``seq_len`` and is non-negligible at our 4k context.
      * LM head        ``6 · V · d`` — the unembedding projection to vocab.

    The input-embedding lookup is ~0 FLOPs. Activation recomputation is **not**
    counted: this is *model* FLOPs (MFU), not *hardware* FLOPs (HFU), so a run
    with activation checkpointing still reports against the same numerator.
    """
    dense = 72.0 * n_layers * d_model**2
    attention = 12.0 * n_layers * seq_len * d_model
    lm_head = 6.0 * vocab_size * d_model
    return dense + attention + lm_head


def mfu(
    flops_per_token: float,
    tokens_per_second: float,
    n_gpus: int,
    peak_flops_per_gpu: float = 1e15,    # H100 bf16 peak ~ 1 PFLOP/s
) -> float:
    """PaLM-style MFU = flops_per_token · tokens_per_sec / (n_gpus · peak).

    ``n_gpus`` is ALL accelerators (the full world size, **including** tensor-
    parallel ranks). TP splits each token's matmuls across its ranks, so those
    GPUs do consume the model FLOPs and belong in the denominator. The paired
    ``tokens_per_second`` is the *unique*-token rate (TP-paired ranks process
    the same tokens, so it excludes the TP factor). Numerator and denominator
    treating TP differently is correct — the work is full-model FLOPs over
    unique tokens, spread across every chip — not a mismatch.
    """
    return flops_per_token * tokens_per_second / (n_gpus * peak_flops_per_gpu)


class MetricsLogger:
    """Writes metrics to W&B (if configured) and to a local JSONL.

    Designed to be cheap on the hot path — JSON line per ``log()`` call,
    flush every 100 lines. ``log()`` accepts arbitrary scalar key→value
    metadata; non-numeric values are coerced to ``str``.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        wandb_run: Any | None = None,
        flush_every: int = 100,
    ) -> None:
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "at", encoding="utf-8")
        self._wandb = wandb_run
        self._flush_every = flush_every
        self._counter = 0

    def log(self, step: int, **kv: Any) -> None:
        line = {"step": step, "time": time.time(), **kv}
        # JSON cannot serialise tensors / ndarrays — coerce.
        for k, v in list(line.items()):
            if hasattr(v, "item"):
                try:
                    line[k] = v.item()
                except Exception:
                    line[k] = str(v)
        self._fh.write(json.dumps(line) + "\n")
        self._counter += 1
        if self._counter % self._flush_every == 0:
            self._fh.flush()
        if self._wandb is not None:
            try:
                self._wandb.log(line, step=step)
            except Exception as e:
                LOG.warning("wandb.log failed: %s", e)

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


@contextmanager
def time_section(label: str, sink: dict[str, float] | None = None):
    """Light-weight timer; writes ``label_ms`` into ``sink`` if provided."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        if sink is not None:
            sink[f"{label}_ms"] = dt
