"""Build the training DataLoader.

We deliberately don't use ``torch.utils.data.IterableDataset``'s worker
sharding — our :class:`MixSampler` is already rank-aware, and adding
worker-level sharding would split a rank's stream non-deterministically
across workers. Instead we use ``num_workers=0`` for the sampler and pin
memory at the collate boundary. The bottleneck on H100 is GPU compute,
not data prep, because tokenisation happens at prep time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from pretrain.config.schema import DataConfig
from pretrain.data.manifest import SourceManifest
from pretrain.data.mix_sampler import MixSampler, MixSamplerState


class _MixDataset(IterableDataset):
    def __init__(self, sampler: MixSampler, micro_batch_size: int) -> None:
        super().__init__()
        self._sampler = sampler
        self._mb = micro_batch_size

    def __iter__(self):
        it = iter(self._sampler)
        while True:
            chunks = [next(it) for _ in range(self._mb)]
            yield np.stack(chunks, axis=0)        # [mb, seq_len+1]


def _collate(batch: list[np.ndarray]) -> dict[str, torch.Tensor]:
    # Each ``batch`` element is already a [mb, seq_len+1] ndarray since the
    # IterableDataset yields a full micro-batch.
    arr = batch[0]
    seq = torch.from_numpy(arr).long()
    return {
        "input_ids": seq[:, :-1].contiguous(),
        "labels": seq[:, 1:].contiguous(),
    }


def _resolve_sources(
    cfg: DataConfig,
) -> tuple[list[SourceManifest], list[str], list[float]]:
    """Load manifests and resolve per-document sampling weights from a config.

    Shared by the legacy per-rank :func:`build_loader` and the canonical
    :func:`build_global_loader` so both speak the same weights.
    """
    manifests: list[SourceManifest] = []
    manifest_dirs: list[str] = []
    weights: list[float] = []
    for src in cfg.sources:
        # The manifest is at ``<path>/manifest.yaml`` and shards live alongside.
        manifest_path = Path(src.path) / "manifest.yaml"
        manifest = SourceManifest.load(manifest_path)
        manifests.append(manifest)
        manifest_dirs.append(str(Path(src.path)))
        weights.append(src.weight)

    if cfg.weights_are_token_shares:
        # Convert token-share targets to per-document sampling probabilities.
        # Derivation: each draw picks source i with prob p_i and contributes
        # ~L_i tokens (L_i = manifest.total_tokens / total_documents). The
        # expected per-source token share is τ_i = p_i·L_i / Σ p_j·L_j, so
        # to achieve a target τ_i we set p_i ∝ τ_i / L_i, then normalize.
        # The sampler normalizes again internally; we still normalize here
        # so the values logged/inspected are interpretable.
        avg_tok_per_doc = [
            m.total_tokens / max(m.total_documents, 1) for m in manifests
        ]
        if any(l <= 0 for l in avg_tok_per_doc):
            bad = [m.name for m, l in zip(manifests, avg_tok_per_doc) if l <= 0]
            raise ValueError(
                f"cannot convert token-share weights: source(s) {bad} have "
                "zero tokens or zero documents in their manifest"
            )
        doc_weights = [w / l for w, l in zip(weights, avg_tok_per_doc)]
        total = sum(doc_weights)
        weights = [w / total for w in doc_weights]

    return manifests, manifest_dirs, weights


def build_loader(
    cfg: DataConfig,
    *,
    micro_batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    eos_id: int,
    sampler_state: MixSamplerState | None = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, MixSampler]:
    """Returns ``(DataLoader, MixSampler)``. The sampler is also returned so
    the train loop can read ``sampler.state()`` for checkpointing.
    """
    manifests, manifest_dirs, weights = _resolve_sources(cfg)

    sampler = MixSampler(
        manifests=manifests,
        manifest_dirs=manifest_dirs,
        weights=weights,
        seq_len=cfg.seq_len,
        seed=seed,
        rank=rank,
        world_size=world_size,
        eos_id=eos_id,
        state=sampler_state,
    )
    dataset = _MixDataset(sampler, micro_batch_size=micro_batch_size)
    loader = DataLoader(
        dataset,
        batch_size=None,                # micro-batch is already inside _MixDataset
        num_workers=0,                  # see module docstring
        collate_fn=lambda b: _collate([b]),
        pin_memory=pin_memory,
    )
    return loader, sampler


class _ViewDataset(IterableDataset):
    """Wraps a :class:`ShardedWindowView`; yields one micro-batch per step."""

    def __init__(self, view) -> None:
        super().__init__()
        self._view = view

    def __iter__(self):
        yield from iter(self._view)


def build_global_loader(
    cfg: DataConfig,
    train,
    *,
    rank: int,
    world_size: int,
    seed: int,
    eos_id: int,
    state=None,
    start_consumed_tokens: int = 0,
    start_step: int = 0,
    pin_memory: bool = True,
):
    """Canonical, topology-independent loader: one rank's slice of the global stream.

    Returns ``(DataLoader, ShardedWindowView)``. The view is also returned so the
    train loop can read ``view.state()`` (the *global* stream position, identical
    on every rank) for the single ``global_stream.json`` checkpoint, and call
    ``view.accum_for_step(M)`` to size each step's micro-batch loop.

    ``rank``/``world_size`` here are the DP-unique-token degree (``dp_rank`` /
    ``dp_world_size``), matching the legacy :func:`build_loader` folding.
    """
    from pretrain.data.global_stream import ShardedWindowView

    manifests, manifest_dirs, weights = _resolve_sources(cfg)
    view = ShardedWindowView(
        manifests,
        manifest_dirs,
        weights,
        train,
        seed=seed,
        rank=rank,
        world_size=world_size,
        micro_batch_size=train.micro_batch_size,
        eos_id=eos_id,
        state=state,
        start_consumed_tokens=start_consumed_tokens,
        start_step=start_step,
    )
    loader = DataLoader(
        _ViewDataset(view),
        batch_size=None,
        num_workers=0,
        collate_fn=lambda b: _collate([b]),
        pin_memory=pin_memory,
    )
    return loader, view
