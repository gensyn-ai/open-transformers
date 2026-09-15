"""Parallelize a Llama 3 (repop runtime) model:

    1. Activation Checkpointing (AC) per ``TransformerBlock``.
    2. FSDP2 ``fully_shard`` (HSDP if ``dp_replicate > 1``).

Tensor parallel was removed — we run ``tp=1`` (FSDP / HSDP) only.

``torch.compile`` is intentionally not applied: repop's custom autograd
CFunctions are opaque to Dynamo, so per-block compile produces graph
fragments around the kernel boundaries that don't fuse usefully.

Order matters:
  - AC first, so the checkpoint wrapper becomes the unit FSDP sees.
  - FSDP last; ``fully_shard`` wraps the (potentially AC) block and
    walks through it to find params.
"""

from __future__ import annotations

import logging
import os

import torch

from pretrain.config.schema import RootConfig
from pretrain.model.llama3 import Llama3
from pretrain.model.precision import (
    DEFAULT_POLICY,
    FP32_POLICY,
    MixedPrecisionPolicyFactory,
)
from pretrain.parallel.parallel_dims import ParallelDims

LOG = logging.getLogger(__name__)


def parallelize_llama3_repop(
    model: Llama3, cfg: RootConfig, parallel_dims: ParallelDims
) -> Llama3:
    """Apply AC -> FSDP to ``model`` in place.

    On CPU outside a distributed context, only activation checkpointing is
    applied (FSDP needs a process group, and ``fully_shard`` is a
    no-op at world_size=1) — single-process dev mode (the test suite,
    interactive use, and the single-device audit). AC is applied even here
    because it needs neither a PG nor CUDA and it renames params with the
    ``_checkpoint_wrapped_module.`` prefix; keeping that consistent with the
    production path is what lets a single-device audit's state hash (which
    includes param FQNs) match the cluster's even on a CUDA-less box. When a
    distributed process group is initialised (e.g. multi-rank CPU+gloo used for
    the verification harness, or the production CUDA+NCCL path) the full pipeline
    runs regardless of CUDA availability.
    """
    is_distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if not is_distributed and not torch.cuda.is_available():
        # Single device mode: skip FSDP (it requires a PG; fully_shard is a no-op
        # at ws=1 regardless), but STILL apply activation checkpointing so the
        # param FQNs (``_checkpoint_wrapped_module.`` prefix) match what the
        # production/PG path produces — otherwise a single-device audit's state
        # hash diverges from the cluster's purely on parameter names, even
        # though every weight value is byte-identical.
        LOG.warning(
            "not distributed + no CUDA — parallelize applies AC only "
            "(FSDP skipped; dev mode)"
        )
        _apply_ac(model, cfg)
        return model

    # int8 QAT under FSDP2/HSDP:
    #
    #   * LSQ (LSQQuantizedLinear) IS safe. fully_shard shards on dim-0
    #     (out_features); the learnable weight_scale is [out], so weight and
    #     scale shard on the same axis and stay consistent. FSDP all-gathers the
    #     full param before forward, so the LSQ matmul runs on the full
    #     [out, in] weight + full [out] scale — and the only reduction (the scale
    #     gradient, sum2d_dim1) is over dim-1 (K), which FSDP does not shard. So
    #     FSDP cannot corrupt the per-channel scale.
    #
    #   * absmax QuantizedLinear under ws>1 is out of scope (not yet validated).
    #
    # BFR note: under FSDP, grads are reduce-scattered, so byte-equality holds
    # at a *fixed* (world_size, mesh, NCCL config), not across device counts.
    ws = (
        torch.distributed.get_world_size() if is_distributed else 1
    )
    if cfg.model.qat.enabled and cfg.model.qat.method != "lsq" and ws > 1:
        raise NotImplementedError(
            f"int8 QAT method={cfg.model.qat.method!r} is single-device "
            f"only (world_size={ws}); per-channel absmax is computed over "
            "the sharded output axis and is not yet validated under FSDP2. "
            "Only method='lsq' is supported at ws>1. Run with "
            "nproc_per_node=1, or set model.qat.method=lsq."
        )

    _apply_ac(model, cfg)
    _apply_fsdp(model, cfg, parallel_dims)

    return model


# ---------------------------------------------------------------------------
# Activation Checkpointing.
# ---------------------------------------------------------------------------
def _apply_ac(model: Llama3, cfg: RootConfig) -> None:
    """Wrap each transformer block with the FSDP/DTensor-aware checkpoint
    wrapper so backward re-materialises attn+ffn outputs."""
    if not cfg.run.activation_checkpoint:
        return
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    every_other = cfg.run.ac_every_other_block
    checkpoint_kwargs = {}
    if os.environ.get("REPOP_CPU_ATTENTION_CHECKPOINT_CACHE") == "1" and all(
        p.device.type == "cpu" for p in model.parameters()
    ):
        from repop.nn.flash_attention import cpu_attention_checkpoint_contexts

        checkpoint_kwargs["context_fn"] = cpu_attention_checkpoint_contexts
        LOG.info("CPU activation checkpointing retains native attention results")
    for i, block in enumerate(model.blocks):
        if every_other and i % 2 == 1:
            continue
        model.blocks[i] = checkpoint_wrapper(
            block, preserve_rng_state=False, **checkpoint_kwargs
        )


# ---------------------------------------------------------------------------
# FSDP2 / HSDP.
# ---------------------------------------------------------------------------
def _apply_fsdp(
    model: Llama3, cfg: RootConfig, parallel_dims: ParallelDims
) -> None:
    """Apply ``fully_shard`` to the embedding sub-modules and each block.

    Under HSDP (``dp_replicate > 1``) we pass the 2D ``("dp_replicate",
    "fsdp")`` sub-mesh so FSDP2 does an inner-shard / outer-replicate
    composition. Under pure FSDP we pass the 1D ``"fsdp"`` mesh.
    """
    ws = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    import os as _os
    if ws <= 1 and _os.environ.get("REPOP_FORCE_FSDP_WS1", "0") != "1":
        # No replication work to do. AC/compile already applied.
        # REPOP_FORCE_FSDP_WS1=1 forces the fully_shard wrap even at ws=1 (a
        # 1-rank PG) so a single-process run still gets the bf16 param cast —
        # used to isolate the per-rank bf16 emulation from the multi-rank fold.
        return

    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        # NGC pytorch:25.01 ships a pre-release torch 2.6 (2.6.0a0+...)
        # snapshotted before fully_shard's public re-export landed; the
        # function still exists at its old _composable home. Drop this
        # branch once we're back on a container with a release torch 2.6+.
        try:
            from torch.distributed._composable.fsdp import fully_shard
        except ImportError:  # pragma: no cover
            LOG.warning("torch.distributed.fsdp.fully_shard not available; skipping FSDP wrap")
            return

    deterministic = getattr(cfg.run, "reduction_mode", "nccl") == "deterministic_allgather"
    if parallel_dims.dp_replicate_enabled and not deterministic:
        # Native HSDP: 2D mesh, NCCL does the cross-replica all-reduce.
        dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
    else:
        # Pure FSDP, or deterministic HSDP. For deterministic HSDP we wrap with
        # the 1D *shard* mesh and drive the cross-replica reduction through a
        # deterministic all-reduce hook below — native HSDP's NCCL all-reduce
        # isn't reproducible on a single device for >2 replicas.
        dp_mesh = parallel_dims.get_mesh("fsdp")

    policy = DEFAULT_POLICY if cfg.run.mixed_precision else FP32_POLICY
    # int8 QAT now runs under bf16 param_dtype (DEFAULT_POLICY) like every other
    # module. FSDP2 all-gathers the weight (and the [out] weight_scale) as bf16
    # for the forward while keeping the *sharded master fp32* (so the optimizer
    # update keeps full precision); the LSQ Function upcasts the scale to fp32
    # before the fused weight-quant kernel (repop.qat.lsq), which accepts a bf16
    # weight directly. This halves the weight all-gather traffic. The single-
    # device audit reproduces the same bf16 param cast at ws=1 (see
    # pretrain.cli.audit_replay), so BFR byte-equality against the cluster
    # checkpoint still holds at a fixed (world_size, mesh, NCCL config).
    mp_policy = MixedPrecisionPolicyFactory.build_fsdp_policy(policy)

    # Wrap the leaves of UntiedEmbedding for the same reason as the
    # pre-TP code: ``model.embedding.__call__`` is never invoked
    # (we call ``encode`` / ``project`` directly), so wrapping the
    # leaves is what causes the gather/reshard hooks to fire.
    fully_shard(model.embedding.tok_embeddings, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model.embedding.output, mesh=dp_mesh, mp_policy=mp_policy)
    for block in model.blocks:
        fully_shard(block, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

    # Cross-topology-auditable runs: replace FSDP2's NCCL reduce-scatter (whose
    # summation order is world-size-dependent) with a fixed ascending-rank one so
    # the gradient — and thus every checkpoint — is bitwise-identical regardless
    # of device count, and reproducible by the single-device audit.
    if deterministic:
        from pretrain.parallel.deterministic_reduce import (
            apply_deterministic_reduce_scatter,
            apply_deterministic_replicate_all_reduce,
        )

        wrapped = [
            model.embedding.tok_embeddings,
            model.embedding.output,
            *model.blocks,
            model,
        ]
        n = apply_deterministic_reduce_scatter(wrapped)
        LOG.info("deterministic reduce-scatter installed on %d FSDP modules", n)
        if parallel_dims.dp_replicate_enabled:
            # The shard reduce-scatter above covers the fsdp dim; this adds the
            # fixed-order cross-replica reduction over the dp_replicate group so
            # HSDP is fully topology-invariant for any replicate count.
            rep_group = parallel_dims.get_mesh("dp_replicate").get_group()
            hook = apply_deterministic_replicate_all_reduce(wrapped, rep_group)
            # Stash on the root module so the training loop can drain per-step
            # timing (PRETRAIN_CUDA_EVENT_TIMING) into W&B. Plain attribute (the
            # hook isn't a Module/Tensor) → not registered as a submodule.
            model._replicate_reduce_hook = hook
            LOG.info(
                "deterministic replicate all-reduce installed on %d modules "
                "(dp_replicate=%d)", hook.n_installed, parallel_dims.dp_replicate,
            )

    LOG.info(
        "FSDP2 wrap applied (ws=%d, dp_replicate=%d, dp_shard=%d); ac=%s",
        ws, parallel_dims.dp_replicate, parallel_dims.dp_shard,
        cfg.run.activation_checkpoint,
    )


__all__ = ["parallelize_llama3_repop"]
