"""The training loop. Skeleton-faithful to plan/05 §5.

The hot path here mirrors the pseudocode in the plan:
  - per-step: pick grad-accum, micro-batch loop, clip, optimizer step
  - eval / checkpoint cadence by consumed-token thresholds
  - spike protocol on grad-norm
  - metrics + W&B every step

The loop is intentionally readable rather than over-abstracted. New
training behaviours go in via the registries (optim, schedule, model
modules); the loop itself rarely changes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from pretrain.config.schema import RootConfig
from pretrain.data.global_stream import GlobalStreamState
from pretrain.data.loader import build_global_loader, build_loader
from pretrain.data.mix_sampler import MixSamplerState
from pretrain.model import build_model
from pretrain.model.fused_loss import fused_ce_z_loss
from pretrain.model.init import init_weights
from pretrain.obs.metrics import (
    MetricsLogger,
    mfu,
    time_section,
    tokens_per_sec,
    transformer_flops_per_token,
)
from pretrain.obs.wandb_run import WandBRun
from pretrain.optim.registry import build_optimizer
from pretrain.optim.schedules import build_schedule, schedule_lr
from pretrain.parallel.env import (
    assert_flash_sdp_enabled,
    init_distributed,
    is_main_process,
    rank,
    set_seed,
    shutdown_distributed,
    world_size,
)
from pretrain.parallel.deterministic_reduce import (
    GRAD_NORM_ALGO,
    REPLICATE_REDUCE_ALGO,
    deterministic_scalar_sum,
)
from pretrain.parallel.fsdp import wrap_model_ddp
from pretrain.parallel.parallel_dims import ParallelDims
from pretrain.parallel.parallelize_llama3_repop import parallelize_llama3_repop
from pretrain.train import global_clip
from pretrain.train.batch_schedule import grad_accum_steps, microbatches_per_step
from pretrain.train.checkpoint import Checkpointer, CheckpointMeta
from pretrain.train.midtrain import check_linear_anneal_resume
from pretrain.train.spike_protocol import Halt, SpikeProtocol
from pretrain.train.state_hash import (
    RunningBatchHasher,
    combine_shard_state,
    compute_state_hash,
    finalize_state_hash,
    local_shard_state_digest,
)
from pretrain.util.git import git_diff, git_sha, run_provenance

LOG = logging.getLogger(__name__)


# SIGTERM handling for graceful pre-eviction checkpointing. Kubelet sends
# SIGTERM ``terminationGracePeriodSeconds`` before SIGKILL on preemption /
# eviction / ``kubectl delete``. The handler just flips a flag; the loop
# all-reduces it at the top of each iteration so every rank exits at the
# same step boundary (``dcp.save`` is collective — a unilateral save would
# deadlock the ranks that didn't get the signal in time).
#
# Module-level so the C-level signal handler can flip it without closure
# tricks; only one ``train()`` runs per process so there's no cross-talk.
_emergency_exit_requested = False


def _handle_sigterm(signum, frame):  # noqa: ARG001
    global _emergency_exit_requested
    if not _emergency_exit_requested:
        LOG.warning("SIGTERM received; will save emergency checkpoint at next step boundary")
        _emergency_exit_requested = True


# Non-REPOP* env that is part of the cross-device-reproducibility contract:
# cuBLAS workspace determinism and the build-time gencode the run's kernels
# were compiled for. Captured verbatim only when set (empty ⇒ unrecorded so the
# audit falls back to its own default rather than replaying a blank).
_CONTRACT_ENV_KEYS = ("CUBLAS_WORKSPACE_CONFIG", "TORCH_CUDA_ARCH_LIST")


def _capture_repop_env() -> dict[str, str]:
    """The kernel/determinism env the audit must re-apply to dispatch identical
    repop kernels and torch determinism flags: every ``REPOP*`` var present in
    the environment (LSQ/int8-PV backward selection — and any repop default that
    repop itself materialised into ``os.environ`` at import) plus the cuBLAS/arch
    contract keys. Otherwise those silently differ in the replay and it diverges.

    We capture only what is *actually set* and never fabricate a value for a var
    left at repop's default (e.g. REPOP_USE_HFMA2_MMACC). Forcing a guessed
    default into the replay would *change* behaviour if repop's real default ever
    differed from the guess — the opposite of faithful capture. An unset var
    stays unset on both sides, so the same pinned repop build applies the same
    default at train and audit time. The audit's arch guard separately *assumes*
    the HFMA2 fast path is on when the var is unrecorded (BFR only on sm_90+) —
    a conservative read-only check, not an injected value.
    """
    env = {k: v for k, v in os.environ.items() if k.startswith("REPOP")}
    for k in _CONTRACT_ENV_KEYS:
        v = os.environ.get(k)
        if v:
            env[k] = v
    return env


@dataclass
class LoopState:
    consumed_tokens: int = 0
    optimizer_step: int = 0
    skipped_step_count: int = 0


def _autocast_ctx(
    use_cuda: bool,
    mixed_precision: bool = True,
    dtype: torch.dtype = torch.bfloat16,
):
    if use_cuda and mixed_precision:
        return torch.autocast(device_type="cuda", dtype=dtype)
    # mixed_precision=False → run the model in fp32 end-to-end.
    # On CPU/MPS we also disable autocast (bf16 emulation is uneven).
    device_type = "cuda" if use_cuda else "cpu"
    return torch.autocast(device_type=device_type, enabled=False)


def _eos_id_from_data_cfg(cfg: RootConfig) -> int:
    return cfg.data.document_separator_id


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """fp32 reduction through repop's reproducible cross-entropy kernel.

    The kernel doesn't support ``ignore_index`` — safe here because the
    data loader emits shifted token-id labels with no -100 mask. ``logits``
    is a plain full-vocab ``Tensor`` (tp=1; TP was removed).
    """
    from repop.nn.loss import cross_entropy as repop_ce

    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    return repop_ce(flat_logits, flat_labels)


def _per_param_weight_norms(model: torch.nn.Module) -> dict[str, float]:
    """Global L2 norm of every parameter's WEIGHT, keyed by name.

    Same collective contract as the weight-side fold below
    (``full_tensor()`` on a sharded DTensor all-reduces the partial norm, so
    every rank must call this in lock-step). Purpose: a tensor whose logged
    ‖w‖ follows the pure weight-decay law w₀·e^(−λ∫γₜdt) for an extended
    stretch is receiving no effective gradient — the frozen-tensor signature
    (seen in the July 2026 1B QAT incident run) — which an offline dashboard can test
    directly from these values plus the logged lr. Diagnostic only; gate via
    ``grad_norm_log.weight_stats_every_n_steps``.
    """
    from torch.distributed.tensor import DTensor
    from torch.nn.utils import get_total_norm

    norms: dict[str, float] = {}
    for name, p in model.named_parameters():
        n = get_total_norm([p.data], norm_type=2.0, error_if_nonfinite=False)
        if isinstance(n, DTensor):
            n = n.full_tensor()
        norms[name] = float(n)
    return norms


def _qk_gain_max(model: torch.nn.Module) -> dict[str, float]:
    """Per-layer attention-logit temperature gauge: max |γ_q ⊙ γ_k| over the
    QK-norm gain vectors. Attention entropy is bounded by the logit scale, and
    the gains sit in the no-decay group, so unbounded gain growth is the slow
    entropy-collapse drift QK-norm alone does not prevent. Collective under
    FSDP (``full_tensor`` on sharded gains) — call on every rank in lock-step.
    Empty when the model has no q_norm/k_norm modules.
    """
    out: dict[str, float] = {}
    for name, mod in model.named_modules():
        q = getattr(mod, "q_norm", None)
        k = getattr(mod, "k_norm", None)
        if q is None or k is None:
            continue
        if getattr(q, "weight", None) is None or getattr(k, "weight", None) is None:
            continue  # gain-free QK-norm: no temperature to gauge
        gq, gk = q.weight, k.weight
        if hasattr(gq, "full_tensor"):
            gq = gq.full_tensor()
        if hasattr(gk, "full_tensor"):
            gk = gk.full_tensor()
        li = _layer_index(name)
        key = str(li) if li is not None else name
        out[key] = round(float((gq.detach().float() * gk.detach().float()).abs().max()), 6)
    return out


def _grad_norm_category(name: str) -> str:
    """Bucket a parameter name into a diagnostic category. ``lsq_scale`` is
    broken out first so the LSQ learned step-size grads are visible
    separately from the int8 weight grads they scale."""
    if "weight_scale" in name:
        return "lsq_scale"
    if "tok_embeddings" in name or "embed" in name:
        return "embed"
    if "lm_head" in name or name.endswith("output.weight"):
        return "lm_head"
    if ".attn." in name:          # incl. wq/wk/wv/wo + q_norm/k_norm
        return "attn"
    if ".ffn." in name or "w_gate" in name or "w_up" in name or "w_down" in name:
        return "ffn"
    if "norm" in name:            # block input/ffn RMSNorms + final norm
        return "norm"
    return "other"


def _layer_index(name: str) -> int | None:
    parts = name.split(".")
    for i, tok in enumerate(parts):
        if tok in ("layers", "blocks") and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _write_grad_norm_log(
    fh,
    step: int,
    consumed_tokens: int,
    norms: dict[str, float],
    top_k: int,
    weight_norms: dict[str, float] | None = None,
    qk_gain_max: dict[str, float] | None = None,
) -> None:
    """Append one JSON line: global + per-category + per-layer L2 norms (folded
    from the per-param norms via ||g||₂ = sqrt(Σ ||g_i||₂²)) plus the ``top_k``
    largest individual params. One ``diff``-able row per logged step. Rows on
    the ``weight_stats_every_n_steps`` cadence additionally carry per-param
    weight norms and the per-layer QK-gain temperature (see
    :func:`_per_param_weight_norms` / :func:`_qk_gain_max`)."""
    cat_sq: dict[str, float] = {}
    layer_sq: dict[int, float] = {}
    for name, v in norms.items():
        cat_sq[_grad_norm_category(name)] = cat_sq.get(_grad_norm_category(name), 0.0) + v * v
        li = _layer_index(name)
        if li is not None:
            layer_sq[li] = layer_sq.get(li, 0.0) + v * v
    rec = {
        "step": step,
        "consumed_tokens": consumed_tokens,
        "global": round(math.sqrt(sum(v * v for v in norms.values())), 6),
        "by_category": {k: round(math.sqrt(s), 6) for k, s in sorted(cat_sq.items())},
        "by_layer": {
            str(k): round(math.sqrt(layer_sq[k]), 6) for k in sorted(layer_sq)
        },
        "top": [
            [n, round(v, 6)]
            for n, v in sorted(norms.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        ],
    }
    if weight_norms is not None:
        rec["weight_norms"] = {n: round(v, 6) for n, v in weight_norms.items()}
    if qk_gain_max is not None:
        rec["qk_gain_max"] = qk_gain_max
    fh.write(json.dumps(rec) + "\n")


def _assert_run_id_agrees(run_id: str, device: torch.device) -> None:
    """Hard-fail if ranks disagree on ``run_id``.

    ``run_id`` selects each rank's ``runs/<run_id>/`` output subtree. If the
    launcher hands different pods different run_ids — e.g. a stale
    cross-submission value read from the RWX PVC's incoherent dentry cache
    during the run_id handoff — the job silently splits its checkpoint across
    two directories and overwrites a prior run's, corrupting both. One tiny
    all_gather at startup turns that into an immediate, loud failure. Uses a
    fixed 64-byte tensor collective (not all_gather_object) so it works on the
    NCCL default PG without object pickling.
    """
    if not torch.distributed.is_initialized():
        return
    raw = run_id.encode("utf-8")
    if len(raw) > 64:
        raise ValueError(
            f"run_id too long for the agreement check ({len(raw)}B > 64): {run_id!r}"
        )
    buf = torch.zeros(64, dtype=torch.uint8, device=device)
    if raw:
        buf[: len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    ws = torch.distributed.get_world_size()
    gathered = [torch.zeros_like(buf) for _ in range(ws)]
    torch.distributed.all_gather(gathered, buf)
    ids = [
        bytes(t.cpu().tolist()).rstrip(b"\x00").decode("utf-8", "replace")
        for t in gathered
    ]
    if len(set(ids)) != 1:
        from collections import Counter

        per_rank = ", ".join(f"rank{r}={i!r}" for r, i in enumerate(ids))
        raise RuntimeError(
            "run_id disagreement across ranks — the launcher handed different "
            f"run_ids to different pods ({dict(Counter(ids))}). Each pod writes "
            f"a different runs/<run_id>/ tree, which corrupts checkpoints. "
            f"Per-rank: {per_rank}"
        )


def train(cfg: RootConfig, resume_from: str | None = None) -> None:
    """Entry point invoked by ``pretrain.cli.train``."""
    init_distributed()
    # Install the graceful-shutdown handler after init_distributed so the
    # process group exists by the time SIGTERM can fire — the per-step
    # check all-reduces on the default PG.
    signal.signal(signal.SIGTERM, _handle_sigterm)
    set_seed(cfg.run.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Code provenance: which commit (and whether dirty) of transformer-pretraining
    # and the repop source repo this run is actually executing. Read live from
    # git on rank 0 only — the training pod keeps transformer-pretraining's .git
    # but the repop checkout's .git may be absent, in which case it logs
    # "unknown" rather than failing. Logged once, up front, so every run's logs
    # are self-identifying.
    if is_main_process():
        for prov in run_provenance():
            LOG.info("code provenance: %s", prov.describe())
    # Fail fast if the launcher handed different run_ids to different pods —
    # the split-brain that silently splits and corrupts checkpoints. Cheap:
    # one 64-byte all_gather before any model build or checkpoint write.
    _assert_run_id_agrees(cfg.run.run_id or "default", device)
    # Plan/06 §3: error out at startup rather than silently fall back to a
    # slow SDPA backend. No-op outside CUDA.
    assert_flash_sdp_enabled()

    # 1) Model. Build → init → FSDP wrap (which also applies AC + per-block
    # compile in that order). FSDP-then-compile is the documented pattern;
    LOG.info("building model %s on device=%s", cfg.model.name, device)
    model = build_model(cfg.model, device=device)
    # Seed init explicitly with cfg.run.seed (no rank offset) so every
    # rank constructs the same full model before FSDP shards it. The
    # global RNG's per-rank offset from set_seed() is left in place for
    # any future runtime stochasticity (dropout etc.).
    init_weights(model, seed=cfg.run.seed)
    # Build parallel dims + device mesh. dp_shard=-1 lets ParallelDims fill
    # the remaining ranks; explicit value is honored when set.
    parallel_dims = ParallelDims(
        dp_replicate=cfg.run.dp_replicate_size,
        dp_shard=cfg.run.dp_shard_size,
        world_size=world_size(),
    )
    if torch.cuda.is_available():
        parallel_dims.build_mesh()
    if cfg.run.parallel == "ddp":
        model = wrap_model_ddp(model, cfg)
    else:
        model = parallelize_llama3_repop(model, cfg, parallel_dims)

    # 2) Optimizer + schedule.
    optimizer = build_optimizer(model, cfg.optim)
    # Eager zero-state priming: keeps the optim-state key set constant across
    # the whole run (params with grad=None — e.g. LSQ scales during the bf16
    # warm-start — would otherwise have no state, breaking pre-flip resumes).
    # Bitwise-neutral vs lazy init; mirrored in audit_replay.
    from pretrain.optim.adamw_repop import prime_optimizer_state

    prime_optimizer_state(optimizer)
    lr_schedule = build_schedule(cfg.schedule, cfg.train, cfg.optim)
    # Clipping: deterministic global-norm clip (pretrain.train.global_clip) —
    # stateless by design.

    # 3) Data.
    # Stride the MixSampler by data-parallel rank. With the
    # ("dp_replicate", "fsdp") mesh and no TP, dp_rank is just the global rank,
    # and ``dp_world_size`` (= dp_replicate * dp_shard = world_size) is the
    # denominator for the batch / token-accounting math below (grad_accum,
    # tokens_this_step).
    dp_world_size = parallel_dims.dp_world_size
    dp_rank = parallel_dims.dp_rank()
    # Auditable mode: canonical world-size-independent data stream + (already
    # wired) deterministic reduce-scatter, so the run can be reproduced bitwise
    # on a single device. Default mode keeps the legacy per-rank MixSampler.
    auditable = cfg.run.reduction_mode == "deterministic_allgather"
    if auditable:
        loader, sampler = build_global_loader(
            cfg.data,
            cfg.train,
            rank=dp_rank,
            world_size=dp_world_size,
            seed=cfg.run.seed,
            eos_id=_eos_id_from_data_cfg(cfg),
        )
    else:
        loader, sampler = build_loader(
            cfg.data,
            micro_batch_size=cfg.train.micro_batch_size,
            rank=dp_rank,
            world_size=dp_world_size,
            seed=cfg.run.seed,
            eos_id=_eos_id_from_data_cfg(cfg),
        )

    # 4) Checkpoint + observability.
    run_dir = Path(cfg.run.output_dir) / (cfg.run.run_id or "default")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpointer(run_dir / "checkpoints")
    # Refuse to cold-start on top of an existing run's checkpoints. Without
    # --resume-from, training restarts at step 0 and DCP overwrites
    # step_*/dcp in place — the reported overwrite. The usual trigger is a
    # run_id collision (pinned RUN_ID, or a pod recreation reusing the same
    # run_id). To continue a run pass --resume-from; to start fresh use a new
    # run_id. Decide on rank 0 and broadcast so a per-rank PVC-cache
    # disagreement can't leave some ranks raising while others hang at the
    # next collective.
    if resume_from is None:
        exists_flag = torch.tensor(
            [1 if ckpt.has_checkpoints() else 0], device=device, dtype=torch.int32
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(exists_flag, src=0)
        if exists_flag.item() != 0:
            raise RuntimeError(
                f"cold start (no --resume-from) but {run_dir / 'checkpoints'} "
                f"already contains checkpoints — refusing to overwrite a prior "
                f"run. Pass --resume-from to continue it, or use a fresh run_id."
            )
    metrics = (
        MetricsLogger(
            run_dir / cfg.logging.metrics_jsonl_path,
            wandb_run=WandBRun(
                project=cfg.logging.wandb_project,
                run_name=cfg.run.run_id or cfg.model.name,
                config=cfg.model_dump(),
                run_dir=str(run_dir),
            ),
        )
        if is_main_process()
        else None
    )
    # Dedicated state-hash stream — separate file so cross-run diffs are
    # one ``diff`` away without grepping the noisy main metrics JSONL.
    # Rank-0 only; opened only when periodic hashing is on.
    state_hash_fh = None
    if cfg.train.state_hash.every_n_steps > 0 and is_main_process():
        state_hash_path = run_dir / "logs" / "state_hashes.jsonl"
        state_hash_path.parent.mkdir(parents=True, exist_ok=True)
        state_hash_fh = open(state_hash_path, "at", encoding="utf-8", buffering=1)
    # Per-layer grad-norm diagnostic stream — rank-0 only; the norm
    # computation itself is collective (see _per_param_weight_norms) and runs
    # on every rank when due.
    grad_log_fh = None
    if cfg.train.grad_norm_log.every_n_steps > 0 and is_main_process():
        grad_log_path = run_dir / cfg.train.grad_norm_log.path
        grad_log_path.parent.mkdir(parents=True, exist_ok=True)
        grad_log_fh = open(grad_log_path, "at", encoding="utf-8", buffering=1)

    # Per-rank rolling batch hasher + a gloo group spanning DP ranks at
    # this rank's TP coord. Built unconditionally when state-hash + batch
    # are enabled (every rank, not just main) because ``global_digest``
    # is a collective. ``make_dp_gloo_group`` itself is collective via
    # ``new_group`` and must be invoked at the same point on every rank.
    batch_hasher: RunningBatchHasher | None = None
    dp_gloo_group = None
    # The DP gloo group all-gathers the 32-byte per-rank shard-state and batch
    # digests at periodic hash time. Built whenever periodic hashing is on (not
    # just for the batch term — the sharded weight/grad/moment combine needs it
    # too). The one-shot init hash stays on the topology-invariant full_tensor
    # path (see below), so it needs no group. ``make_dp_gloo_group`` is
    # collective (new_group) → same point on every rank.
    _periodic_hashing = cfg.train.state_hash.every_n_steps > 0
    if _periodic_hashing:
        dp_gloo_group = parallel_dims.make_dp_gloo_group()
    if _periodic_hashing and cfg.train.state_hash.include_batch:
        batch_hasher = RunningBatchHasher()

    def _sharded_state_hash(
        prev_hash: str | None, *, include_grads: bool, include_batch: bool = True
    ) -> str:
        """Cluster-side sharded canonical hash. Each rank digests only its
        LOCAL shard (no full_tensor), the DP gloo group all-gathers + combines
        the 32-byte per-rank digests, then param_groups + the running batch
        digest + the previous chained hash are folded in. Two collectives
        (shard-state combine, then batch global_digest) issued in a fixed order
        on every rank → all ranks return the same hex. The single-device
        audit reproduces this via state_hash.audit_shard_state_digest."""
        local = local_shard_state_digest(
            model, optimizer=optimizer, include_grads=include_grads
        )
        shard_state = combine_shard_state(local, dp_gloo_group)
        bd = (
            batch_hasher.global_digest(dp_gloo_group)
            if (include_batch and batch_hasher is not None)
            else None
        )
        return finalize_state_hash(
            prev_hash=prev_hash,
            shard_state_digest=shard_state,
            optimizer=optimizer,
            batch_digest=bd,
        )
    # SpikeProtocol is constructed BEFORE the resume block so the
    # resume path can replay its mid-cooldown / halt-window state into
    # the same instance. Otherwise a spike whose cooldown straddled the
    # checkpoint boundary would silently end on resume, stepping the
    # optimizer instead of skipping it.
    spike = SpikeProtocol(
        threshold=cfg.train.spike.grad_norm_threshold,
        skips_in_window_to_halt=cfg.train.spike.skips_in_window_to_halt,
        halt_window_steps=cfg.train.spike.halt_window_steps,
        skip_steps_on_spike=cfg.train.spike.skip_steps_on_spike,
        start_step=cfg.train.spike.start_step,
    )

    # 5) Resume.
    state = LoopState()
    sampler_state: MixSamplerState | GlobalStreamState | None = None
    # Running state-hash chain — None until the first hashed step (cold
    # start) or until ``meta.chained_hash`` is restored (resume). Declared
    # here, before the resume block, so the resume branch can override and
    # ``_save_checkpoint`` (defined below) captures the right cell.
    chained_hash: str | None = None
    # Phase-boundary LR re-warmup anchor: set to the resume token count when we
    # reset the optimizer, so the LR can ramp from 0 over schedule.rewarm_tokens
    # measured from here (cold moments). Stays None on a normal resume / fresh run.
    rewarm_anchor_tokens: int | None = None
    # Dead-tensor assertion state: param name -> consecutive exactly-zero
    # logged-grad-norm count. See the check in the grad-log block.
    _dead_counts: dict[str, int] = {}
    if resume_from:
        sampler_state, meta, extras = ckpt.load(
            resume_from, model, optimizer,
            model_weights_only=cfg.train.resume_reset_optimizer,
        )
        state.consumed_tokens = meta.consumed_tokens
        state.optimizer_step = meta.step
        chained_hash = meta.chained_hash
        # Midtraining anneal guard: fail at launch on anchor/budget arithmetic
        # that would otherwise silently mis-schedule (no-op unless the
        # schedule is linear_anneal).
        check_linear_anneal_resume(cfg, meta.consumed_tokens)
        if cfg.train.resume_reset_optimizer:
            rewarm_anchor_tokens = meta.consumed_tokens
            LOG.info(
                "resume_reset_optimizer: loaded model weights only; optimizer "
                "moments start cold. LR re-warm over %d tokens from %d.",
                cfg.schedule.rewarm_tokens, meta.consumed_tokens,
            )
        elif cfg.schedule.rewarm_on_resume and cfg.schedule.rewarm_tokens > 0:
            # Moments-kept fork with a deliberate LR ramp (e.g. waking frozen
            # params). Inherit a recorded anchor so a crash-resume of THIS run
            # continues the original ramp instead of re-warming from the crash
            # point; anchor here only when the loaded meta carries none.
            _prev_anchor = getattr(meta, "rewarm_anchor_tokens", -1)
            rewarm_anchor_tokens = (
                _prev_anchor if _prev_anchor >= 0 else meta.consumed_tokens
            )
            LOG.info(
                "rewarm_on_resume: LR re-warm over %d tokens anchored at %d "
                "(moments kept; anchor %s).",
                cfg.schedule.rewarm_tokens, rewarm_anchor_tokens,
                "inherited" if _prev_anchor >= 0 else "set at this resume",
            )
        # Reseed the running batch hasher with its saved per-rank chain.
        # Without this the chain restarts from zero on resume and
        # ``state_hash`` diverges from a continuous run from step+1
        # onward, even if weights and optimizer moments are bit-equal.
        bh_digest = extras.get("batch_hasher_digest")
        if batch_hasher is not None and bh_digest is not None:
            batch_hasher = RunningBatchHasher(prev_digest=bh_digest)
        elif batch_hasher is not None:
            LOG.warning(
                "no batch_hasher digest in %s; chain restarts (state_hash "
                "will not match a continuous run)", resume_from,
            )
        sp_state = extras.get("spike_state")
        if sp_state is not None:
            spike.load_state_dict(sp_state)
        # Rebuild loader at the correct stream state. Same dp_rank /
        # dp_world_size folding as the initial build above.
        if auditable:
            loader, sampler = build_global_loader(
                cfg.data,
                cfg.train,
                rank=dp_rank,
                world_size=dp_world_size,
                seed=cfg.run.seed,
                eos_id=_eos_id_from_data_cfg(cfg),
                state=sampler_state,
                start_consumed_tokens=state.consumed_tokens,
                start_step=state.optimizer_step,
            )
        else:
            loader, sampler = build_loader(
                cfg.data,
                micro_batch_size=cfg.train.micro_batch_size,
                rank=dp_rank,
                world_size=dp_world_size,
                seed=cfg.run.seed,
                eos_id=_eos_id_from_data_cfg(cfg),
                sampler_state=sampler_state,
            )
        LOG.info(
            "resumed from %s at step=%d consumed_tokens=%d chained_hash=%s",
            resume_from, state.optimizer_step, state.consumed_tokens,
            (chained_hash[:16] + "…") if chained_hash else "none",
        )

    # 6) The loop.
    LOG.info("entering training loop; total_tokens=%s", cfg.train.total_tokens)
    last_ckpt_at = state.consumed_tokens
    started_at = time.time()
    loader_iter = iter(loader)
    # MFU numerator — full fwd+bwd model FLOPs per token (dense + the O(seq)
    # attention term + LM head). Constant across the run, so compute once.
    flops_per_token = transformer_flops_per_token(
        n_layers=cfg.model.n_layers,
        d_model=cfg.model.d_model,
        seq_len=cfg.train.seq_len,
        vocab_size=cfg.model.vocab_size,
    )

    def _save_checkpoint(state_hash: str | None = None) -> None:
        stream_state = sampler.state()
        meta = CheckpointMeta(
            consumed_tokens=state.consumed_tokens,
            step=state.optimizer_step,
            git_sha=git_sha(),
            config_resolved=cfg.model_dump_json(indent=2),
            tokenizer_hash="",       # filled by CLI from manifest
            container_digest="",     # filled by CLI from env
            chained_hash=chained_hash,
            reduction_mode=cfg.run.reduction_mode,
            dp_world_size=dp_world_size,
            dp_replicate=parallel_dims.dp_replicate,
            dp_shard=parallel_dims.dp_shard,
            # Record the cross-replica reduction order so the audit replays it.
            # Only the auditable path installs the deterministic replicate
            # all-reduce; legacy NCCL runs aren't bit-reproducible anyway.
            replicate_reduce_algo=(
                REPLICATE_REDUCE_ALGO if auditable else "nccl"
            ),
            # Grad-norm fold the run used for clipping + the spike trigger.
            # global_clip uses the deterministic ascending-shard fold
            # UNCONDITIONALLY (correct under both reduction modes), so record
            # it unconditionally; see deterministic_reduce.GRAD_NORM_ALGO.
            grad_norm_algo=GRAD_NORM_ALGO,
            # Clipping algorithm, so the audit reconstructs the exact clipper
            # (robust to config drift, like repop_env). "global" = the
            # stateless deterministic global-norm clip; its max_norm comes
            # from config_resolved (train.grad_clip).
            clip_algo="global",
            rewarm_anchor_tokens=(
                rewarm_anchor_tokens if rewarm_anchor_tokens is not None else -1
            ),
            seed=cfg.run.seed,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            windows_emitted=(
                getattr(stream_state, "windows_emitted", 0) if auditable else 0
            ),
            repop_env=_capture_repop_env(),
        )
        # global_digest is collective — every rank must call it.
        ckpt_batch_digest = (
            batch_hasher.global_digest(dp_gloo_group)
            if batch_hasher is not None
            else None
        )
        bh_local = (
            batch_hasher.local_digest() if batch_hasher is not None else None
        )
        # Every save (cadence, final step, emergency exit) must thread the
        # step's already-computed canonical hash in. Recomputing one here is
        # NEVER correct on a hashed run: by save time ``chained_hash`` may
        # already have advanced to this step's own hash (double-link) and the
        # grads may be cleared (hashed as "grad_none") — a digest no replay
        # can reproduce. Exactly that recompute stamped the 1B run's final
        # checkpoint (step 80957) with an unverifiable state_hash.txt
        # so fail loudly rather than publish another one.
        if state_hash is None and _periodic_hashing:
            raise ValueError(
                f"checkpoint save at step={state.optimizer_step} has no "
                f"pre-computed state hash. Saves on a hashed run must stamp "
                f"the hash computed AT the step (pre-zero_grad, chained from "
                f"the pre-advance chain) — recomputing it after the fact "
                f"produces an unreproducible digest."
            )
        # Auditable runs persist one global_stream.json (identical on every
        # rank); legacy runs persist per-rank sampler state.
        ckpt.save(
            state.optimizer_step,
            model,
            optimizer,
            None if auditable else stream_state,
            meta,
            batch_digest=ckpt_batch_digest,
            batch_hasher_digest=bh_local,
            spike_state=spike.state_dict(),
            global_stream_state=stream_state if auditable else None,
            # Pre-computed canonical hash (weights+optim+grads+batch, chained)
            # from the loop's post-step computation. The checkpoint writes THIS
            # to state_hash.txt so it matches state_hashes.jsonl exactly (or,
            # on an off-cadence save, the side-link the audit
            # reproduces). Always non-None on hashed runs — see the guard
            # above; only legacy no-hash runs pass None (the checkpoint then
            # computes its own best-effort full-tensor digest).
            state_hash=state_hash,
        )
    # Per-phase GPU timing via cudaEvent. Off by default; flip on for one
    # diagnostic run with PRETRAIN_CUDA_EVENT_TIMING=1. Cost: one host-side
    # sync at end of step (~ms-scale; negligible vs typical step time, but
    # disables some pipelining of the next step's data load).
    cuda_timing_enabled = (
        use_cuda
        and os.environ.get("PRETRAIN_CUDA_EVENT_TIMING", "0") == "1"
    )

    def _ev() -> torch.cuda.Event | None:
        return torch.cuda.Event(enable_timing=True) if cuda_timing_enabled else None

    # Init audit artifact. On a cold start, before any optimizer step, record
    # the freshly initialized weights' hash so the initialization itself is
    # reproducible/auditable. We persist ONLY the hash (``state_hash_init.txt``),
    # NOT a full step-0 checkpoint: init is reconstructable from the seed
    # (device-independent repop trunc_normal), so ``audit_replay --from-init``
    # rebuilds build_model + init_weights(seed), verifies it reproduces this
    # hash, then replays forward to the first real checkpoint on the SAME
    # codepath as any interval audit. A step-0 DCP checkpoint would be both
    # redundant (init is regenerable) and unusable as a replay source anyway —
    # the optimizer state is lazily created on the first .step(), so a step-0
    # checkpoint has no optimizer moments and an interval load from it fails on
    # the missing ``optim.state.*`` leaves. The run descriptor the audit needs
    # (seed / config / repop_env / topology) lives in every checkpoint's
    # meta.json, so no step-0 metadata file is required either.
    if resume_from is None and cfg.train.state_hash.at_init:
        # Include the optimizer so the init hash covers its config (param_groups:
        # lr/betas/eps/weight_decay/amsgrad) and the zero state that
        # prime_optimizer_state materialised above (audit_replay primes the
        # same way, so the two hashes agree). Grads are absent at init, so
        # include_grads=False. The hash is standalone (prev_hash=None) and NOT
        # chained into the periodic step-N hashes, so the existing chain (and the
        # step-10/20/… digests) are unchanged.
        #
        # The init hash deliberately stays on the topology-invariant full_tensor
        # path (compute_state_hash, NOT the sharded v3 path the periodic hashes
        # use): it is computed once at startup so its cost is irrelevant, and the
        # invariance lets ``audit_replay --from-init --config-name`` verify a run's
        # init from seed+config alone, with no checkpoint to recover dp_shard from.
        # full_tensor() is collective → runs on every rank; only rank 0 writes.
        init_hash = compute_state_hash(
            model, optimizer=optimizer, include_grads=False, prev_hash=None
        )
        if state_hash_fh is not None:
            state_hash_fh.write(json.dumps({
                "step": 0,
                "consumed_tokens": 0,
                "kind": "init",
                "state_hash": init_hash,
            }) + "\n")
        if is_main_process():
            (ckpt.root / "state_hash_init.txt").write_text(init_hash + "\n")
        LOG.info("init weights-only state hash: %s (-> state_hash_init.txt)", init_hash[:16])

    try:
        while state.consumed_tokens < cfg.train.total_tokens:
            # Graceful shutdown (SIGTERM) is polled at the END of the
            # iteration — after the optimizer step, before the canonical
            # hash — NOT here. Saving from the top of the loop wrote a
            # checkpoint whose state_hash.txt no replay could reproduce:
            # the previous iteration had already advanced the chain past
            # its own hash (double-link) and cleared the grads via
            # zero_grad (hashed as "grad_none"). See the guard in the save
            # helper and the emergency-exit poll below.

            timings: dict[str, float] = {}

            if auditable:
                # Canonical, world-size-independent micro-batch count. The
                # audit emulates the same N-way split, so each rank must take
                # an equal share — require M divisible by the DP degree (true
                # for the power-of-2 topologies we run). Per-rank accum then
                # equals M // N, and the ShardedWindowView yields exactly that
                # many micro-batches for this step.
                M = microbatches_per_step(state.consumed_tokens, cfg.train)
                if M % dp_world_size != 0:
                    raise ValueError(
                        f"auditable run needs microbatches_per_step ({M}) divisible "
                        f"by dp_world_size ({dp_world_size}); adjust global_batch_tokens "
                        f"or the topology so each rank takes an equal share."
                    )
                accum = M // dp_world_size
            else:
                accum = grad_accum_steps(
                    state.consumed_tokens, cfg.train, dp_world_size=dp_world_size
                )

            lr = lr_schedule(state.consumed_tokens, cfg.train.total_tokens)
            # Phase-boundary re-warm: linearly ramp LR 0->schedule over
            # rewarm_tokens from the reset point, so cold optimizer moments don't
            # take a full-LR step before exp_avg_sq settles (SPAM 2501.06842).
            if rewarm_anchor_tokens is not None and cfg.schedule.rewarm_tokens > 0:
                elapsed = state.consumed_tokens - rewarm_anchor_tokens
                if elapsed < cfg.schedule.rewarm_tokens:
                    lr = lr * max(0.0, elapsed / cfg.schedule.rewarm_tokens)
            schedule_lr(optimizer, lr)

            # bf16 warm-start (qat.enable_at_step): set the runtime QAT toggle
            # as a pure function of (step, config) at the top of EVERY step —
            # idempotent and state-free, so resume needs nothing extra and the
            # audit mirrors the same call. See model/modules/qat_warmstart.
            if cfg.model.qat.enabled and cfg.model.qat.enable_at_step > 0:
                from pretrain.model.modules.qat_warmstart import set_qat_active

                set_qat_active(
                    model,
                    state.optimizer_step >= cfg.model.qat.enable_at_step,
                )

            micro_loss_total = 0.0
            micro_zloss_total = 0.0
            t_step = time.perf_counter()

            # Per-phase events (None when timing disabled).
            fwd_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            bwd_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            e_clip_start = e_clip_end = e_opt_end = None

            with time_section("step", timings):
                for k in range(accum):
                    batch = next(loader_iter)
                    # Hash the pinned-CPU loader output BEFORE the device
                    # copy — moves zero bytes off-host and adds no sync.
                    # Each microbatch chains into the per-rank running
                    # digest; cross-rank combine happens at hash time.
                    if batch_hasher is not None:
                        batch_hasher.update(batch)
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)

                    # FSDP2 grad-sync gating for grad accumulation: defer
                    # the reduce-scatter / cross-mesh all-reduce of grads
                    # until the LAST microbatch of the step. Without this,
                    # every microbatch triggers a full grad sync — under
                    # HSDP that means a cross-pod grad all-reduce per
                    # micro, which dominates wall time at the 8B / 32-
                    # micro / NCCL-Socket scale. The API is a no-op on
                    # modules that haven't been ``fully_shard``-wrapped.
                    if hasattr(model, "set_requires_gradient_sync"):
                        model.set_requires_gradient_sync(k == accum - 1)

                    e_fwd_start = _ev()
                    e_fwd_end = _ev()
                    e_bwd_end = _ev()
                    if e_fwd_start is not None:
                        e_fwd_start.record()

                    with _autocast_ctx(use_cuda, mixed_precision=cfg.run.mixed_precision):
                        out = model(input_ids)
                        # CE + z-loss FUSED over the full-vocab logits: one shared
                        # softmax, one combined grad-logits, recomputed in row-
                        # chunks (pretrain.model.fused_loss). Computing them
                        # separately materialised the [N, V] softmax/grad twice at
                        # the lm_head and hung the first step (phase2-zloss).
                        z_coeff = cfg.model.z_loss.coeff if cfg.model.z_loss.enabled else 0.0
                        ce, zloss = fused_ce_z_loss(out.logits, labels, z_coeff)
                        # Scale by a host-computed reciprocal, NOT ``/ accum``.
                        # ``tensor / scalar`` is correctly-rounded true division on
                        # CPU/MPS but reciprocal-multiply on CUDA, so for non-power-
                        # of-2 accum (e.g. 6 in the main phase, 12 late) the 1/accum
                        # grad scaling diverges in the last bit across devices and
                        # breaks the single-device audit. ``* (1.0/accum)`` forces
                        # mult-by-reciprocal everywhere — a no-op on CUDA (already
                        # what it does), and makes CPU/MPS match it. Mirrors the
                        # × const AVG fix in deterministic_reduce / audit_replay.
                        loss = (ce + zloss) * (1.0 / accum)

                    if e_fwd_end is not None:
                        e_fwd_end.record()

                    loss.backward()

                    if e_bwd_end is not None:
                        e_bwd_end.record()
                        fwd_pairs.append((e_fwd_start, e_fwd_end))
                        bwd_pairs.append((e_fwd_end, e_bwd_end))

                    micro_loss_total += float(ce.detach())
                    micro_zloss_total += float(zloss.detach()) if isinstance(zloss, torch.Tensor) else 0.0

                # Cross-replica (HSDP) reduction. In the default bucketed mode the
                # per-module all-reduce hooks only stashed each module's reduce-
                # scattered shard during the (last-microbatch) backward; do the
                # single coalesced recursive-doubling all-reduce now — after the
                # full backward, before the grad-norm fold / optimizer read .grad
                # (which aliases those buffers). Must run on every rank (it's a
                # collective). No-op when not bucketed / dp_replicate==1 / nccl mode.
                rhook = getattr(model, "_replicate_reduce_hook", None)
                if rhook is not None:
                    rhook.flush()

                e_clip_start = _ev()
                if e_clip_start is not None:
                    e_clip_start.record()

                # Deterministic global-norm clip (BFR tensor÷tensor coefficient
                # — see global_clip). Returns the fold's PRE-clip per-tensor
                # norms (telemetry reuses them: one collective, not one per
                # parameter) plus the global norm for the spike protocol.
                _pt_norms, grad_norm = global_clip.clip_train(
                    model, cfg.train.grad_clip, device
                )
                gn = float(grad_norm)
                _gnl = cfg.train.grad_norm_log
                if _gnl.every_n_steps > 0 and (state.optimizer_step % _gnl.every_n_steps == 0):
                    _pl_norms = {n: float(v) for n, v in _pt_norms.items()}
                    # Dead-tensor assertion: a parameter whose gradient is
                    # EXACTLY zero for N consecutive logged steps is a bug
                    # (frozen clip state, dead quantization path, detached
                    # graph), not a training regime — fail loudly instead of
                    # training it silently for 50k steps (the 20260703 q-side
                    # freeze). Counting is rank-uniform: the norms come from
                    # the same collective fold on every rank.
                    if cfg.train.dead_tensor_assert_steps > 0:
                        for _n, _v in _pl_norms.items():
                            if _v == 0.0:
                                _dead_counts[_n] = _dead_counts.get(_n, 0) + 1
                                if _dead_counts[_n] >= cfg.train.dead_tensor_assert_steps:
                                    raise RuntimeError(
                                        f"dead tensor: {_n} has had EXACTLY zero "
                                        f"gradient for {_dead_counts[_n]} consecutive "
                                        f"logged steps (through step "
                                        f"{state.optimizer_step}). This is the "
                                        f"20260703 freeze signature — refusing to "
                                        f"train it silently."
                                    )
                            else:
                                _dead_counts.pop(_n, None)
                    # Weight-side diagnostics on their own (coarser) cadence.
                    # Both helpers are collective; the gate (optimizer_step) is
                    # identical on every rank, so they stay in lock-step.
                    _w_norms = _qk_gains = None
                    _wsn = _gnl.weight_stats_every_n_steps
                    if _wsn > 0 and state.optimizer_step % _wsn == 0:
                        _w_norms = _per_param_weight_norms(model)
                        _qk_gains = _qk_gain_max(model)
                    if grad_log_fh is not None:
                        _write_grad_norm_log(
                            grad_log_fh, state.optimizer_step,
                            state.consumed_tokens, _pl_norms, _gnl.top_k,
                            weight_norms=_w_norms, qk_gain_max=_qk_gains,
                        )

                ce_avg = micro_loss_total / accum
                zl_avg = micro_zloss_total / accum

                e_clip_end = _ev()
                if e_clip_end is not None:
                    e_clip_end.record()

                # The canonical state hash is computed AFTER the optimizer step
                # (below, after the step counter increments) on the POST-step
                # weights/moments + the step's gradients — which are still live
                # because ``zero_grad`` is deferred to the end of the iteration.
                # The one value is logged to state_hashes.jsonl, written verbatim
                # to the checkpoint's state_hash.txt, and reproduced bit-for-bit
                # by pretrain.cli.audit_replay.
                sh = cfg.train.state_hash
                if spike.should_skip(grad_norm=gn, loss=ce_avg, step=state.optimizer_step):
                    spike.record_skip(state.optimizer_step)
                    state.skipped_step_count += 1
                else:
                    optimizer.step()

                e_opt_end = _ev()
                if e_opt_end is not None:
                    e_opt_end.record()

            step_dt = time.perf_counter() - t_step

            # ---- Global-batch loss (telemetry ONLY) ---------------------------
            # ``ce_avg``/``zl_avg`` above are rank-local (this rank's ``accum``
            # microbatches — 1/dp_world_size of the global batch), so the
            # ``loss_ce`` curve is a noisy subset. Fold the per-rank partial sums
            # deterministically (all_gather + host ascending-rank fp64 fold) into
            # the true global-batch mean. Deterministic so the logged value is
            # topology-invariant and audit-reproducible in principle; NOTHING may
            # consume it besides ``metrics`` — the spike gate stays on the local
            # ``ce_avg`` (mirrored by audit_replay), and it never touches state
            # or the state hash. Equal-weight AVG over ranks is exact: the fused
            # CE is a mean over a fixed micro_batch_size*seq_len rows and
            # ``accum`` is rank-uniform. Must run on EVERY rank (collective) —
            # keep it outside the rank-0 ``metrics`` gate. Reciprocal-multiply,
            # not divide, per the BFR convention.
            _loss_sums = deterministic_scalar_sum(
                torch.tensor(
                    [micro_loss_total, micro_zloss_total],
                    dtype=torch.float64,
                    device=device,
                )
            )
            _inv_global = 1.0 / (accum * max(world_size(), 1))
            ce_global = _loss_sums[0] * _inv_global
            zl_global = _loss_sums[1] * _inv_global

            # Resolve CUDA-event durations. ``elapsed_time`` does NOT
            # implicitly wait — it raises if either event hasn't completed
            # yet — so explicitly sync the final event first. Events are
            # recorded in order on the same stream, so syncing the last
            # one guarantees all prior events have fired. A skipped spike
            # step still measures correctly — opt phase just shows the
            # zero_grad cost.
            phase_ms: dict[str, float] = {}
            if cuda_timing_enabled and e_opt_end is not None:
                e_opt_end.synchronize()
                fwd_ms = sum(s.elapsed_time(e) for s, e in fwd_pairs)
                bwd_ms = sum(s.elapsed_time(e) for s, e in bwd_pairs)
                clip_ms = e_clip_start.elapsed_time(e_clip_end)
                opt_ms = e_clip_end.elapsed_time(e_opt_end)
                phase_ms = {
                    "phase_fwd_ms": fwd_ms,
                    "phase_bwd_ms": bwd_ms,
                    "phase_clip_ms": clip_ms,
                    "phase_opt_ms": opt_ms,
                    "phase_cuda_total_ms": fwd_ms + bwd_ms + clip_ms + opt_ms,
                    "phase_fwd_per_micro_ms": fwd_ms / max(accum, 1),
                    "phase_bwd_per_micro_ms": bwd_ms / max(accum, 1),
                }
            # Cross-replica (HSDP) reduction timing, same env gate as the phase
            # metrics. Drain on EVERY rank (it clears the hook's accumulated CUDA
            # events — skipping a rank would leak them); only rank 0 logs it via
            # ``metrics`` below. Empty when timing off or dp_replicate == 1.
            replicate_ms: dict[str, float] = {}
            if cuda_timing_enabled:
                rhook = getattr(model, "_replicate_reduce_hook", None)
                if rhook is not None:
                    stats = rhook.pop_stats()
                    if stats is not None:
                        replicate_ms = stats
            tokens_this_step = (
                accum * cfg.train.micro_batch_size * cfg.train.seq_len * dp_world_size
            )
            state.consumed_tokens += tokens_this_step
            state.optimizer_step += 1

            # ---- LSQ weight-scale refresh (post-step, pre-hash) --------------
            # Re-pin each LSQ layer's per-channel weight_scale to its just-
            # updated weights, to stop the from-scratch int8-QAT scale-runaway
            # (see qat.scale_refresh_every_n_steps). MUST run before the state
            # hash below — it mutates weight_scale, and audit_replay mirrors
            # this exact call at the same point so refresh-enabled runs stay
            # bit-reproducible. The gate (optimizer_step) is identical on every
            # rank; the reduction is over the un-sharded K axis (no collective),
            # so each rank refreshes its own shard in lock-step. Runs regardless
            # of a spike-skip (idempotent when weights are unchanged), matching
            # audit_replay. No-op unless method="lsq" and the knob is > 0.
            _sr = cfg.model.qat.scale_refresh_every_n_steps
            if _sr > 0 and state.optimizer_step % _sr == 0:
                from repop.qat.lsq import refresh_lsq_weight_scales

                refresh_lsq_weight_scales(model)


            # ---- Emergency-exit poll (post-step, pre-hash) --------------------
            # If SIGTERM fired on *any* rank since the last check, save one
            # final checkpoint and exit on *all* ranks at the same step
            # boundary. Polled HERE — after the optimizer step, before the
            # canonical hash and ``zero_grad`` — so the shutdown save stamps
            # this step's hash exactly as a cadence save would: gradients
            # still live, chain not yet advanced past this step.
            # The all-reduce keeps ranks in lock-step so the collective hash
            # + ``dcp.save`` below cannot deadlock. Cost is one tiny int32
            # collective per step — negligible.
            exit_flag = torch.tensor(
                [1 if _emergency_exit_requested else 0],
                device=device,
                dtype=torch.int32,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    exit_flag, op=torch.distributed.ReduceOp.MAX
                )
            _exit_due = exit_flag.item() != 0
            # Last iteration: the while condition fails after this step, so any
            # unsaved tail must be checkpointed by THIS iteration's save arm —
            # at the step, like every other save (never after the fact).
            _final_step = state.consumed_tokens >= cfg.train.total_tokens

            # ---- Canonical state hash (post-step, gradients still live) -------
            # One value = H(weights + optimizer state + gradients + running batch
            # digest, chained to the previous captured hash). Written identically
            # to state_hashes.jsonl (below) and the checkpoint's state_hash.txt
            # (threaded into _save_checkpoint), and reproduced by audit_replay.
            # Computed when a hash is due OR any checkpoint save fires this step
            # (cadence, emergency exit, final step), so every checkpoint gets the
            # matching value. The gate (optimizer_step / consumed_tokens /
            # all-reduced exit flag) is identical on every rank, so the collective
            # shard-state all_gather inside _sharded_state_hash stays in lock-step.
            state_hash_hex: str | None = None
            ckpt_state_hash: str | None = None
            hash_dt = 0.0
            _hash_due = sh.every_n_steps > 0 and state.optimizer_step % sh.every_n_steps == 0
            # Checkpoint cadence: step-based when ckpt_every_steps > 0 (mirrors the
            # hash gate above so a ckpt step is always also a hash step → the
            # sharded hash below is computed once and shared),
            # else the legacy token cadence.
            if cfg.train.ckpt_every_steps > 0:
                _ckpt_due = state.optimizer_step % cfg.train.ckpt_every_steps == 0
            else:
                _ckpt_due = state.consumed_tokens - last_ckpt_at >= cfg.train.ckpt_every_tokens
            if _hash_due or _ckpt_due or _exit_due or _final_step:
                t_hash = time.perf_counter()
                _h = _sharded_state_hash(
                    chained_hash, include_grads=sh.include_grads
                )
                hash_dt = time.perf_counter() - t_hash
                ckpt_state_hash = _h
                if _hash_due:
                    chained_hash = _h
                    state_hash_hex = _h

            # State-hash stream — rank-0, gated by config. Logged at
            # INFO so it lands in the cluster pod logs alongside the
            # standard step messages, and appended to a dedicated JSONL
            # so two runs can be diff'd without touching the noisy main
            # metrics file. Truncated digest in the log line keeps each
            # line scannable; the JSONL keeps the full 64-char hex.
            if state_hash_hex is not None and state_hash_fh is not None:
                LOG.info(
                    "state_hash step=%d %s",
                    state.optimizer_step, state_hash_hex[:16],
                )
                state_hash_fh.write(json.dumps({
                    "step": state.optimizer_step,
                    "consumed_tokens": state.consumed_tokens,
                    "state_hash": state_hash_hex,
                }) + "\n")

            # Checkpoint save: cadence, emergency exit, or the run's final
            # step with an unsaved tail. All three arms fire on exactly the
            # step whose canonical hash was materialized above and thread that
            # hash in, so the checkpoint's state_hash.txt matches this step's
            # state_hashes.jsonl entry exactly (or, off-cadence, the
            # side-link chained from the running chain). Historically the final
            # / emergency saves fired OUTSIDE the step and recomputed the
            # hash post-zero_grad — double-linked, grad-less, unreproducible.
            # Runs BEFORE the metrics call so ``step_wall_time_ms`` below
            # covers the save.
            ckpt_dt = 0.0
            if _ckpt_due or _exit_due or (
                _final_step and state.consumed_tokens > last_ckpt_at
            ):
                t_ckpt = time.perf_counter()
                _save_checkpoint(ckpt_state_hash)
                # ckpt.garbage_collect()
                ckpt_dt = time.perf_counter() - t_ckpt
                last_ckpt_at = state.consumed_tokens

            # Metrics. ``step_compute_time_ms`` times only the fwd+bwd+clip+opt
            # region (plus the in-loop batch fetch); ``tokens_per_sec``/``mfu``
            # stay defined on it. ``step_wall_time_ms`` is the full iteration
            # wall clock from the same start point, additionally covering the
            # LSQ scale refresh, the canonical state hash (``hash_time_ms``)
            # and the checkpoint save (``ckpt_save_time_ms``) — the sustained
            # rate is tokens/step over THIS, not over the compute time.
            if metrics is not None:
                tps = tokens_per_sec(tokens_this_step, step_dt)
                wall_dt = time.perf_counter() - t_step
                extra: dict[str, object] = {}
                if state_hash_hex is not None:
                    extra["state_hash"] = state_hash_hex
                metrics.log(
                    state.optimizer_step,
                    consumed_tokens=state.consumed_tokens,
                    lr=lr,
                    accum_steps=accum,
                    grad_norm_pre_clip=gn,
                    loss_ce=ce_avg,
                    loss_zloss=zl_avg,
                    perplexity=math.exp(ce_avg),
                    # Global-batch loss from the deterministic fold above. New
                    # keys (not a redefinition of ``loss_ce``) so mid-run curves
                    # keep their meaning; ``loss_ce`` stays rank-local.
                    loss_ce_global=ce_global,
                    loss_zloss_global=zl_global,
                    perplexity_global=math.exp(ce_global),
                    tokens_per_sec=tps,
                    tokens_per_sec_per_gpu=tps / max(world_size(), 1),
                    mfu=mfu(
                        flops_per_token=flops_per_token,
                        tokens_per_second=tps,
                        # All accelerators incl. TP ranks — see mfu() docstring.
                        n_gpus=max(world_size(), 1),
                    ),
                    step_compute_time_ms=step_dt * 1000.0,
                    step_wall_time_ms=wall_dt * 1000.0,
                    hash_time_ms=hash_dt * 1000.0,
                    ckpt_save_time_ms=ckpt_dt * 1000.0,
                    skipped_steps_total=state.skipped_step_count,
                    **phase_ms,
                    **replicate_ms,
                    **extra,
                )

            # Gradients were kept alive through the canonical hash + the
            # checkpoint (both hash them); clear them now, before the next
            # step's backward accumulates.
            optimizer.zero_grad(set_to_none=True)

            if _exit_due:
                LOG.warning(
                    "emergency exit: checkpoint saved at step=%d "
                    "consumed_tokens=%d; exiting",
                    state.optimizer_step, state.consumed_tokens,
                )
                # The ``finally`` block below waits for the async-save
                # future, closes wandb, and tears down NCCL — so a plain
                # raise here cleans up everything without duplicating logic.
                raise SystemExit(0)

        # No post-loop final save: the run's last checkpoint was written
        # in-loop by the ``_final_step`` arm, which stamps the step's
        # canonical hash. Saving here — after zero_grad, after the chain
        # advanced — is exactly what produced the 1B run's unreproducible
        # step-80957 state_hash.txt.

    except Halt as h:
        LOG.error("HALT: %s", h)
        if metrics is not None:
            metrics.log(state.optimizer_step, halt=str(h))
    finally:
        # Close metrics (flushes JSONL + finishes the W&B run).
        if metrics is not None:
            metrics.close()
        if state_hash_fh is not None:
            try:
                state_hash_fh.close()
            except Exception:
                pass
        if grad_log_fh is not None:
            try:
                grad_log_fh.close()
            except Exception:
                pass
        # Tear down the NCCL process group cleanly.
        shutdown_distributed()
        elapsed = time.time() - started_at
        LOG.info(
            "training stopped: tokens=%d steps=%d skips=%d elapsed_s=%.1f",
            state.consumed_tokens, state.optimizer_step,
            state.skipped_step_count, elapsed,
        )
