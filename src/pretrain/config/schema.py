"""Pydantic schemas for the resolved Hydra config.

The schema is the single source of truth for every hyperparameter. Code
reads from a `RootConfig` instance; nothing else has its own defaults.
A typo in YAML fails here, not at step 4000.

See plan/02_repo_layout.md §2 for the design rationale.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    # Native wheels expose methods as Cython functions. Treat them as methods,
    # just as Pydantic already treats ordinary Python functions, not fields.
    model_config = ConfigDict(
        extra="forbid", frozen=False, ignored_types=(type(lambda: None),)
    )


class InitConfig(_Base):
    # Every parameter is drawn from a mean-0 truncated normal with this std,
    # with NO depth/width scaling — the OLMo 2 (arXiv:2501.00656 §3.2) recipe,
    # which found a flat 0.02 more stable than the GPT-NeoX/Zhang-2019 scaled
    # init (later layers shrunk by 1/sqrt(2*n_layers)) it supersedes.
    std: float = 0.02


class ZLossConfig(_Base):
    enabled: bool = True
    coeff: float = 1.0e-5


class ModuleSelection(_Base):
    """Names resolved through registries — see pretrain.model.registry."""

    attention: str = "gqa_qknorm_repop"
    ffn: str = "swiglu_repop"
    norm: str = "rmsnorm_repop"
    rope: str = "rope_default"
    embedding: str = "untied_repop"


class QATConfig(_Base):
    """int8 quantization-aware training on the attention + FFN Linear
    projections. Embedding and LM head stay full precision.

    Parallelism (see the guard in ``pretrain.parallel.parallelize_llama3_repop``):
      - ``method="lsq"`` runs under pure FSDP2 / HSDP — weight and the ``[out]``
        ``weight_scale`` shard on the same dim-0 axis, FSDP all-gathers both
        before forward, and the scale gradient reduces over the un-sharded K
        axis. BFR byte-equality then holds at a fixed ``(world_size, mesh, NCCL
        config)``, not across device counts.
      - ``method="absmax"`` stays single-device only (per-channel amax over the
        sharded output axis is not yet validated under FSDP2).
      - Neither method runs under tensor parallel: the repop ``local_map``
        wrapper bypasses the quant forward.

    ``method`` selects the QAT layer:
      - ``"lsq"`` (default): ``repop.qat.LSQQuantizedLinear`` — Learned Step
        Size quantization. The per-channel weight scale is a learnable
        Parameter (self-initialized from weight stats), which is designed for
        from-scratch QAT and matches the bench's BFR reference path. Always
        per-channel (``per_channel_w`` is ignored).
      - ``"absmax"``: ``repop.qat.QuantizedLinear`` — fixed absmax scale
        computed from the weights (honors ``per_channel_w``).
    Both are cross-device reproducible (BFR) and expose a ``.weight``
    Parameter, so weight init and the optimizer param-group split are
    unchanged by the choice.
    """

    enabled: bool = False
    method: Literal["lsq", "absmax"] = "lsq"
    weight_bits: int = 8
    act_bits: int = 8
    per_channel_w: bool = True
    # Keep the first N transformer blocks' linears in bf16 (no QAT). The
    # int8-weight STE grad-blowup is concentrated in the early blocks;
    # exempting them is a mixed-precision fix that
    # preserves int8 throughput on the remaining N_layers-N blocks. 0 = all int8.
    exempt_first_n_blocks: int = 0

    # Periodically re-pin every LSQ layer's per-channel ``weight_scale`` to the
    # LSQ-optimal ``2*mean|w_row|/sqrt(qmax)`` of its CURRENT weights, every N
    # optimizer steps (0 disables). In from-scratch int8-weight QAT the learned
    # scale decouples from the weights — Adam drifts the tiny scale-gradient
    # upward, collapsing the weights into ~1 bit of the int8 grid until the STE
    # weight-grad detonates the grad-norm. Re-pinning holds the quant SNR near
    # optimal at no matmul/inference cost (scale frozen at deploy). The scale
    # drifts slowly, so N~50-100 holds SNR with negligible step-time cost
    # (calls ``repop.qat.lsq.refresh_lsq_weight_scales`` between optimizer
    # steps); ``method="lsq"`` only — a no-op under "absmax".
    #
    # AUDIT/BFR: the refresh reduces via ``repop.ops.sum_dim`` over the
    # un-sharded K axis, so it is byte-identical cross-RANK on the same arch and
    # ``audit_replay`` reproduces it (the call is mirrored there). It is NOT yet
    # guaranteed byte-identical cross-ARCH (``sum_dim`` drifts ~1 ULP CPU<->GPU
    # at large K, which can flip a round() at a quant boundary) — keep this 0
    # for cross-arch-reproduced audit runs until the chunked-HFMA2 deterministic
    # sum lands. Default 0 (opt-in, BFR-continuity preserving).
    scale_refresh_every_n_steps: int = 0
    # bf16 warm-start: run the first N optimizer steps unquantized (forward
    # routes through the plain repop linear kernel, bitwise-identical to a
    # non-QAT layer), then enable LSQ quantization at step N. Removes int8
    # STE noise from the fragile warmup phase and the init-time quantization
    # pathology class (the 20260703 zero-q-grad original sin). The every-step
    # scale refresh keeps weight_scale pinned throughout, so the first
    # quantized step uses an optimal grid. Pure function of (step, config) —
    # set identically each step by the loop and the audit; no segment
    # boundary. 0 = quantize from step 0 (legacy).
    enable_at_step: int = 0

    @model_validator(mode="after")
    def _validate(self) -> "QATConfig":
        if self.enable_at_step < 0:
            raise ValueError(
                f"qat.enable_at_step must be ≥ 0 (0 = quantize from step 0); "
                f"got {self.enable_at_step}"
            )
        if self.scale_refresh_every_n_steps < 0:
            raise ValueError(
                f"qat.scale_refresh_every_n_steps must be ≥ 0 (0 disables); "
                f"got {self.scale_refresh_every_n_steps}"
            )
        return self


class ModelConfig(_Base):
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_intermediate: int
    vocab_size: int
    max_seq_len_pretrain: int
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1.0e-5
    qk_norm: bool = True
    # Learnable per-dim γ on the QK norms. True = legacy (the 20260703 runs);
    # False = gain-free QK-norm (fresh-run default direction): deletes the
    # unbounded attention-temperature axis that drove the entropy collapse.
    # Kept default-True so old runs' config_resolved parses/rebuilds byte-
    # compatible models for the audit.
    qk_norm_gain: bool = True
    # Hybrid sliding-window attention via repop causal_flash_attention.
    # ``swa_full_every`` layers form a group whose last layer is full causal
    # and the rest use a ``swa_window``-key sliding window. Default 5 => 4
    # sliding-window layers per 1 full-causal (causal_flash_attn) layer (4:1).
    # Set swa_full_every=1 to make every layer full causal. ``swa_window`` is
    # in keys and must be a multiple of the flash block size (32).
    swa_window: int = 512
    swa_full_every: int = 5
    # Route attention through repop's int8-PV flash kernel
    # (``int8pv_causal_flash_attention``) instead of the bf16 FMA flash
    # (``causal_flash_attention``). Moves the P@V matmul onto int8 tensor
    # cores — the bench's ``int8_pv_bfr`` path — and stays BFR (forward is
    # byte-equal per device; STE float-shadow backward). Quantizes attention
    # P@V to int8, so it changes convergence vs the bf16 path; set False to
    # fall back to the bf16 flash kernel.
    attn_int8_pv: bool = True
    # Optional RMSNorm on the embedding output before the residual stream.
    # Bounds the input magnitude block 0 (and the embedding gradient) sees —
    # the fix for the LSQ-int8-QAT × embedding-growth grad-norm blow-up.
    # Default off for back-compat / BFR continuity;
    # turn on for QAT runs. Pre-block norms already normalize each block's
    # input direction, but not the residual *magnitude* at the network bottom.
    emb_norm: bool = False
    init: InitConfig = Field(default_factory=InitConfig)
    z_loss: ZLossConfig = Field(default_factory=ZLossConfig)
    modules: ModuleSelection = Field(default_factory=ModuleSelection)
    qat: QATConfig = Field(default_factory=QATConfig)

    @model_validator(mode="after")
    def _validate(self) -> "ModelConfig":
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model ({self.d_model}) must equal n_heads * head_dim "
                f"({self.n_heads} * {self.head_dim})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.vocab_size % 128 != 0:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) should be a multiple of 128 "
                f"(tensor-core friendliness)"
            )
        if self.swa_window < 0 or self.swa_window % 32 != 0:
            raise ValueError(
                f"swa_window ({self.swa_window}) must be >= 0 and a multiple "
                f"of the flash block size (32)"
            )
        if self.swa_full_every < 1:
            raise ValueError(
                f"swa_full_every ({self.swa_full_every}) must be >= 1"
            )
        return self


class DataSourceConfig(_Base):
    name: str
    path: str            # directory containing .bin/.idx shards (or manifest)
    weight: float        # mix sampling weight (relative)


class DataConfig(_Base):
    sources: list[DataSourceConfig]
    seq_len: int = 4096
    document_separator_id: int = 0   # EOS — set by tokenizer
    pack_strategy: Literal["concat_eos", "varlen"] = "concat_eos"
    # If True, each source's `weight` is a *token-share* target (fraction
    # of training tokens to draw from this source). The loader converts
    # to per-doc sampling probabilities using each manifest's average
    # tokens-per-document, so the resulting empirical token-mix matches
    # `weight`. If False (default, kept for back-compat), `weight` is the
    # per-document probability directly — meaning sources with shorter
    # docs (e.g. Stack at ~500 tok/doc) end up with a much smaller token
    # share than their `weight` would suggest. New configs should set
    # this True and express recipe percentages as token-shares.
    weights_are_token_shares: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "DataConfig":
        if not self.sources:
            raise ValueError("at least one data source required")
        weights = [s.weight for s in self.sources]
        total = sum(weights)
        if total <= 0:
            raise ValueError("source weights must sum to a positive number")
        if any(w < 0 for w in weights):
            raise ValueError("source weights must be non-negative")
        return self

    def normalised_weights(self) -> list[float]:
        total = sum(s.weight for s in self.sources)
        return [s.weight / total for s in self.sources]


class OptimConfig(_Base):
    name: Literal["adamw", "adamw_repop", "muon_hybrid"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.1
    fused: bool = True
    no_decay_param_names: list[str] = Field(
        default_factory=lambda: [".bias", "norm.weight", ".rope_"]
    )
    # peak_lr is the schedule's peak; lives here for readability.
    peak_lr: float = 3.0e-4


class ScheduleConfig(_Base):
    name: Literal["cosine", "wsd", "linear_anneal"] = "cosine"
    warmup_steps: int = 2000
    min_lr_frac: float = 0.10              # cosine/linear_anneal endpoint relative to peak
    # WSD-only fields; ignored by cosine.
    wsd_decay_start_frac: float = 0.90     # decay tail begins at 90% of total tokens
    wsd_decay_to_frac: float = 0.10
    # linear_anneal-only: ABSOLUTE token count where the linear decay to
    # ``min_lr_frac * peak`` begins — the midtraining branch point (OLMo 2
    # §4.1). Set to the pretrain checkpoint's consumed_tokens; the resume
    # guard (pretrain.train.midtrain) enforces the match. 0 = decay from
    # the end of warmup (cold-start use).
    anneal_start_tokens: int = 0
    # Phase-boundary LR re-warmup. When resuming with a RESET optimizer
    # (``train.resume_reset_optimizer``), the moments start cold, so we linearly
    # ramp the LR from 0 to the schedule value over this many tokens measured
    # FROM the resume point — the standard fix for the cold-moment instability of
    # a reset (SPAM 2501.06842 used ~150 steps). 0 disables. Active on a
    # reset-optimizer resume, or on ANY resume when ``rewarm_on_resume`` is set.
    rewarm_tokens: int = 0
    # Anchor the LR re-warm on a normal (moments-kept) resume too. For forks
    # that wake previously-frozen parameters (the 20260703 un-freeze), a
    # full-LR first step lands on renegotiating tensors; ramping over
    # ``rewarm_tokens`` softens the transient. The anchor is RECORDED in
    # checkpoint meta (``rewarm_anchor_tokens``) so the audit reproduces the
    # exact LR sequence for any interval, and a crash-resume of the forked run
    # INHERITS the original anchor instead of spuriously re-warming.
    rewarm_on_resume: bool = False


class SpikeConfig(_Base):
    grad_norm_threshold: float = 5.0
    # The three halt fields below must satisfy ``_validate``: a default that
    # cannot halt would make any partial ``spike:`` block (or a bare
    # ``SpikeConfig()``) unloadable, and silently un-haltable if it weren't.
    skip_steps_on_spike: int = 5
    skips_in_window_to_halt: int = 5
    halt_window_steps: int = 100
    # Don't enforce until this optimizer step. Init-phase grad norms are
    # naturally large for a fresh 128k-vocab model; the protocol targets
    # post-warmup divergence. Default 0 = enforce from step 1 (legacy).
    start_step: int = 0

    @model_validator(mode="after")
    def _validate(self) -> "SpikeConfig":
        # ``spike_protocol.record_skip`` registers only FRESH trigger events
        # against the halt window; the ``skip_steps_on_spike - 1`` cooldown
        # steps that follow one just decrement the cooldown. Fresh events are
        # therefore forced ``skip_steps_on_spike`` apart, so the oldest of
        # ``skips_in_window_to_halt`` of them sits
        # ``(skips_in_window_to_halt - 1) * skip_steps_on_spike`` steps behind
        # the newest and must still fall inside ``halt_window_steps`` — the
        # prune keeps events with ``step >= newest - halt_window_steps``.
        # Violate that and the deque can never reach the halt count: an
        # unstable run skips every step forever, never steps the optimizer,
        # and never pages on-call.
        spacing = (self.skips_in_window_to_halt - 1) * self.skip_steps_on_spike
        if spacing > self.halt_window_steps:
            raise ValueError(
                f"spike halt is mathematically impossible: "
                f"(skips_in_window_to_halt-1) * skip_steps_on_spike = "
                f"{spacing} > halt_window_steps = {self.halt_window_steps}; "
                f"the halt deque can never hold {self.skips_in_window_to_halt} "
                f"fresh events, so the run skips forever instead of halting. "
                f"Raise spike.halt_window_steps to at least {spacing}, or "
                f"lower spike.skip_steps_on_spike / "
                f"spike.skips_in_window_to_halt"
            )
        return self


class GlobalBatchTokens(_Base):
    warmup: int = 1_048_576
    main: int = 2_097_152
    late: int = 4_194_304


class StateHashConfig(_Base):
    """Periodic bitwise state hashing for cross-run divergence detection.

    When enabled, the loop emits a topology-invariant blake2b digest on
    every Nth optimizer step and at every checkpoint. The digest covers
    model weights + AdamW moments + (optionally) post-clip gradients +
    (optionally) a cross-rank running digest of every batch consumed so
    far. Digests are chained step-to-step so any divergence is visible
    at the first divergent step.

    Compare two runs::

        diff runs/A/logs/state_hashes.jsonl runs/B/logs/state_hashes.jsonl

    For **cross-topology** comparison (e.g. FSDP-only vs HSDP+TP at the
    same checkpoint), flip ``include_batch`` off — the batch term
    depends on ``dp_world_size`` striding and is *not* cross-topology
    invariant. Weights / moments / grads stay invariant via
    ``full_tensor()``.
    """

    # 0 disables hashing. ≥1 hashes on steps N, 2N, 3N, … (using the
    # post-increment step number). Cost per hashed step is one
    # ``full_tensor()`` all-gather per parameter + moment (+ grad if
    # ``include_grads``) — leave off, or use a generous N, on hot paths.
    every_n_steps: int = 0

    # Include post-clip gradients in the digest. Off saves roughly one
    # extra all-gather per parameter at hash time. Leaving on catches
    # divergence introduced inside a single backward pass.
    include_grads: bool = True

    # Include a cross-rank running digest of all batches consumed so
    # far. Flip OFF for cross-topology comparison (different
    # ``dp_world_size`` ⇒ different per-dp_rank document striding ⇒
    # different digests by design). Leaving ON is the right default for
    # within-topology checks — catches data-order divergence.
    include_batch: bool = True

    # Emit a step-0 "init" checkpoint + a weights-only state hash on cold
    # start, BEFORE the first optimizer step. This makes the model
    # initialization itself auditable: ``pretrain.cli.audit_replay
    # --from-init`` reconstructs ``build_model`` + ``init_weights(seed)`` on
    # one (any) device and verifies it reproduces this hash bit-for-bit —
    # only meaningful now that init is device-independent (repop trunc_normal).
    # The init hash is standalone (prev_hash=None, weights only); it does NOT
    # chain into the periodic step-N hashes, so existing chains are unchanged.
    at_init: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "StateHashConfig":
        if self.every_n_steps < 0:
            raise ValueError(
                f"state_hash.every_n_steps must be ≥ 0 (0 disables); "
                f"got {self.every_n_steps}"
            )
        return self


class GradNormLogConfig(_Base):
    """Per-parameter / per-layer PRE-clip gradient-norm logging — a
    diagnostic for isolating which parameters drive grad-norm spikes
    (e.g. LSQ ``weight_scale`` vs attention vs FFN weights).

    When enabled the loop computes the global L2 norm of every parameter's
    gradient (the same topology-aware fold the global clip uses) BEFORE the
    clip rescales them, buckets them by category + layer index, and appends
    one JSON line per logged step to ``path`` under the run dir. Rank-0 only
    writes, but the norm computation is collective so every rank calls it.

    Cost per logged step: one tiny scalar all-reduce per parameter (~170 at
    1B). Negligible at a generous ``every_n_steps``; leave at 0 on hot runs.
    """

    every_n_steps: int = 0  # 0 disables; ≥1 logs on steps N, 2N, …
    path: str = "logs/grad_norms.jsonl"
    # How many largest-norm individual params to record per logged step.
    top_k: int = 12
    # Weight-side diagnostics cadence: on steps where BOTH this and
    # ``every_n_steps`` fire, the logged row also carries per-parameter WEIGHT
    # L2 norms and the per-layer max |γ_q ⊙ γ_k| QK-norm gain product. The
    # weight norms let a dashboard detect a tensor whose ‖w‖ tracks the pure
    # weight-decay law w₀·e^(−λ∫lr) — i.e. a tensor receiving no effective
    # gradient (the run-20260703 q-side freeze); the gain product is the
    # attention-logit temperature (entropy-collapse gauge). Same collective
    # cost profile as the grad norms (~one tiny fold per param). Use a
    # multiple of ``every_n_steps`` (e.g. 100); 0 disables.
    weight_stats_every_n_steps: int = 0

    @model_validator(mode="after")
    def _validate(self) -> "GradNormLogConfig":
        if self.every_n_steps < 0:
            raise ValueError(
                f"grad_norm_log.every_n_steps must be ≥ 0 (0 disables); "
                f"got {self.every_n_steps}"
            )
        if self.weight_stats_every_n_steps < 0:
            raise ValueError(
                f"grad_norm_log.weight_stats_every_n_steps must be ≥ 0 "
                f"(0 disables); got {self.weight_stats_every_n_steps}"
            )
        return self


class TrainConfig(_Base):
    total_tokens: int = 150_000_000_000
    seq_len: int = 4096
    micro_batch_size: int = 4
    global_batch_tokens: GlobalBatchTokens = Field(default_factory=GlobalBatchTokens)
    warmup_to_main_at_tokens: int = 4_000_000_000
    main_to_late_at_tokens: int = 140_000_000_000
    ckpt_every_tokens: int = 5_000_000_000
    # When > 0, checkpoints save on an OPTIMIZER-STEP cadence
    # (``optimizer_step % ckpt_every_steps == 0``) instead of the token cadence
    # above. Set it equal to ``state_hash.every_n_steps`` so the two always
    # coincide: the post-step canonical hash materializes the full model
    # (a collective ``full_tensor()``) exactly once and the checkpoint reuses it,
    # instead of the token cadence landing on an off-hash step (which happens
    # once the batch size differs by phase) and forcing a second materialization.
    # ``ckpt_every_tokens`` is still used as the audit/fetch default-interval hint
    # and as the fallback cadence when this is 0.
    ckpt_every_steps: int = 0
    # Resume with a FRESH optimizer: load only the model weights from
    # --resume-from and start optimizer moments from zero (step counter still
    # comes from the checkpoint meta, so the LR schedule continues by tokens).
    # Use at a phase boundary that changes optimizer grouping/hyperparameters
    # (e.g. moving the embedding to no-decay), where reusing the checkpoint's
    # positionally-keyed moments would misalign. Pair with schedule.rewarm_tokens
    # to ramp the LR back up over the cold-moment window. Default false = normal
    # resume (model + optimizer).
    resume_reset_optimizer: bool = False
    spike: SpikeConfig = Field(default_factory=SpikeConfig)
    grad_clip: float = 1.0
    # QK-norm gain controls (pretrain.train.qk_gain_control; mirrored in
    # audit_replay). The γq⊙γk product is a learned attention temperature with
    # no restoring force — a July 2026 1B QAT run rode it into entropy
    # collapse (product 19.5 → 28+, loss up while grads calmed).
    # Hard cap on |γ| for every q_norm/k_norm gain, applied in place after the
    # optimizer step (runs on spike-skips too; idempotent). 0 disables.
    qk_gain_clamp: float = 0.0
    # Staged wake-up: zero the q_norm gain GRADIENTS (post-clip, pre-step)
    # while optimizer_step < this, so wq re-inflates against a fixed query
    # temperature before the gains are released. 0 disables.
    qk_freeze_q_gains_until_step: int = 0
    state_hash: StateHashConfig = Field(default_factory=StateHashConfig)
    grad_norm_log: GradNormLogConfig = Field(default_factory=GradNormLogConfig)
    # Dead-tensor assertion: hard-fail if any parameter's gradient norm is
    # EXACTLY zero for this many consecutive logged steps (requires
    # grad_norm_log.every_n_steps > 0). Guard against the 20260703 original
    # sin — 72 tensors trained zero for 50k steps in silence. 0 disables.
    dead_tensor_assert_steps: int = 0


class LoggingConfig(_Base):
    wandb_project: str = "pretrain-8b"
    metrics_jsonl_path: str = "logs/metrics.jsonl"


class RunConfig(_Base):
    run_id: str = ""
    seed: int = 42
    output_dir: str = "runs"
    nproc_per_node: int = 8
    # FSDP shard dimension. -1 → world_size / dp_replicate_size.
    dp_shard_size: int = -1
    # HSDP-style replica dimension. 1 disables replication (pure FSDP).
    dp_replicate_size: int = 1
    # "fsdp" → FSDP2 fully_shard across the DP mesh (param/grad/optim sharded).
    # "ddp"  → pure data parallel; full model replicated on each rank. Only
    # viable for models that fit (params + grads + optim state + activations)
    # on a single device.
    parallel: Literal["fsdp", "ddp"] = "fsdp"
    # Gradient reduction across the DP mesh.
    #   "deterministic_allgather" (DEFAULT) → fixed ascending-rank reduce-scatter
    #                             + (under HSDP) a fixed-order cross-replica
    #                             all-reduce (see pretrain.parallel.
    #                             deterministic_reduce). The reduced gradient is
    #                             reproducible regardless of device count, so any
    #                             checkpoint can be advanced to the next one on a
    #                             single device and match bit-for-bit
    #                             (pretrain.cli.audit_replay). Costs an all-gather
    #                             vs a reduce-scatter; also drives the canonical,
    #                             world-size-independent data stream.
    #   "nccl"                  → FSDP2's default NCCL reduce-scatter (faster, but
    #                             the summation order is topology-dependent, so
    #                             results are bitwise-stable only at a fixed world
    #                             size). Opt out here for throughput-only runs.
    reduction_mode: Literal["nccl", "deterministic_allgather"] = "deterministic_allgather"
    # When false, disables autocast and the FSDP2 MixedPrecisionPolicy so
    # params, activations, gradients, and reductions all stay in fp32.
    # Optimizer state (AdamW moments) is fp32 either way.
    mixed_precision: bool = True
    activation_checkpoint: bool = True
    ac_every_other_block: bool = False    # tuned during 1B proxy
    # Path to the tokenizer.json the shards were produced with. Used by
    # the in-loop DCLM-CORE eval runner to wrap the model in lm-eval's
    # interface. Empty disables the DCLM eval (loop still emits an eval
    # cadence marker so dashboards see the rhythm).
    tokenizer_path: str = ""


class RootConfig(_Base):
    """Top-level resolved config. One of these is built by `load_config`."""

    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    schedule: ScheduleConfig
    train: TrainConfig
    run: RunConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _validate_seq_len(self) -> "RootConfig":
        # Three places hold a sequence length: the data loader packs to
        # ``data.seq_len``, the trainer accounts tokens-per-step against
        # ``train.seq_len``, and the model's RoPE table is built for
        # ``model.max_seq_len_pretrain``. A drift between any pair fails
        # at runtime several minutes into training; assert agreement here.
        if self.data.seq_len != self.train.seq_len:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) must equal "
                f"train.seq_len ({self.train.seq_len}) — the loader and the "
                f"per-step token accountant must agree"
            )
        if self.data.seq_len > self.model.max_seq_len_pretrain:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) exceeds "
                f"model.max_seq_len_pretrain ({self.model.max_seq_len_pretrain}) "
                f"— RoPE tables are pre-built for max_seq_len_pretrain"
            )
        return self
