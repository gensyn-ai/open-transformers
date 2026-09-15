"""Distributed checkpointing via ``torch.distributed.checkpoint`` (DCP).

We save:
  - sharded model state
  - sharded optimizer state
  - mix-sampler state (per-source consumed_documents + epoch)
  - scheduler state (re-derived from consumed_tokens, but stored too for
    safety)
  - run metadata: git SHA + diff, resolved config, tokenizer hash,
    container digest, restart count

Save mode: synchronous ``dcp.save`` on a dedicated gloo process group
(matching torchtitan's PG choice). The PG isolates DCP's metadata
collectives from the main training loop's NCCL ops on the default PG.
``async_save`` was used previously but the background-thread window
complicated diagnosing the missing-shards bug, and ``DefaultStager`` /
``close()`` machinery for clean async shutdown only exists on torch
≥ 2.9 — out of reach until the cluster's CUDA-12.2 drivers are
upgraded.

For CPU smoke tests we fall back to ``torch.save`` because DCP requires
distributed init.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import torch

from pretrain.data.global_stream import GlobalStreamState
from pretrain.data.mix_sampler import MixSamplerState
from pretrain.train.state_hash import compute_state_hash

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# DCP metadata trust boundary
# --------------------------------------------------------------------------- #
# ``dcp.load`` reads ``<ckpt>/dcp/.metadata`` with a plain ``pickle.load``
# (torch's ``FileSystemReader.read_metadata``), which the ``weights_only=True``
# hardening on our own ``torch.load`` call sites cannot reach: a doctored
# ``.metadata`` executes the checkpoint author's code in the loader's process
# the moment the directory is opened. A checkpoint directory is untrusted
# input — ``audit_replay`` is designed to be pointed at a dir produced by
# someone else — so callers first unpickle the metadata with an allowlisting
# ``Unpickler`` that admits only the globals a genuine DCP metadata blob
# contains. Anything else hard-fails BEFORE ``dcp.load``.
#
# The allowlist is NOT a namespace prefix. CPython's unpickler resolves a
# dotted ``name`` by walking attributes (``_getattribute``), and submodules
# under ``torch.distributed.checkpoint`` re-export arbitrary modules at module
# scope (e.g. ``...format_utils.os`` IS the ``os`` module), so a
# ``GLOBAL torch.distributed.checkpoint.format_utils / os.system`` payload
# would resolve to ``os.system`` and run with attacker args — inside the
# validator itself. The rules below close that:
#   * any ``name`` containing "." is rejected outright (kills the attribute
#     walk);
#   * in the DCP namespace only CLASSES whose own ``__module__`` is also in
#     that namespace are admitted — this admits every metadata dataclass
#     (drift-tolerant across torch versions) while rejecting module-scope
#     functions (``dcp_to_torch_save``, ``load``) and RE-exported dangerous
#     classes (``subprocess.Popen``'s ``__module__`` is ``subprocess``, not
#     the DCP namespace);
#   * a tiny exact ``(module, name)`` set covers the non-class value
#     constructors a real metadata references.
_DCP_NAMESPACE = "torch.distributed.checkpoint"

_DCP_META_ALLOWED_EXACT = frozenset({
    # torch.layout values pickle via this reconstructor (torch.strided, ...).
    ("torch.serialization", "_get_layout"),
    # Present in some torch versions' Metadata payloads; harmless container.
    ("collections", "OrderedDict"),
    # ``_StorageInfo.relative_path`` round-trips as a pathlib object.
    ("pathlib", "Path"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    ("pathlib", "PurePath"),
    ("pathlib", "PurePosixPath"),
    ("pathlib", "PureWindowsPath"),
})


class _DCPMetadataUnpickler(pickle.Unpickler):
    """Unpickler for ``dcp/.metadata`` that rejects any global outside the
    known DCP metadata schema (Metadata / TensorStorageMetadata / ... and the
    torch dtype/layout/Size values they carry). See the module comment above
    for why the DCP namespace is matched by class-identity, not by prefix."""

    def find_class(self, module: str, name: str):
        # A dotted name is always an attribute-walk attempt (real metadata
        # globals are all bare top-level names) — refuse before resolving it.
        if "." in name:
            self._reject(module, name)
        if module == _DCP_NAMESPACE or module.startswith(_DCP_NAMESPACE + "."):
            obj = super().find_class(module, name)
            if isinstance(obj, type) and getattr(obj, "__module__", "").startswith(
                _DCP_NAMESPACE
            ):
                return obj
            self._reject(module, name)  # a function or a re-exported foreign class
        if (module, name) in _DCP_META_ALLOWED_EXACT:
            return super().find_class(module, name)
        if module == "torch":
            obj = getattr(torch, name, None)
            if name == "Size" or isinstance(
                obj, (torch.dtype, torch.layout, torch.memory_format)
            ):
                return super().find_class(module, name)
        self._reject(module, name)

    @staticmethod
    def _reject(module: str, name: str):
        raise pickle.UnpicklingError(
            f"dcp metadata references {module}.{name}, which is outside the "
            "allowlist of globals a genuine DCP metadata file contains — "
            "refusing to unpickle it (checkpoint directories are untrusted "
            "input; a doctored .metadata would otherwise execute arbitrary "
            "code via torch's pickle-based metadata reader)."
        )


def _validate_dcp_metadata(dcp_dir: Path) -> None:
    """Unpickle every metadata file under ``dcp_dir`` with the allowlisting
    unpickler, purely for validation — ``dcp.load`` then re-reads it through
    torch's own reader. Raises ``pickle.UnpicklingError`` on the first
    disallowed global. Metadata files are tiny, so the double read is free
    next to the shard I/O. No-op when ``dcp_dir`` is absent — the caller's
    ``dcp.load`` then fails on its own terms."""
    if not dcp_dir.is_dir():
        return
    meta_files = sorted(
        p
        for p in dcp_dir.iterdir()
        if p.is_file() and (p.name == ".metadata" or p.name.endswith(".metadata"))
    )
    for mf in meta_files:
        with open(mf, "rb") as f:
            _DCPMetadataUnpickler(f).load()


def load_dcp_validated(state: dict, dcp_dir: Path | str, **dcp_load_kwargs) -> None:
    """``dcp.load`` with the untrusted-input gate applied first. The single
    entry point every checkpoint reader must use, so a new call site cannot
    forget the ``.metadata`` validation (the gate was previously copy-pasted
    at each ``dcp.load``)."""
    import torch.distributed.checkpoint as dcp

    dcp_dir = Path(dcp_dir)
    _validate_dcp_metadata(dcp_dir)
    dcp.load(state, checkpoint_id=str(dcp_dir), **dcp_load_kwargs)

# meta.json keys that were removed from CheckpointMeta but appear in metas
# written by pre-removal code. Dropped by ``CheckpointMeta.from_dict``; only
# keys the reader never acts on belong here — an unknown key NOT in this list
# still fails construction loudly (a machine-written meta with a genuinely
# unknown key means reader and writer disagree about the schema).
_RETIRED_META_KEYS = frozenset({
    # AdaGC hyperparameters (clipper retired 2026-07-21, fields deleted
    # 2026-08-04). Inert for clip_algo="global" runs; genuinely-AdaGC
    # checkpoints are refused by audit_replay's clip_algo gate instead.
    "adagc_lambda_rel",
    "adagc_lambda_abs",
    "adagc_beta",
    "adagc_t_start",
    "adagc_eps",
    "adagc_gamma_min",
})


@dataclasses.dataclass
class CheckpointMeta:
    consumed_tokens: int
    step: int
    git_sha: str
    config_resolved: str          # YAML
    tokenizer_hash: str
    container_digest: str
    restart_count: int = 0
    # Latest value of the running state-hash chain at save time. Restored
    # into the loop's ``chained_hash`` on resume so the post-resume hash
    # chains from the pre-resume one — without this, the chain breaks at
    # every resume and cross-run state-hash compares stop being meaningful.
    chained_hash: str | None = None
    # ---- Run descriptor (topology-invariant audit) -------------------------- #
    # Populated for runs using the canonical data stream + deterministic
    # reduction so a single-device audit can reproduce the next checkpoint
    # bitwise. All default to a benign value so older checkpoints still load.
    reduction_mode: str = "nccl"
    # dp-unique-token degree the run executed at (the N the audit emulates).
    dp_world_size: int = 0
    # HSDP mesh: dp_world_size == dp_replicate * dp_shard. The audit folds
    # per-rank partials shard-inner (ascending, reduce-scatter) then
    # replicate-outer to match the two-level reduction.
    dp_replicate: int = 1
    dp_shard: int = 0
    # Which cross-replica (replicate-outer) all-reduce the run used, so the
    # audit replays the matching combination order. "recursive_doubling" is the
    # current default (balanced adjacent-pair tree); checkpoints predating this
    # field were trained with the legacy "ascending_allgather" and the audit
    # treats a missing field as such. Only meaningful when dp_replicate > 1.
    replicate_reduce_algo: str = "ascending_allgather"
    # Grad-norm fold used for clipping + the spike-protocol trigger. The current
    # auditable path is the fixed ascending-shard sum-of-squares fold
    # (deterministic_reduce.GRAD_NORM_ALGO); checkpoints predating this field used
    # the non-deterministic ``DTensor.full_tensor()`` norm, recorded as
    # "full_tensor" so the audit keeps its legacy norm path for them.
    grad_norm_algo: str = "full_tensor"
    # Gradient-clipping algorithm. "global" = single global clip coefficient
    # (train.grad_clip), the stateless deterministic global-norm clip
    # (pretrain.train.global_clip). Recorded so the audit reconstructs the
    # exact clipper the run used (robust to config drift, like repop_env).
    clip_algo: str = "global"
    # LR re-warm anchor (token count of the resume the ramp is measured from);
    # -1 = no re-warm. Recorded so the audit reproduces the exact LR of any
    # interval inside the ramp window, and so a crash-resume of a re-warming
    # run inherits the original anchor instead of re-warming from the crash
    # point. Metas predating the field read as -1 (no ramp) — matching the old
    # code's behaviour for normal resumes.
    rewarm_anchor_tokens: int = -1
    seed: int = 0
    torch_version: str = ""
    numpy_version: str = ""
    # The cross-device-reproducibility kernel/determinism env the run used: the
    # REPOP_* vars present in os.environ (LSQ/int8-PV backward variants, plus any
    # repop default repop materialised into the env) and the cuBLAS/arch contract
    # keys (CUBLAS_WORKSPACE_CONFIG, TORCH_CUDA_ARCH_LIST). Only what was actually
    # set is recorded — vars left at repop's default are NOT fabricated (forcing a
    # guessed value would change behaviour). The audit re-applies these so it
    # dispatches identical kernels — otherwise QAT/int8/cuBLAS paths silently fall
    # back and diverge. Built by loop._capture_repop_env.
    repop_env: dict = dataclasses.field(default_factory=dict)
    # Global packed-window position at save time (== GlobalStreamState.
    # windows_emitted). Cross-checked by the audit against the value it
    # re-derives from (seed, manifests, step).
    windows_emitted: int = 0

    @classmethod
    def from_dict(cls, meta_obj: dict) -> "CheckpointMeta":
        """Build from a meta.json dict, dropping retired keys written by
        older code (``_RETIRED_META_KEYS``). Any other unknown key still
        raises ``TypeError`` — use this instead of ``cls(**meta_obj)``
        whenever the dict was read from disk."""
        return cls(**{k: v for k, v in meta_obj.items() if k not in _RETIRED_META_KEYS})


class Checkpointer:
    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ckpt_pg: Any = None   # dedicated gloo PG for ckpt collectives

    def _get_ckpt_pg(self) -> Any:
        # Dedicated gloo PG so DCP's metadata collectives never share order
        # with the main-thread NCCL collectives on the default PG. Matches
        # torchtitan's pattern. Created lazily because dist may not be
        # initialized at Checkpointer construction time.
        if self._ckpt_pg is not None:
            return self._ckpt_pg
        if not torch.distributed.is_initialized():
            return None
        self._ckpt_pg = torch.distributed.new_group(backend="gloo")
        return self._ckpt_pg

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        sampler_state: MixSamplerState | None,
        meta: CheckpointMeta,
        batch_digest: bytes | None = None,
        *,
        batch_hasher_digest: bytes | None = None,
        spike_state: dict | None = None,
        global_stream_state: GlobalStreamState | None = None,
        state_hash: str | None = None,
    ) -> Path:
        ckpt_dir = self.root / f"step_{step:09d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # state_hash.txt MUST equal the loop's state_hashes.jsonl entry for this
        # step. The loop computes the canonical hash post-step (weights +
        # optimizer state + gradients + running batch digest, chained) while the
        # grads are still live and passes it in here. We write that verbatim — do
        # NOT recompute, both because recomputing post-zero_grad loses the grads
        # and because re-chaining off ``meta.chained_hash`` (already advanced)
        # would double-apply the chain. Fallback (emergency/final saves with no
        # pre-computed hash): compute the same canonical form here; grads may be
        # absent then (hashed as "grad_none"). full_tensor() is collective — must
        # run on every rank, so this is unguarded.
        if state_hash is None:
            state_hash = compute_state_hash(
                model,
                optimizer=optimizer,
                include_grads=True,
                batch_digest=batch_digest,
                prev_hash=meta.chained_hash,
            )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            (ckpt_dir / "state_hash.txt").write_text(state_hash + "\n")
        # Model + optimizer shards through DCP.
        self._dcp_save(ckpt_dir, model, optimizer)
        # Per-rank metadata (sampler / RNG / batch hasher) + rank-0 metas
        # (meta.json / spike_protocol.json) all go through rank 0 to
        # sidestep the multi-pod NFS write-visibility issue. See
        # _gather_and_write_per_rank for details.
        self._gather_and_write_per_rank(
            ckpt_dir,
            sampler_state=sampler_state,
            batch_hasher_digest=batch_hasher_digest,
            spike_state=spike_state,
            meta=meta,
            global_stream_state=global_stream_state,
        )
        # _COMPLETE sentinel written last by rank 0. ``latest()`` skips
        # dirs without it so a partial save (e.g., crash mid-write) is
        # not picked as the resume target.
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            (ckpt_dir / "_COMPLETE").write_text("")
        return ckpt_dir

    @staticmethod
    def _load_loop_extras(ckpt_dir: Path) -> dict:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        out: dict = {"batch_hasher_digest": None, "spike_state": None}
        bh_path = ckpt_dir / f"batch_hasher.rank_{rank}.bin"
        if bh_path.exists():
            out["batch_hasher_digest"] = bh_path.read_bytes()
        sp_path = ckpt_dir / "spike_protocol.json"
        if sp_path.exists():
            out["spike_state"] = json.loads(sp_path.read_text())
        return out

    def load(
        self,
        ckpt_dir: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        optim_state_offload: bool = False,
        model_weights_only: bool = False,
    ) -> tuple[MixSamplerState | GlobalStreamState, CheckpointMeta, dict]:
        """Restore model + optimizer + loop state.

        ``model_weights_only`` (phase-boundary reset): load ONLY the model
        weights from the checkpoint and leave ``optimizer`` at its freshly-built
        state (zero moments). Use when the new phase changes the optimizer's
        param grouping or hyperparameters so the checkpoint's positionally-keyed
        moments can't be reused (see ``TrainConfig.resume_reset_optimizer``).
        The DCP read is planned against only the ``model`` template, so the
        checkpoint's optimizer shards are simply not requested. RNG and loop
        state (consumed_tokens / step / sampler) still load, so data and the
        token-clocked LR schedule continue.

        ``optim_state_offload`` (single-device audit only): the caller has
        pre-populated ``optimizer.state`` with a CPU template, so we skip
        ``_init_optim_state`` (which would materialise every moment on the GPU —
        64 GB for 8B) and skip ``optimizer.load_state_dict`` (which would move
        the loaded moments back onto the param device). DCP fills the CPU
        template tensors in place, leaving the optimizer state on CPU for
        per-parameter streamed stepping. No effect on the training resume path.
        """
        ckpt_dir = Path(ckpt_dir)
        # Untrusted-input gate, OUTSIDE the try below: a doctored dcp/.metadata
        # must hard-fail, not fall through to the fallback.pt branch.
        dcp_dir = ckpt_dir / "dcp"
        if dcp_dir.is_dir():
            _validate_dcp_metadata(dcp_dir)
        # Try DCP first; fall back to torch.save format.
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import _init_optim_state

            # ``dcp.load`` plans the read against keys already present
            # in the supplied template state dict. A freshly-built
            # optimizer hasn't taken a step yet, so
            # ``optimizer.state_dict()`` returns
            # ``{"state": {}, "param_groups": [...]}`` — DCP then
            # silently skips every ``exp_avg``/``exp_avg_sq``/``step``
            # shard in the checkpoint and the optimizer warm-starts
            # from zero moments with ``step=1``, which is exactly the
            # "loss + state-hash diverge at step N+1" symptom on resume.
            #
            # ``_init_optim_state`` populates ``optimizer.state`` by
            # taking one ``step()`` with zero gradients and lr=0 — a
            # no-op on the parameters but enough to materialise every
            # per-param ``exp_avg`` / ``exp_avg_sq`` / ``step`` entry
            # with the right DTensor sharding. After this primer,
            # ``optimizer.state_dict()`` has a proper FSDP-aware
            # template DCP can plan against.
            #
            # We deliberately do NOT use ``get_optimizer_state_dict`` /
            # ``set_optimizer_state_dict`` here: they pull every
            # FSDP-wrapped sub-module through ``FSDP.optim_state_dict``
            # (the FSDP1 API), which reads ``submodule._state_dict_type``
            # set up via ``FSDP.state_dict_type``. Those attributes
            # don't exist in the same way on FSDP2 ``fully_shard``
            # modules, so the call falls back to a FULL_STATE_DICT
            # codepath where only the coordinator rank holds non-empty
            # state — DCP then writes only that rank's shard, and on a
            # 16-rank job you get four ``__N_0.distcp`` files instead
            # of sixteen. Raw ``state_dict()`` keeps DTensor metadata
            # intact on every rank and produces a proper per-rank
            # sharded write.
            # Skip the GPU-materialising optimizer-state primer when the caller
            # pre-seeded a CPU template (audit offload); otherwise prime it so
            # DCP has a properly-sharded template to plan against.
            if not optim_state_offload and not model_weights_only:
                _init_optim_state(optimizer)
            # model_weights_only: plan the DCP read against the model template
            # alone, so the checkpoint's optimizer shards are never requested and
            # the freshly-built optimizer keeps its zero moments.
            if model_weights_only:
                state = {"model": model.state_dict()}
            else:
                state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
            # Route DCP load through the same dedicated gloo PG that
            # save uses, instead of the default NCCL PG. The
            # planner's metadata collectives are CPU-side object
            # gathers; running them through NCCL keeps pod 0 contended
            # for longer, which on restart correlated with the
            # rendezvous-TCPStore "Broken pipe" failures (the elastic
            # agent's 60s poll on rdzv times out when pod 0 is busy
            # during load).
            pg = self._get_ckpt_pg()
            dcp.load(
                state,
                checkpoint_id=str(ckpt_dir / "dcp"),
                process_group=pg,
            )
            model.load_state_dict(state["model"])
            if not optim_state_offload and not model_weights_only:
                optimizer.load_state_dict(state["optim"])
            # else: DCP filled the pre-seeded CPU state tensors in place
            # (``optimizer.state_dict()`` returns references to them), so the
            # optimizer state is already loaded and stays on CPU — a
            # ``load_state_dict`` here would copy it onto the param device.
        except Exception as e:
            # In distributed mode, a DCP load failure must NOT silently
            # fall back to ``fallback.pt`` — that file only exists on the
            # CPU-smoke-test path (no dist init), and trying to read it
            # here raises FileNotFoundError that masks the real DCP
            # error in the traceback. Worse, an unhandled exception on
            # one rank's worker (especially pod 0's) takes down its
            # elastic agent, which closes the rendezvous TCPStore and
            # makes every other pod's agent fail with "failed to recv,
            # got 0 bytes" — manifesting as a rendezvous error that
            # hides the real DCP error elsewhere.
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                LOG.error(
                    "rank %d: DCP load failed from %s: %s", rank, ckpt_dir, e,
                )
                raise
            LOG.warning(
                "DCP load failed (%s); trying torch.save fallback (smoke-test path)", e
            )
            # weights_only=True: a checkpoint directory is untrusted input —
            # ``audit_replay`` is designed to be pointed at a dir produced by
            # someone else (see ``--save-checkpoint-dir``) or fetched from a
            # ``--gcs-root`` mirror. Unpickling arbitrary objects from it would
            # run the checkpoint author's code in the auditor's process, and
            # this branch is reachable on exactly that path (a DCP read that
            # raises with dist uninitialised — every fp32/CPU replay). The
            # payload is ``{"model": ..., "optim": ...}`` state dicts written
            # below, which contain only tensors and plain scalars, so the
            # allowlisted unpickler loads them unchanged.
            blob = torch.load(ckpt_dir / "fallback.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(blob["model"])
            if not model_weights_only:
                optimizer.load_state_dict(blob["optim"])

        # Per-rank torch RNG state. Required for deterministic resume —
        # restoring rank-0's state on every rank would re-converge the
        # streams and break anything that uses the global generator
        # per-rank (e.g. data shuffling). Hard-error if missing: a
        # silent cold-seed fallback would make state_hash diverge from
        # a continuous run starting at step+1, defeating the whole
        # reproducibility setup.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        torch_rng_path = ckpt_dir / f"rng.rank_{rank}.pt"
        if not torch_rng_path.exists():
            raise FileNotFoundError(
                f"missing per-rank RNG file {torch_rng_path}. checkpoint "
                f"at {ckpt_dir} is incomplete — likely a partial multi-pod "
                f"NFS write at save time (the failure mode that motivated "
                f"the rank-0-writes-everything path in "
                f"Checkpointer._gather_and_write_per_rank). cannot resume "
                f"deterministically without this file."
            )
        # weights_only=True for the same reason as fallback.pt above.
        # ``_write_one_rank_files`` saves ``{"cpu": ByteTensor, "cuda":
        # list[ByteTensor] | None}`` — tensors, a list and None, all of which
        # the allowlisted unpickler handles with no extra globals registered.
        blob = torch.load(torch_rng_path, map_location="cpu", weights_only=True)
        torch.set_rng_state(blob["cpu"])
        if blob.get("cuda") is not None and torch.cuda.is_available():
            saved_cuda = blob["cuda"]
            ndev = torch.cuda.device_count()
            if len(saved_cuda) == ndev:
                torch.cuda.set_rng_state_all(saved_cuda)
            else:
                # Device-count mismatch — the single-device audit loading a
                # multi-GPU run's checkpoint (it saved one CUDA RNG state per
                # rank-visible GPU). ``set_rng_state_all`` requires exactly
                # ``device_count`` states, so restore element-wise up to what
                # we have. Per-rank CUDA RNG only drives stochastic ops
                # (dropout), which auditable runs avoid, so this is benign.
                for i in range(min(ndev, len(saved_cuda))):
                    torch.cuda.set_rng_state(saved_cuda[i], device=i)

        # Data-stream state. Canonical-stream (auditable) runs write a single
        # global_stream.json (identical on every rank); legacy runs write a
        # per-rank sampler.rank_N.json. Prefer the global file when present.
        meta_obj = json.loads((ckpt_dir / "meta.json").read_text())
        meta = CheckpointMeta.from_dict(meta_obj)
        extras = self._load_loop_extras(ckpt_dir)

        gstream_path = ckpt_dir / "global_stream.json"
        if gstream_path.exists():
            blob = json.loads(gstream_path.read_text())
            stream_state: MixSamplerState | GlobalStreamState = GlobalStreamState(
                consumed_documents_per_source=blob["consumed_documents_per_source"],
                epoch_per_source=blob["epoch_per_source"],
                mix_rng_state=blob.get("mix_rng_state"),
                carry_over=blob.get("carry_over") or [],
                windows_emitted=blob.get("windows_emitted", 0),
            )
            return stream_state, meta, extras

        # Per-rank sampler state. Each rank's MixSampler has its own
        # ``_rng`` (seeded with ``[seed, rank, world_size]``), per-walker
        # ``consumed`` counter, and residual ``carry_over``. Two ranks
        # pick different sources at each refill and walk the global perm
        # at offset ``rank``, so by step N their states have diverged
        # in every component — restoring rank-0's snapshot on every rank
        # would cause every rank to replay rank-0's document stream,
        # silently breaking both the data mix and per-rank disjointness.
        # Hard-error if missing: same reasoning as the RNG above.
        sampler_blob_path = ckpt_dir / f"sampler.rank_{rank}.json"
        if not sampler_blob_path.exists():
            raise FileNotFoundError(
                f"missing per-rank sampler file {sampler_blob_path}. "
                f"checkpoint at {ckpt_dir} is incomplete — likely a partial "
                f"multi-pod NFS write at save time. cannot resume "
                f"deterministically without this file."
            )
        blob = json.loads(sampler_blob_path.read_text())
        sampler_state = MixSamplerState(
            consumed_documents_per_source=blob["consumed_documents_per_source"],
            epoch_per_source=blob["epoch_per_source"],
            mix_rng_state=blob.get("mix_rng_state"),
            carry_over=blob.get("carry_over") or [],
        )
        return sampler_state, meta, extras

    def _dcp_save(
        self,
        ckpt_dir: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """DCP save of model + optimizer shards only.

        Per-rank state (sampler, RNG, batch hasher) does NOT go through
        DCP — see ``_gather_and_write_per_rank`` for why.
        """
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.default_planner import (
                DefaultSavePlanner,
            )

            # Raw ``state_dict()`` keeps every rank's DTensor shard
            # metadata intact so DCP plans and writes a proper per-rank
            # sharded checkpoint (one ``__N_0.distcp`` per rank). The
            # FQN-keyed ``get_optimizer_state_dict`` helper was tried
            # here and silently dropped to a coordinator-only write
            # under FSDP2 (only ranks 0-3 produced shards out of 16),
            # because it routes through ``FSDP.optim_state_dict`` which
            # is FSDP1-only and doesn't recognise ``fully_shard``
            # modules' sharding metadata.
            state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
            pg = self._get_ckpt_pg()
            # ``dedup_save_to_lowest_rank=True`` under HSDP: every
            # parameter chunk has ``dp_replicate`` identical replicas,
            # and the default load-balanced dedup spreads those
            # replicated writes across all ranks (32-of-32 for an 8×4
            # mesh). At that fan-out we observed 20/32 ``__N_0.distcp``
            # files silently absent from the PVC every checkpoint even
            # though ``.metadata`` claimed each rank had written —
            # Filestore RWX at multi-pod scale per
            # [[project_cluster_dns_and_pvc_rdzv]] is the leading
            # suspect; see also pytorch/pytorch#99976, #104081, #125740
            # which describe the same multi-node DCP failure mode with
            # no upstream resolution. Concentrating every chunk on
            # ``replicate=0`` cuts writers to ``fsdp × tp`` (4 for the
            # 4×4×1 test, 8 for a TP=2 scale-up) so a single replica
            # landing a complete write is sufficient for resume. Same
            # total bytes, smaller failure surface, more deterministic
            # writer set (no tie-breaking through Python set iteration
            # in the load-balancer's ``min``). The per-rank loop_extras
            # use the same "rank 0 writes everything" mitigation —
            # see ``_gather_and_write_per_rank``.
            planner = DefaultSavePlanner(dedup_save_to_lowest_rank=True)
            # Synchronous save. ``async_save`` is appealing for hiding
            # disk-write time behind the next training step, but it adds
            # a background-thread race window that complicated diagnosis
            # of the missing-shards issue, and required ``DefaultStager``
            # / ``close()`` machinery only available on torch ≥ 2.9 —
            # which the host drivers (CUDA 12.2) won't accept. ``dcp.save``
            # blocks until every rank's writes are fsync'd and the
            # ``.metadata`` is committed, which is the property we care
            # about. The cost is one extra ckpt-write duration per save
            # (≈10–30 s for a 1B model on Filestore at 15-min cadence).
            dcp.save(
                state,
                checkpoint_id=str(ckpt_dir / "dcp"),
                process_group=pg,
                planner=planner,
            )
        except Exception as e:
            # Only fall back to torch.save for CPU smoke tests where dist
            # was never initialized. In a real distributed run we must NOT
            # paper over DCP failures: per-rank torch.save to the same path
            # races, sharded state can't be reconstructed from one file, and
            # silently warning hid the original NCCL-thread-race bug for an
            # entire run before the next collective timed out.
            if torch.distributed.is_initialized():
                raise
            LOG.warning(
                "DCP unavailable (%s); using torch.save (smoke-test path)", e
            )
            state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
            torch.save(state, ckpt_dir / "fallback.pt")

    def _gather_and_write_per_rank(
        self,
        ckpt_dir: Path,
        *,
        sampler_state: MixSamplerState | None,
        batch_hasher_digest: bytes | None,
        spike_state: dict | None,
        meta: CheckpointMeta,
        global_stream_state: GlobalStreamState | None = None,
    ) -> None:
        """Gather every rank's per-rank state to rank 0; rank 0 writes
        all per-rank files plus rank-0 metas (meta.json, spike_protocol.json).

        Background: the straightforward approach — every rank writes its
        own ``sampler.rank_N.json`` / ``rng.rank_N.pt`` /
        ``batch_hasher.rank_N.bin`` via torch.save / write_text — was
        observed to silently lose writes on 3 of 8 pods (ranks 4-7,
        16-23) under our cluster's Filestore RWX backend at 32-rank
        scale. Pods 0, 2, 3, 6, 7 either wrote real data or zero-byte
        files; pods 1, 4, 5 wrote nothing visible to the PVC. The save
        raised no exception; the failure surfaced on the NEXT resume
        as a FileNotFoundError when the missing-file ranks fell into
        the legacy fallback (since removed — now a hard error).

        Mitigation: every rank pickles its blob, gloo-gather to rank 0,
        rank 0 writes every per-rank file from a single thread on pod 0.
        Pod 0's writes ARE observed to land reliably (DCP shards under
        ``dedup_save_to_lowest_rank=True`` come from pod 0 alone and
        always land). This matches the rationale for that DCP planner
        flag — concentrate writes on the one pod whose writes are
        observed to land. Per-rank read semantics are preserved: each
        rank reads its own ``sampler.rank_N.json`` etc. on resume,
        unchanged from before.

        Spike protocol state is identical across ranks under
        deterministic training (grad_norm is all-reduced; NaN/Inf
        propagates everywhere via FSDP grad sync), so rank 0 alone is
        authoritative — written here in the same rank-0 block.
        """
        # Each rank assembles its blob locally. Tensors are picklable;
        # gather_object handles variable-size Python objects. ``sampler_state``
        # is None in canonical-stream (auditable) mode — there the stream
        # position is global and identical on every rank, so it's written once
        # as ``global_stream.json`` by rank 0 rather than gathered per-rank.
        local_blob: dict[str, Any] = {
            "sampler_state_dict": (
                {
                    "consumed_documents_per_source": (
                        sampler_state.consumed_documents_per_source
                    ),
                    "epoch_per_source": sampler_state.epoch_per_source,
                    "mix_rng_state": sampler_state.mix_rng_state,
                    "carry_over": sampler_state.carry_over,
                }
                if sampler_state is not None
                else None
            ),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "batch_hasher_digest": batch_hasher_digest,
        }

        if not torch.distributed.is_initialized():
            # Single-process path (CPU smoke tests). Write the local
            # blob directly as rank 0.
            self._write_one_rank_files(ckpt_dir, rank=0, blob=local_blob)
            self._write_rank0_extras(
                ckpt_dir, meta=meta, spike_state=spike_state,
                global_stream_state=global_stream_state,
            )
            return

        pg = self._get_ckpt_pg()
        world = torch.distributed.get_world_size(group=pg)
        rank = torch.distributed.get_rank(group=pg)

        # gloo gather_object: pickled Python objects, variable size.
        # Total payload is tiny (~few MB across 32 ranks: sampler state
        # up to ~125 KB/rank, RNG ~5-10 KB/rank, batch hasher 32 B/rank).
        if rank == 0:
            gathered: list[Any] = [None] * world
            torch.distributed.gather_object(
                local_blob, gathered, dst=0, group=pg,
            )
            for r, blob in enumerate(gathered):
                assert blob is not None, (
                    f"rank {r} contributed no blob to the gather"
                )
                self._write_one_rank_files(ckpt_dir, rank=r, blob=blob)
            self._write_rank0_extras(
                ckpt_dir, meta=meta, spike_state=spike_state,
                global_stream_state=global_stream_state,
            )
        else:
            torch.distributed.gather_object(
                local_blob, None, dst=0, group=pg,
            )

    @staticmethod
    def _write_one_rank_files(
        ckpt_dir: Path, *, rank: int, blob: dict
    ) -> None:
        # sampler_state_dict is None in canonical-stream mode (one global file
        # is written by rank 0 instead — see _write_rank0_extras).
        if blob.get("sampler_state_dict") is not None:
            (ckpt_dir / f"sampler.rank_{rank}.json").write_text(
                json.dumps(blob["sampler_state_dict"])
            )
        rng_blob = {"cpu": blob["rng_cpu"], "cuda": blob["rng_cuda"]}
        torch.save(rng_blob, ckpt_dir / f"rng.rank_{rank}.pt")
        if blob["batch_hasher_digest"] is not None:
            (ckpt_dir / f"batch_hasher.rank_{rank}.bin").write_bytes(
                blob["batch_hasher_digest"]
            )

    @staticmethod
    def _write_rank0_extras(
        ckpt_dir: Path,
        *,
        meta: CheckpointMeta,
        spike_state: dict | None,
        global_stream_state: GlobalStreamState | None = None,
    ) -> None:
        (ckpt_dir / "meta.json").write_text(
            json.dumps(dataclasses.asdict(meta), indent=2)
        )
        if spike_state is not None:
            (ckpt_dir / "spike_protocol.json").write_text(
                json.dumps(spike_state, indent=2)
            )
        # The canonical global stream position is identical on every rank, so a
        # single file suffices (vs the per-rank sampler.rank_N.json). The audit
        # reconstructs the stream from (seed, manifests, step); this file is the
        # bit-exact resume state + a cross-check.
        if global_stream_state is not None:
            (ckpt_dir / "global_stream.json").write_text(
                json.dumps(dataclasses.asdict(global_stream_state))
            )

    def has_checkpoints(self) -> bool:
        """True if any ``step_*`` checkpoint dir already exists (complete or
        partial). Used by the loop to refuse a cold start that would write on
        top of a prior run's checkpoints — the run_id-collision overwrite that
        ``_assert_run_id_agrees`` and the uid-keyed launcher handoff guard
        against from the other direction.
        """
        return any(self.root.glob("step_*"))

    def latest(self) -> Path | None:
        """Most recent COMPLETE checkpoint, or None.

        Skips dirs without the ``_COMPLETE`` sentinel — those are either
        in-progress saves or saves that crashed before rank 0 wrote the
        sentinel as the final action of ``save()``. Without this filter,
        ``--resume-from latest`` would pick up a partial save and crash
        on missing per-rank files (the original failure mode that
        motivated this whole change).
        """
        candidates = sorted(self.root.glob("step_*"), key=lambda p: p.name)
        for p in reversed(candidates):
            if (p / "_COMPLETE").exists():
                return p
        return None

    def garbage_collect(self, keep_last: int = 4, keep_every: int = 10) -> None:
        """Keep the last ``keep_last`` checkpoints + every ``keep_every``-th
        forever (plan/05 §6).
        """
        ckpts = sorted(self.root.glob("step_*"), key=lambda p: p.name)
        if len(ckpts) <= keep_last:
            return
        to_keep: set[Path] = set(ckpts[-keep_last:])
        for c in ckpts:
            try:
                step = int(c.name.split("_")[-1])
            except ValueError:
                continue
            if step % keep_every == 0:
                to_keep.add(c)
        for c in ckpts:
            if c in to_keep:
                continue
            import shutil

            shutil.rmtree(c, ignore_errors=True)
