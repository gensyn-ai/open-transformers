"""Verify a downloaded audit hand-off against the published state hash — on
disk, before replaying anything.

An auditor who receives someone else's hand-off checkpoint has, before this,
only two ways to relate it to the run: the file digest its uploader declared,
and the ``state_hash.txt`` the uploader wrote into it. Neither is evidence —
both come from the same author. The actual commitment is the run's own chained
v3 hash, published in ``logs/state_hashes.jsonl``, and reproducing it requires
the target step's final gradients, which an ordinary checkpoint does not carry.

The hand-off includes ``gradients.safetensors`` beside the checkpoint.
This module uses them: load the checkpoint, reattach the gradients bit for
bit, re-slice the state into the recorded shard layout and recompute

    finalize_state_hash(prev_hash=<previous published hash>,
                        shard_state_digest=audit_shard_state_digest(...),
                        optimizer=..., batch_digest=...)

then compare against the hash the run published *for this checkpoint's step*.
No forward, no backward, no data — seconds and a checkpoint load, against a
replay's hours.

**Both hashes must come from the run's published log, not from the artifact.**
Nothing here reads ``state_hash.txt`` as the target: an author who can write
the tensors can write that file too. It is read only to fail early when the
author's own claim already disagrees with the published log.

What this proves and what it does not
-------------------------------------
A match proves the hashed state — weights, optimizer moments and param groups,
the target step's gradients, and the running batch digest — is bitwise the
state the run committed to at that step. It says nothing about the rest of the
checkpoint: RNG, the data-stream cursor, spike state and the descriptor keys in
``meta.json`` are *not* covered by the v3 hash, so a continuation still trusts
them (see ``docs/audit-replay-usage.md``, "Trust boundary").

Entry point: ``pretrain-audit-verify-handoff``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from pretrain.cli.audit_replay import (
    _GRAD_SIDECAR_FILENAME,
    _GRAD_SIDECAR_FORMAT,
    _GRAD_SIDECAR_VERSION,
    _MAX_AUDIT_WORLD,
    _apply_repop_env,
    _looks_like_digest,
)
from pretrain.config import load_config, parse_config_resolved
from pretrain.train.state_hash import (
    audit_shard_state_digest,
    combine_batch_digests,
    finalize_state_hash,
)

LOG = logging.getLogger("pretrain.audit.verify")


class HandoffError(Exception):
    """The hand-off cannot be verified. Never a "probably fine" — every raise
    here is a reason not to spend a replay on this artifact."""


def attach_gradients(model: torch.nn.Module, sidecar: Path) -> int:
    """Reattach the gradient sidecar to ``model``, bits unchanged.

    The sidecar's tensor keys and its ``none_grad_names`` metadata must together
    account for every named parameter exactly once: that is what makes a missing
    or corrupt entry impossible to mistake for a legitimate ``grad is None``.
    Dtype and shape are checked against the loaded parameter, so a sidecar
    written for a different model is refused here rather than producing a
    mismatching digest with no explanation. Returns the number of tensors
    attached.
    """
    if not sidecar.is_file():
        raise HandoffError(
            f"{sidecar} is missing. The published state hash folds in the target "
            "step's gradients, so without the sidecar the hash cannot be "
            "reconstructed — and an absent sidecar must never be read as "
            "'every gradient was None'."
        )
    import safetensors

    try:
        with safetensors.safe_open(str(sidecar), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if (metadata.get("format") != _GRAD_SIDECAR_FORMAT
                    or metadata.get("format_version") != str(_GRAD_SIDECAR_VERSION)):
                raise HandoffError(f"{sidecar}: unsupported gradient sidecar format/version.")
            none_names = json.loads(metadata["none_grad_names"])
            if not isinstance(none_names, list) or not all(isinstance(n, str) for n in none_names):
                raise HandoffError(f"{sidecar}: none_grad_names must be a list of strings.")
            tensors = {name: f.get_tensor(name) for name in f.keys()}
    except (OSError, ValueError, KeyError, safetensors.SafetensorError) as exc:
        raise HandoffError(f"{sidecar}: invalid gradient sidecar: {exc}") from exc
    params = dict(model.named_parameters())

    named, none_set = set(tensors), set(none_names)
    if len(none_set) != len(none_names):
        raise HandoffError(f"{sidecar} lists a duplicate name in none_grad_names.")
    if overlap := sorted(named & none_set):
        raise HandoffError(
            f"{sidecar} lists {overlap[:4]} both as a tensor and as a None "
            "gradient; the two are mutually exclusive."
        )
    if extra := sorted((named | none_set) - set(params)):
        raise HandoffError(
            f"{sidecar} carries entries for parameters this model does not "
            f"have: {extra[:4]}."
        )
    if missing := sorted(set(params) - named - none_set):
        raise HandoffError(
            f"{sidecar} accounts for no gradient at all for {missing[:4]} — "
            "neither a tensor nor a None entry. Incomplete sidecar."
        )

    for name, grad in tensors.items():
        p = params[name]
        if grad.dtype != p.dtype or tuple(grad.shape) != tuple(p.shape):
            raise HandoffError(
                f"{sidecar}: gradient for {name} is {grad.dtype} "
                f"{tuple(grad.shape)}, but the parameter is {p.dtype} "
                f"{tuple(p.shape)}."
            )
    for name, p in params.items():
        grad = tensors.get(name)
        p.grad = None if grad is None else grad.to(device=p.device)
    return len(tensors)


def read_batch_digest(checkpoint: Path, dp_world_size: int) -> bytes:
    """Rebuild the run's global batch digest from the hand-off's per-rank chains.

    ``batch_hasher.rank_<r>.bin`` is one rank's running chain; the run combined
    them in DP-rank order (:func:`combine_batch_digests`). Every rank's file is
    required — a hand-off that carries only rank 0 cannot reproduce a
    multi-rank run's digest, and silently hashing fewer ranks would compute a
    different schema and report a mismatch as a divergence.
    """
    digests = []
    for r in range(dp_world_size):
        p = checkpoint / f"batch_hasher.rank_{r}.bin"
        if not p.is_file():
            raise HandoffError(
                f"{p} is missing: the run folds a cross-rank batch digest into "
                f"its state hash, and this hand-off declares dp_world_size="
                f"{dp_world_size}, so all {dp_world_size} per-rank chains are "
                "needed to rebuild it."
            )
        digests.append(p.read_bytes())
    try:
        return combine_batch_digests(digests)
    except ValueError as exc:
        raise HandoffError(f"{checkpoint}: invalid batch-hasher chains: {exc}") from exc


def verify_loaded_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
    *,
    expect_hash: str,
    prev_hash: str,
    dp_world_size: int,
    dp_shard: int,
    include_grads: bool,
    include_batch: bool,
) -> dict:
    """The gate itself, over an already-loaded model + optimizer.

    Split out from :func:`verify_handoff` so the reconstruction is exercised
    without building the run's real model (which needs the native repop
    runtime): the hash is pure byte-slicing and blake2b, and this is the part
    that has to be right.

    Gradients are released before returning — they are ~1x the model size and
    are only needed for the digest; a continuation must not inherit them.
    """
    attached = None
    try:
        if include_grads:
            attached = attach_gradients(model, checkpoint / _GRAD_SIDECAR_FILENAME)
        batch_digest = read_batch_digest(checkpoint, dp_world_size) if include_batch else None

        reconstructed = finalize_state_hash(
            prev_hash=prev_hash,
            shard_state_digest=audit_shard_state_digest(
                model, dp_shard, dp_world_size,
                optimizer=optimizer, include_grads=include_grads,
            ),
            optimizer=optimizer,
            batch_digest=batch_digest,
        )
    finally:
        model.zero_grad(set_to_none=True)

    return {
        "verified": reconstructed == expect_hash,
        "reconstructed_hash": reconstructed,
        "expected_hash": expect_hash,
        "prev_hash": prev_hash,
        "dp_world_size": dp_world_size,
        "dp_shard": dp_shard,
        "include_grads": include_grads,
        "include_batch": include_batch,
        "gradients_attached": attached,
    }


def _descriptor(checkpoint: Path) -> dict:
    meta_path = checkpoint / "meta.json"
    if not meta_path.is_file():
        raise HandoffError(
            f"{meta_path} is missing — this is not a complete hand-off "
            "checkpoint, so its topology and hash cadence are unknown."
        )
    if not (checkpoint / "_COMPLETE").is_file():
        raise HandoffError(
            f"{checkpoint} carries no _COMPLETE marker: the writer never "
            "finished it, or the download is partial. Refusing to verify a "
            "checkpoint that was never published."
        )
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        raise HandoffError(f"{meta_path} is not readable JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise HandoffError(f"{meta_path} must contain a JSON object.")
    if type(meta.get("step")) is not int or meta["step"] <= 0:
        raise HandoffError(f"{meta_path}: step must be a positive integer.")
    return meta


def verify_handoff(
    checkpoint: str | Path,
    *,
    expect_hash: str,
    prev_hash: str,
    config_name: str | None = None,
    device: str = "cpu",
) -> dict:
    """Reconstruct this hand-off's published v3 state hash and compare.

    ``expect_hash`` is the run's published hash for the checkpoint's OWN step;
    ``prev_hash`` is the published hash of the previous HASHED step (with a
    cadence of N that is step-N, not step-1). Both are the caller's job to take
    from the independently published log, never from the artifact.
    """
    checkpoint = Path(checkpoint)
    # Literal commitments only: never resolve a checkpoint's own claim file.
    for flag, value in (("--expect-hash", expect_hash), ("--prev-hash", prev_hash)):
        if not _looks_like_digest(value.strip()):
            raise HandoffError(f"{flag} must be a full literal hex digest from the "
                               "published log, not a file path or truncated digest.")
    expect_hash, prev_hash = expect_hash.strip().lower(), prev_hash.strip().lower()

    meta = _descriptor(checkpoint)

    cfg = (load_config(config_name) if config_name
           else parse_config_resolved(meta["config_resolved"]))
    # These come from the artifact's own meta.json, which the v3 hash does not
    # authenticate. That is safe here only because they are inputs to the
    # digest: an author who understates dp_shard or turns include_grads off
    # computes a different value and fails the comparison. Nothing is trusted
    # on their word — it is simply checked by the thing being checked.
    sh = cfg.train.state_hash
    step = meta["step"]

    if sh.every_n_steps <= 0:
        raise HandoffError(
            "the run that wrote this hand-off did not hash its state "
            "(state_hash.every_n_steps is 0), so there is no published "
            "commitment to verify it against."
        )
    if step % sh.every_n_steps != 0:
        raise HandoffError(
            f"step {step} is off the run's hash cadence (every "
            f"{sh.every_n_steps} steps), so the published log holds no hash "
            "for it. Such a checkpoint stamps a side-link its metadata does "
            "not chain from; verifying it would compute a different schema "
            "than the one that was published. Unsupported."
        )
    # On a hash-due step the loop stamps the same value in both places. A
    # difference means an off-cadence side-link (handled above) or a doctored
    # artifact — either way the published chain does not describe this file.
    stamped = (checkpoint / "state_hash.txt")
    claimed = stamped.read_text().strip().lower() if stamped.is_file() else None
    if claimed is not None and claimed != str(meta.get("chained_hash", "")).lower():
        raise HandoffError(
            f"{stamped} ({claimed[:16]}…) disagrees with meta.json's running "
            f"chain ({str(meta.get('chained_hash'))[:16]}…). This hand-off is "
            "not a plain hash-due checkpoint; refusing to guess which value "
            "the published chain continues from."
        )
    if claimed is not None and claimed != expect_hash:
        raise HandoffError(
            "this hand-off claims a different state than the run published for "
            f"step {step}:\n"
            f"    the artifact says  {claimed}\n"
            f"    the run published  {expect_hash}\n"
            "The published log is authoritative. Nothing here is worth a replay."
        )

    N = int(meta.get("dp_world_size") or 1)
    dp_replicate = int(meta.get("dp_replicate") or 1)
    dp_shard = int(meta.get("dp_shard") or N)
    if not 1 <= N <= _MAX_AUDIT_WORLD or dp_shard < 1 or dp_replicate < 1:
        raise HandoffError(
            f"hand-off declares an implausible topology (dp_world_size={N}, "
            f"dp_replicate={dp_replicate}, dp_shard={dp_shard}); the cap is "
            f"{_MAX_AUDIT_WORLD}."
        )
    if dp_replicate * dp_shard != N:
        raise HandoffError(
            f"hand-off topology is inconsistent: dp_replicate({dp_replicate}) * "
            f"dp_shard({dp_shard}) != dp_world_size({N})."
        )

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise HandoffError("CUDA is unavailable; use --device cpu or install a CUDA-enabled runtime.")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise HandoffError("MPS is unavailable; use --device cpu or an MPS-enabled runtime.")

    # Apply only after cheap refusals, but before any repop-backed import.
    _apply_repop_env(meta.get("repop_env"))

    from pretrain.model import build_model
    from pretrain.optim.adamw_repop import prime_optimizer_state
    from pretrain.optim.registry import build_optimizer
    from pretrain.parallel.parallel_dims import ParallelDims
    from pretrain.parallel.parallelize_llama3_repop import parallelize_llama3_repop
    from pretrain.train.checkpoint import Checkpointer

    model = build_model(cfg.model, device=dev)
    # Same wrapping the replay applies: activation checkpointing renames
    # parameters, and a model whose FQNs differ from the checkpoint's loads
    # nothing from DCP while reporting success.
    model = parallelize_llama3_repop(
        model, cfg, ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
    )
    optimizer = build_optimizer(model, cfg.optim)
    prime_optimizer_state(optimizer)
    # Checkpointer.load routes DCP through the validating metadata reader.
    Checkpointer(checkpoint.parent).load(checkpoint, model, optimizer)

    result = verify_loaded_state(
        model, optimizer, checkpoint,
        expect_hash=expect_hash, prev_hash=prev_hash,
        dp_world_size=N, dp_shard=dp_shard,
        include_grads=sh.include_grads, include_batch=sh.include_batch,
    )
    result["step"] = step
    result["checkpoint"] = str(checkpoint)
    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pretrain-audit-verify-handoff",
        description="Reconstruct a hand-off checkpoint's published v3 state "
                    "hash from disk (no forward/backward) and compare it.",
    )
    p.add_argument("--checkpoint", required=True,
                   help="the hand-off checkpoint directory (step_<N>)")
    p.add_argument("--expect-hash", required=True,
                   help="the PUBLISHED hash for this checkpoint's own step: a "
                        "literal hex digest. Take it from the "
                        "run's state_hashes.jsonl, never from the artifact.")
    p.add_argument("--prev-hash", required=True,
                   help="the PUBLISHED hash of the previous HASHED step, which "
                        "the chain folds in. With a cadence of N this is step "
                        "N back, not the previous step and not the previous "
                        "checkpoint.")
    p.add_argument("--config-name", default=None,
                   help="override the checkpoint's own resolved config")
    p.add_argument("--device", default="cpu",
                   help="where to materialise the state (default cpu; the "
                        "digest is device-independent by construction)")
    p.add_argument("--json", dest="json_out", default=None,
                   help="also write the result object to this path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _parse_args(argv)
    # DCP wraps load failures in a BaseException subclass, not Exception.
    # Catch it explicitly without swallowing KeyboardInterrupt/SystemExit.
    from torch.distributed.checkpoint.api import CheckpointException

    try:
        result = verify_handoff(
            args.checkpoint,
            expect_hash=args.expect_hash,
            prev_hash=args.prev_hash,
            config_name=args.config_name,
            device=args.device,
        )
    except (Exception, CheckpointException) as exc:
        error = str(exc)
        if not isinstance(exc, HandoffError):
            error = f"{type(exc).__name__}: {exc}"
            LOG.exception("Unexpected handoff verification failure")
        result = {"verified": False, "error": error}

    if args.json_out:
        try:
            Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        except OSError as exc:
            result = {"verified": False, "error": f"cannot write {args.json_out}: {exc}"}
    print(json.dumps(result, indent=2), flush=True)
    if "error" in result:
        LOG.error("%s", result["error"])
        return 1
    if result["verified"]:
        LOG.info(
            "VERIFIED tensor state: step %d reproduces the published hash %s",
            result["step"], result["expected_hash"],
        )
        return 0
    LOG.error(
        "MISMATCH: reconstructed %s, published %s. The hashed state of this "
        "hand-off is not the state the run committed to.",
        result["reconstructed_hash"], result["expected_hash"],
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
