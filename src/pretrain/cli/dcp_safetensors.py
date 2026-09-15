"""Convert a DCP checkpoint to a single ``.safetensors`` file and back.

Our training checkpoints are torch DCP shard dirs (``step_N/dcp/``), which
nothing outside torch speaks; safetensors is the interchange format the rest
of the world does (HF transformers, inference servers, external evals). This
CLI bridges the two, in both directions, without needing distributed init or
even a model build — the full state dict is reconstructed offline from the
DCP metadata alone.

``to-safetensors``:
    Every tensor leaf of the checkpoint's state dict goes into one
    safetensors file, keyed by its dotted FQN (``model.tok_embeddings.weight``,
    ``optim.state.0.exp_avg``). Non-tensor leaves (optimizer ``param_groups``,
    scalar counters, ...) and the exact nesting structure are JSON-encoded
    into the safetensors header metadata (``pretrain.skeleton``), so the
    inverse is lossless. Dtypes are preserved bit-for-bit — no casting.

``from-safetensors``:
    Rebuilds the nested state dict from the file (using the header skeleton
    when present, else the flat tensor keys) and writes a fresh DCP dir as a
    single full-tensor shard. DCP re-shards on load, so the result loads into
    an FSDP/HSDP job of any topology — same contract as torch's
    ``format_utils.torch_save_to_dcp`` output.

When the input is a full checkpoint dir (the ``step_N/`` layout), the rest
of the checkpoint travels too, in a second header blob (``pretrain.sidecar``):
per-rank RNG states and batch-hasher digests, meta.json, global_stream.json,
spike_protocol.json, sampler.rank_N.json and state_hash.txt (~0.4 MB total
against the ~18 GB of tensors at 1B), plus — for chained-audit hand-offs —
the ``gradients.safetensors`` export, whose tensors ride in the
payload and whose header metadata rides in the skeleton (a recipient needs
it to recompute the logged state hash, which covers live grads). RNG ``.pt``
blobs are decoded to tensors at pack time — no pickled bytes are embedded.
``from-safetensors``
then reconstructs the complete directory (``dcp/`` + every sidecar file +
``_COMPLETE``), i.e. a bit-exact resume point that ``audit_replay
--checkpoint`` and the training loop's ``--resume-from`` accept. This is the
Hand-off relay requirement: resuming from weights+optim alone would
draw different data and RNG on the next step and report a NO MATCH that is
indistinguishable from real divergence.

Given a bare ``dcp/`` dir or a ``--keys`` filter there is no sidecar to
carry, and the output of the inverse is correspondingly a bare DCP dir — a
weight-transport artifact (export, surgery, re-import), not a resume point.

A checkpoint directory is untrusted input (same stance as ``audit_replay``):
the DCP ``.metadata`` gate from ``pretrain.train.checkpoint`` runs before any
read, and the non-tensor (bytes) leaves are unpickled with
``weights_only=True`` — torch's own ``DefaultLoadPlanner.load_bytes`` uses a
full pickle, which would execute a doctored checkpoint's code.

Usage:
    python -m pretrain.cli.dcp_safetensors to-safetensors CKPT_DIR out.safetensors
    python -m pretrain.cli.dcp_safetensors to-safetensors CKPT_DIR out.safetensors --keys model
    python -m pretrain.cli.dcp_safetensors from-safetensors in.safetensors OUT_DCP_DIR
    python -m pretrain.cli.dcp_safetensors from-safetensors hf_model.safetensors OUT --top-level-key model

``CKPT_DIR`` may be the checkpoint dir (containing ``dcp/``) or the ``dcp``
dir itself. The full state dict is materialised in host RAM during
conversion — for an 8B run with optimizer state that is >100 GB, so use
``--keys model`` when only the weights are wanted.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import torch

LOG = logging.getLogger(__name__)

# safetensors header-metadata keys (str -> str only, hence the JSON blobs).
_FORMAT_KEY = "pretrain.format"
_FORMAT_VERSION = "dcp-safetensors/1"
_SKELETON_KEY = "pretrain.skeleton"
_SIDECAR_KEY = "pretrain.sidecar"

# The only filenames a sidecar may carry (also the unpack-side gate: these
# names come from an untrusted header, so anything else — in particular
# anything path-like — is refused rather than written to disk).
_SIDECAR_FILE_RE = re.compile(
    r"^(meta\.json|global_stream\.json|spike_protocol\.json|state_hash\.txt"
    r"|sampler\.rank_\d+\.json|batch_hasher\.rank_\d+\.bin)$"
)


# --------------------------------------------------------------------------- #
# Offline (no-dist) DCP read/write
# --------------------------------------------------------------------------- #
def _weights_only_empty_planner():
    """``_EmptyStateDictLoadPlanner`` with the bytes path hardened.

    The stock planner rebuilds the full state dict from metadata (no template
    model needed), but inherits ``DefaultLoadPlanner.load_bytes``, which
    unpickles every non-tensor leaf with ``torch.load(..., weights_only=
    False)`` — arbitrary code execution if the checkpoint is doctored. Real
    checkpoints only store plain containers/scalars in bytes leaves
    (optimizer ``param_groups`` etc.), all of which ``weights_only=True``
    admits, so we override with the safe unpickler. Defined lazily inside a
    function because the base class is a private torch API.
    """
    from torch.distributed.checkpoint._traverse import set_element
    from torch.distributed.checkpoint.default_planner import (
        _EmptyStateDictLoadPlanner,
    )

    class _WeightsOnlyEmptyPlanner(_EmptyStateDictLoadPlanner):
        def load_bytes(self, read_item, value: io.BytesIO) -> None:
            obj = torch.load(value, weights_only=True)
            if self.flatten_state_dict:
                set_element(
                    self.original_state_dict,
                    self.mappings[read_item.dest_index.fqn],
                    obj,
                )
            else:
                self.state_dict[read_item.dest_index.fqn] = obj

    return _WeightsOnlyEmptyPlanner()


def load_dcp_offline(dcp_dir: Path) -> dict[str, Any]:
    """Reconstruct the full unsharded state dict from a DCP dir, no dist init.

    Applies the untrusted-``.metadata`` gate first (see
    ``pretrain.train.checkpoint``). NOTE: torch's ``set_element`` rebuilds
    int-keyed dicts (optimizer ``state``) as *lists* — the dotted FQNs are
    identical either way, so a later ``dcp.load`` against a real optimizer
    template still plans correctly.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    from pretrain.train.checkpoint import _validate_dcp_metadata

    _validate_dcp_metadata(dcp_dir)
    state: dict[str, Any] = {}
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(str(dcp_dir)),
        planner=_weights_only_empty_planner(),
        no_dist=True,
    )
    return state


def save_dcp_offline(state: dict[str, Any], dcp_dir: Path) -> None:
    """Write ``state`` as a single-shard (full-tensor) DCP dir, no dist init."""
    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.checkpoint.state_dict_saver import _save_state_dict

    _save_state_dict(
        state,
        storage_writer=FileSystemWriter(str(dcp_dir)),
        no_dist=True,
    )


# --------------------------------------------------------------------------- #
# state dict <-> (flat tensors, JSON skeleton)
# --------------------------------------------------------------------------- #
# The skeleton mirrors the state dict's structure with every tensor replaced
# by a reference into the safetensors payload and every non-tensor leaf
# inlined as JSON. Nodes are ``{"t": <tag>, ...}``:
#   dict   -> {"t": "dict",   "items": [[key, node], ...]}   (key: str/int/bool/None)
#   list   -> {"t": "list",   "items": [node, ...]}
#   tuple  -> {"t": "tuple",  "items": [node, ...]}
#   tensor -> {"t": "tensor", "key": <flat safetensors key>}
#   scalar -> {"t": "v",      "v": <str/int/bool/None/finite float>}
#   float  -> {"t": "float",  "v": "inf"|"-inf"|"nan"}       (non-finite only)
#   bytes  -> {"t": "bytes",  "v": <base64>}
# Dict entries are stored as [key, value] pairs (not a JSON object) so int
# keys — optimizer ``state`` — survive the round trip typed.


def flatten_state(
    obj: Any, path: tuple = (), tensors: dict[str, torch.Tensor] | None = None
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Split ``obj`` into (skeleton, flat tensor dict). See scheme above."""
    if tensors is None:
        tensors = {}
    if isinstance(obj, torch.Tensor):
        name = ".".join(str(p) for p in path) or "_root"
        # Dotted joins can collide ("a.b" under "c" vs "b" under "c.a");
        # correctness only needs uniqueness, the skeleton holds the mapping.
        base, n = name, 2
        while name in tensors:
            name, n = f"{base}#{n}", n + 1
        tensors[name] = obj.detach().cpu().contiguous()
        return {"t": "tensor", "key": name}, tensors
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            if not (k is None or isinstance(k, (str, int, bool))):
                raise TypeError(
                    f"dict key {k!r} of type {type(k).__name__} at "
                    f"{'.'.join(map(str, path))} is not representable in the "
                    f"safetensors skeleton (str/int/bool/None only)"
                )
            node, _ = flatten_state(v, path + (k,), tensors)
            items.append([k, node])
        return {"t": "dict", "items": items}, tensors
    if isinstance(obj, (list, tuple)):
        tag = "list" if isinstance(obj, list) else "tuple"
        items = [
            flatten_state(v, path + (i,), tensors)[0] for i, v in enumerate(obj)
        ]
        return {"t": tag, "items": items}, tensors
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"t": "float", "v": repr(obj)}, tensors
    if obj is None or isinstance(obj, (str, int, bool, float)):
        return {"t": "v", "v": obj}, tensors
    if isinstance(obj, bytes):
        return {"t": "bytes", "v": base64.b64encode(obj).decode("ascii")}, tensors
    raise TypeError(
        f"value of type {type(obj).__name__} at {'.'.join(map(str, path))} "
        f"has no safetensors representation"
    )


def unflatten_state(node: dict, tensors: dict[str, torch.Tensor]) -> Any:
    """Inverse of :func:`flatten_state`."""
    tag = node["t"]
    if tag == "tensor":
        try:
            return tensors[node["key"]]
        except KeyError:
            raise KeyError(
                f"skeleton references tensor {node['key']!r} which is missing "
                f"from the safetensors payload — truncated or edited file?"
            ) from None
    if tag == "dict":
        return {k: unflatten_state(v, tensors) for k, v in node["items"]}
    if tag == "list":
        return [unflatten_state(v, tensors) for v in node["items"]]
    if tag == "tuple":
        return tuple(unflatten_state(v, tensors) for v in node["items"])
    if tag == "v":
        return node["v"]
    if tag == "float":
        return float(node["v"])
    if tag == "bytes":
        return base64.b64decode(node["v"])
    raise ValueError(f"unknown skeleton node tag {tag!r}")


# --------------------------------------------------------------------------- #
# Comparison helpers (verify paths + tests)
# --------------------------------------------------------------------------- #
def bitwise_tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact comparison (NaN == NaN, unlike ``torch.equal``)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    a = a.detach().cpu().contiguous().reshape(-1)
    b = b.detach().cpu().contiguous().reshape(-1)
    if a.numel() == 0:
        return True
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def state_equal(a: Any, b: Any) -> bool:
    """Recursive equality over state dicts; tensors compared bitwise."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and bitwise_tensor_equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(state_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        # list/tuple distinction is not significant here: DCP's offline
        # reconstruction turns tuples into lists anyway (set_element).
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


# --------------------------------------------------------------------------- #
# Checkpoint sidecar (everything in a step_N/ dir besides dcp/)
# --------------------------------------------------------------------------- #
# Shape: {"files": {name: str|bytes verbatim},
#         "rng":   {rank: {"cpu": Tensor, "cuda": [Tensor, ...] | None}},
#         "grads": None | {"tensors": {name: Tensor}, "metadata": {str: str}}}.
# Text/bin files are carried byte-for-byte. rng.rank_N.pt files are DECODED to
# their tensors (loaded weights_only=True) instead of embedding pickled bytes,
# and re-serialised with torch.save on unpack — semantically identical, which
# is what Checkpointer.load consumes. gradients.safetensors (the chained-audit
# handoff's grad export, needed by a recipient to recompute the logged
# state hash) is likewise decoded: its tensors ride in the payload — they are
# gigabytes, far beyond what a header blob may hold — and its header metadata
# (format keys + none_grad_names) rides in the skeleton, rebuilt verbatim on
# unpack.
_GRAD_SIDECAR_NAME = "gradients.safetensors"


def _collect_sidecar(ckpt_dir: Path) -> dict[str, Any] | None:
    """Gather a checkpoint dir's non-dcp state. None if there is none (bare
    dcp payloads / synthetic dirs)."""
    from safetensors import safe_open

    files: dict[str, Any] = {}
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_file() or not _SIDECAR_FILE_RE.match(p.name):
            continue
        files[p.name] = (
            p.read_bytes() if p.name.endswith(".bin") else p.read_text()
        )
    rng: dict[int, Any] = {}
    for p in sorted(ckpt_dir.glob("rng.rank_*.pt")):
        rank = int(p.stem.rsplit("_", 1)[1])
        blob = torch.load(p, map_location="cpu", weights_only=True)
        rng[rank] = {"cpu": blob["cpu"], "cuda": blob.get("cuda")}
    grads = None
    if (ckpt_dir / _GRAD_SIDECAR_NAME).is_file():
        with safe_open(str(ckpt_dir / _GRAD_SIDECAR_NAME), framework="pt") as f:
            grads = {
                "tensors": {k: f.get_tensor(k) for k in f.keys()},
                "metadata": dict(f.metadata() or {}),
            }
    if not files and not rng and grads is None:
        return None
    known = {
        "dcp", "_COMPLETE", _GRAD_SIDECAR_NAME,
        *files, *(f"rng.rank_{r}.pt" for r in rng),
    }
    unknown = sorted(q.name for q in ckpt_dir.iterdir() if q.name not in known)
    if unknown:
        LOG.warning(
            "unrecognized checkpoint entries NOT carried into the sidecar "
            "(extend _SIDECAR_FILE_RE if these matter): %s", unknown,
        )
    return {"files": files, "rng": rng, "grads": grads}


def _validate_sidecar(sidecar: Any) -> dict[str, Any]:
    """Gate a sidecar decoded from an untrusted header before anything is
    written to disk: filenames must match the exact allowlist (no separators,
    no traversal — they become paths), ranks must be ints, RNG entries must be
    tensors/lists of tensors."""
    if (
        not isinstance(sidecar, dict)
        or set(sidecar) != {"files", "rng", "grads"}
        or not isinstance(sidecar["files"], dict)
        or not isinstance(sidecar["rng"], dict)
    ):
        raise SystemExit("sidecar metadata malformed: expected {files, rng, grads}")
    grads = sidecar["grads"]
    if grads is not None:
        grads_ok = (
            isinstance(grads, dict)
            and set(grads) == {"tensors", "metadata"}
            and isinstance(grads["tensors"], dict)
            and all(
                isinstance(k, str) and isinstance(t, torch.Tensor)
                for k, t in grads["tensors"].items()
            )
            and isinstance(grads["metadata"], dict)
            and all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in grads["metadata"].items()
            )
        )
        if not grads_ok:
            raise SystemExit("sidecar gradient entry is malformed")
    for name in sidecar["files"]:
        if not isinstance(name, str) or not _SIDECAR_FILE_RE.match(name):
            raise SystemExit(
                f"sidecar filename {name!r} is not an allowed checkpoint "
                f"entry — refusing (untrusted header; names become paths)"
            )
    for rank, blob in sidecar["rng"].items():
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise SystemExit(f"sidecar rng rank {rank!r} is not a valid rank")
        cuda_ok = blob.get("cuda") is None or (
            isinstance(blob.get("cuda"), list)
            and all(isinstance(t, torch.Tensor) for t in blob["cuda"])
        )
        if (
            not isinstance(blob, dict)
            or set(blob) != {"cpu", "cuda"}
            or not isinstance(blob.get("cpu"), torch.Tensor)
            or not cuda_ok
        ):
            raise SystemExit(f"sidecar rng entry for rank {rank} is malformed")
    return sidecar


def _write_sidecar(ckpt_dir: Path, sidecar: dict[str, Any]) -> None:
    for name, content in sidecar["files"].items():
        if isinstance(content, bytes):
            (ckpt_dir / name).write_bytes(content)
        else:
            (ckpt_dir / name).write_text(content)
    for rank, blob in sidecar["rng"].items():
        torch.save(
            {"cpu": blob["cpu"], "cuda": blob["cuda"]},
            ckpt_dir / f"rng.rank_{rank}.pt",
        )
    if sidecar["grads"] is not None:
        from safetensors.torch import save_file

        save_file(
            {k: t.contiguous() for k, t in sidecar["grads"]["tensors"].items()},
            str(ckpt_dir / _GRAD_SIDECAR_NAME),
            metadata=sidecar["grads"]["metadata"] or None,
        )
    # Written last, mirroring Checkpointer.save: latest()/resume skip dirs
    # without the sentinel.
    (ckpt_dir / "_COMPLETE").write_text("")


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #
def _resolve_dcp_dir(path: Path) -> Path:
    if (path / "dcp" / ".metadata").is_file():
        return path / "dcp"
    if (path / ".metadata").is_file():
        return path
    raise SystemExit(
        f"{path} is not a DCP checkpoint: expected {path}/dcp/.metadata or "
        f"{path}/.metadata"
    )


def dcp_to_safetensors(
    ckpt_path: Path,
    out_file: Path,
    *,
    keys: list[str] | None = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Convert a DCP dir to one safetensors file. Returns summary stats."""
    from safetensors.torch import save_file

    dcp_dir = _resolve_dcp_dir(ckpt_path)
    LOG.info("reconstructing full state dict from %s (offline, no dist)", dcp_dir)
    state = load_dcp_offline(dcp_dir)

    if keys:
        missing = [k for k in keys if k not in state]
        if missing:
            raise SystemExit(
                f"--keys {missing} not in checkpoint; available top-level "
                f"keys: {sorted(state)}"
            )
        state = {k: state[k] for k in keys}

    skeleton, tensors = flatten_state(state)
    metadata = {
        _FORMAT_KEY: _FORMAT_VERSION,
        _SKELETON_KEY: json.dumps(skeleton, separators=(",", ":")),
    }

    # Sidecar: only meaningful for a full, unfiltered checkpoint dir. A --keys
    # subset must NOT look like a resume point, and a bare dcp/ input has no
    # sidecar to offer.
    sidecar = None
    if keys:
        LOG.info("--keys filter set: packing tensors only, no resume sidecar")
    elif dcp_dir.name == "dcp":
        sidecar = _collect_sidecar(dcp_dir.parent)
    if sidecar is not None:
        sc_skeleton, tensors = flatten_state(sidecar, ("__sidecar__",), tensors)
        metadata[_SIDECAR_KEY] = json.dumps(sc_skeleton, separators=(",", ":"))
        LOG.info(
            "sidecar: %d files + %d per-rank RNG states%s packed",
            len(sidecar["files"]), len(sidecar["rng"]),
            "" if sidecar["grads"] is None else
            f" + {len(sidecar['grads']['tensors'])} handoff gradients",
        )
    elif not keys:
        LOG.warning(
            "no checkpoint sidecar found at %s — output will be a "
            "weight-transport artifact, not a resume point", dcp_dir.parent,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("writing %d tensors to %s", len(tensors), out_file)
    save_file(tensors, str(out_file), metadata=metadata)

    if verify:
        _verify_safetensors_against(out_file, tensors, skeleton)
        LOG.info("verify: %s matches the reconstructed state bit-for-bit", out_file)

    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    return {
        "tensors": len(tensors),
        "bytes": total_bytes,
        "top_level_keys": sorted(state),
        "sidecar_files": None if sidecar is None else len(sidecar["files"]),
    }


def safetensors_to_dcp(
    in_file: Path,
    out_dir: Path,
    *,
    top_level_key: str | None = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Convert a safetensors file to a single-shard DCP dir.

    Files written by ``to-safetensors`` are rebuilt losslessly from the
    header skeleton; when the header also carries a checkpoint sidecar the
    output is the full ``step_N/``-layout directory (``dcp/`` + per-rank RNG
    and batch-hasher files + run-state JSONs + ``_COMPLETE``), a loadable
    resume point. Foreign files (no skeleton — e.g. an HF export) become a
    flat {name: tensor} state dict, optionally nested under
    ``top_level_key`` so the result matches our ``{"model": ...}`` layout.
    """
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(in_file), framework="pt") as f:
        file_meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    sidecar = None
    if _SKELETON_KEY in file_meta:
        if top_level_key is not None:
            raise SystemExit(
                "--top-level-key is only for foreign safetensors files; this "
                "file carries a pretrain skeleton that already defines the "
                "full structure"
            )
        skeleton = json.loads(file_meta[_SKELETON_KEY])
        state = unflatten_state(skeleton, tensors)
        referenced = _skeleton_tensor_keys(skeleton)
        if _SIDECAR_KEY in file_meta:
            sc_skeleton = json.loads(file_meta[_SIDECAR_KEY])
            sidecar = _validate_sidecar(unflatten_state(sc_skeleton, tensors))
            referenced |= _skeleton_tensor_keys(sc_skeleton)
        unused = sorted(set(tensors) - referenced)
        if unused:
            LOG.warning(
                "%d tensors in %s are not referenced by the skeleton and "
                "were dropped: %s", len(unused), in_file, unused[:5],
            )
    else:
        LOG.info("no pretrain skeleton in %s; treating keys as a flat state dict", in_file)
        state = {top_level_key: tensors} if top_level_key else dict(tensors)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"refusing to write into non-empty {out_dir} — DCP writers append "
            f"rather than replace, which would corrupt an existing checkpoint"
        )
    # With a sidecar the output is a full checkpoint dir (dcp/ nested inside);
    # without one it is the bare DCP dir itself.
    dcp_target = out_dir / "dcp" if sidecar is not None else out_dir
    dcp_target.mkdir(parents=True, exist_ok=True)
    LOG.info("writing single-shard DCP checkpoint to %s", dcp_target)
    save_dcp_offline(state, dcp_target)
    if sidecar is not None:
        _write_sidecar(out_dir, sidecar)
        LOG.info(
            "sidecar restored: %d files + %d per-rank RNG states + _COMPLETE",
            len(sidecar["files"]), len(sidecar["rng"]),
        )

    if verify:
        reloaded = load_dcp_offline(dcp_target)
        if not state_equal(state, reloaded):
            raise SystemExit(
                f"verify FAILED: state reloaded from {dcp_target} differs "
                f"from the state decoded out of {in_file}"
            )
        if sidecar is not None and not state_equal(
            sidecar, _collect_sidecar(out_dir)
        ):
            raise SystemExit(
                f"verify FAILED: sidecar re-read from {out_dir} differs from "
                f"the sidecar decoded out of {in_file}"
            )
        LOG.info("verify: %s round-trips bit-for-bit", out_dir)

    n_tensors = len(tensors)
    return {
        "tensors": n_tensors,
        "top_level_keys": sorted(state),
        "sidecar_files": None if sidecar is None else len(sidecar["files"]),
    }


def _skeleton_tensor_keys(node: dict) -> set[str]:
    tag = node["t"]
    if tag == "tensor":
        return {node["key"]}
    if tag == "dict":
        return set().union(*(_skeleton_tensor_keys(v) for _, v in node["items"]), set())
    if tag in ("list", "tuple"):
        return set().union(*(_skeleton_tensor_keys(v) for v in node["items"]), set())
    return set()


def _verify_safetensors_against(
    out_file: Path, tensors: dict[str, torch.Tensor], skeleton: dict
) -> None:
    from safetensors import safe_open

    with safe_open(str(out_file), framework="pt") as f:
        meta = f.metadata() or {}
        if json.loads(meta.get(_SKELETON_KEY, "null")) != skeleton:
            raise SystemExit(f"verify FAILED: skeleton metadata mismatch in {out_file}")
        disk_keys = set(f.keys())
        if disk_keys != set(tensors):
            raise SystemExit(
                f"verify FAILED: tensor key sets differ "
                f"(only-on-disk={sorted(disk_keys - set(tensors))[:5]}, "
                f"only-in-memory={sorted(set(tensors) - disk_keys)[:5]})"
            )
        for k in tensors:
            if not bitwise_tensor_equal(f.get_tensor(k), tensors[k]):
                raise SystemExit(f"verify FAILED: tensor {k!r} differs on disk")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        prog="pretrain.cli.dcp_safetensors",
        description=(
            "Convert a DCP checkpoint to a safetensors file and back. Given "
            "a full step_N/ checkpoint dir, the resume sidecar (per-rank "
            "RNG/batch-hasher state, meta.json, global_stream.json, ...) "
            "travels in the header and the inverse reconstructs a loadable "
            "checkpoint dir; given a bare dcp/ dir or --keys, only tensors "
            "are converted."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    to_st = sub.add_parser("to-safetensors", help="DCP dir -> .safetensors file")
    to_st.add_argument(
        "checkpoint",
        type=Path,
        help="checkpoint dir (containing dcp/) or the dcp dir itself",
    )
    to_st.add_argument("output", type=Path, help="output .safetensors path")
    to_st.add_argument(
        "--keys",
        nargs="+",
        default=None,
        metavar="KEY",
        help=(
            "only convert these top-level state-dict keys (e.g. --keys model "
            "to skip optimizer state — the full state dict is materialised "
            "in RAM, >100 GB with optim state at 8B)"
        ),
    )
    to_st.add_argument(
        "--verify",
        action="store_true",
        help="re-read the written file and compare every tensor bit-for-bit",
    )

    from_st = sub.add_parser("from-safetensors", help=".safetensors file -> DCP dir")
    from_st.add_argument("input", type=Path, help="input .safetensors path")
    from_st.add_argument(
        "output",
        type=Path,
        help=(
            "output dir (created; must not already contain files) — a full "
            "checkpoint dir when the file carries a resume sidecar, else a "
            "bare DCP dir"
        ),
    )
    from_st.add_argument(
        "--top-level-key",
        default=None,
        metavar="KEY",
        help=(
            "for foreign safetensors files (no pretrain skeleton): nest all "
            "tensors under this key, e.g. 'model' to match our {'model': ...} "
            "checkpoint layout"
        ),
    )
    from_st.add_argument(
        "--verify",
        action="store_true",
        help="offline-reload the written DCP dir and compare bit-for-bit",
    )

    args = p.parse_args()
    if args.cmd == "to-safetensors":
        stats = dcp_to_safetensors(
            args.checkpoint, args.output, keys=args.keys, verify=args.verify
        )
        print(
            f"wrote {stats['tensors']} tensors "
            f"({stats['bytes'] / 1e9:.2f} GB) from top-level keys "
            f"{stats['top_level_keys']} to {args.output}"
        )
    else:
        stats = safetensors_to_dcp(
            args.input,
            args.output,
            top_level_key=args.top_level_key,
            verify=args.verify,
        )
        print(
            f"wrote DCP checkpoint at {args.output} "
            f"({stats['tensors']} tensors, top-level keys {stats['top_level_keys']})"
        )


if __name__ == "__main__":
    main()
