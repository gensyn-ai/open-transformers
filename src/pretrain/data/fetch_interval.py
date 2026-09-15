"""Fetch only the dataset shards consumed in one audit interval.

To reproduce a cluster checkpoint interval bitwise (see
``pretrain.cli.audit_replay``) a user needs the data the canonical
:class:`~pretrain.data.global_stream.GlobalStream` consumes between the start
checkpoint and the next one — *not* the full multi-terabyte corpus. This module
works out exactly which shards that interval touches and pulls only those from a
GCS mirror of the dataset.

How it stays minimal **and** correct:

  * The canonical stream's *length-walk* needs only the ``.idx`` files
    (``document_length`` reads the offset table; it never faults the ``.bin``).
    So we download every source's manifest + ``.idx`` (a fraction of a percent of
    the corpus), then walk the stream forward from the checkpoint's saved
    :class:`GlobalStreamState` via
    :meth:`ShardedWindowView.walk_doc_refs` — the *same* refill/window cursor the
    audit drives — to enumerate every ``(source, shard)`` the interval reads.
  * Because that walk reuses ``_SourceWalker``, the shard set cannot drift from
    what the audit actually consumes.
  * We then download each touched ``.bin`` **whole** and verify it against the
    manifest's ``blake2b`` — a stronger guarantee than a replay (it proves the
    bytes equal the cluster's), so no materialising self-check is needed.

Output mirrors the on-disk dataset layout under ``<dest>/shards/<source>/`` so
the audit runs unchanged once its data-source paths point there.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Iterable

from pretrain.data.global_stream import GlobalStreamState, ShardedWindowView
from pretrain.data.loader import _resolve_sources
from pretrain.data.manifest import SourceManifest, blake2b_file

LOG = logging.getLogger("pretrain.fetch_interval")


# --------------------------------------------------------------------------- #
# GCS mirror access
# --------------------------------------------------------------------------- #


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/prefix/...`` into ``(bucket, prefix)`` (no trailing /)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected a gs:// URI, got {uri!r}")
    rest = uri[len("gs://") :]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"gs:// URI is missing a bucket: {uri!r}")
    return bucket, prefix.strip("/")


class GcsMirror:
    """Thin wrapper over a ``gs://bucket/prefix`` root of the dataset mirror.

    ``prefix`` is the GCS equivalent of the local ``data/shards`` directory: it
    contains one ``<source>/`` subdir per source, each holding ``manifest.yaml``
    plus the shard ``.idx`` / ``.bin`` objects.
    """

    def __init__(self, gcs_root: str) -> None:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import storage  # lazy: only needed for real fetches

        self.bucket_name, self.prefix = _parse_gs_uri(gcs_root)
        try:
            self._client = storage.Client()
        except DefaultCredentialsError:
            # A published audit mirror is world-readable, and an auditor
            # verifying it needs no Google account. ``storage.Client()``
            # nonetheless refuses to construct without credentials — it fails
            # before it ever learns the object is public — so an unauthenticated
            # volunteer would have to install the Cloud SDK and complete an
            # OAuth flow purely to satisfy this constructor. Fall back to the
            # anonymous client, which reads public objects fine.
            #
            # Credentialed callers are unaffected: the branch above still wins,
            # and it is the only one that can reach a private mirror. If the
            # mirror IS private, the anonymous client fails at the first request
            # with a 401/403 naming the object, which is a far better error than
            # a DefaultCredentialsError naming nothing.
            self._client = storage.Client.create_anonymous_client()
        self._bucket = self._client.bucket(self.bucket_name)

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def download(self, rel: str, dest: Path) -> None:
        """Download object ``<prefix>/<rel>`` to ``dest`` (parent dirs created)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = self._bucket.blob(self._key(rel))
        # Stage to a temp sibling then rename, so an interrupted download never
        # leaves a half-written file that a later run mistakes for complete.
        tmp = dest.with_suffix(dest.suffix + ".part")
        blob.download_to_filename(str(tmp))
        tmp.replace(dest)


# --------------------------------------------------------------------------- #
# Checkpoint descriptor + config
# --------------------------------------------------------------------------- #


def read_checkpoint_descriptor(ckpt_dir: str | Path) -> tuple[dict, GlobalStreamState]:
    """Read ``meta.json`` + ``global_stream.json`` from a checkpoint dir.

    No DCP / torch load — the interval is fully determined by the small JSON
    descriptors, so the user only needs those two files present locally.
    """
    ckpt_dir = Path(ckpt_dir)
    meta_obj = json.loads((ckpt_dir / "meta.json").read_text())
    if meta_obj.get("reduction_mode") != "deterministic_allgather":
        raise ValueError(
            "checkpoint was not produced in auditable mode "
            f"(reduction_mode={meta_obj.get('reduction_mode')!r}); its data stream "
            "is per-rank, not the canonical global stream this tool slices."
        )
    gpath = ckpt_dir / "global_stream.json"
    if not gpath.exists():
        raise FileNotFoundError(
            f"{gpath} missing — auditable checkpoints write the canonical stream "
            "position here; without it the interval cannot be reconstructed."
        )
    blob = json.loads(gpath.read_text())
    state = GlobalStreamState(
        consumed_documents_per_source=blob["consumed_documents_per_source"],
        epoch_per_source=blob["epoch_per_source"],
        mix_rng_state=blob.get("mix_rng_state"),
        carry_over=blob.get("carry_over") or [],
        windows_emitted=blob.get("windows_emitted", 0),
    )
    return meta_obj, state


def load_run_config(meta_obj: dict, config_name: str | None):
    """Resolve the run's :class:`RootConfig` from meta (or an override name)."""
    if config_name is not None:
        from pretrain.config import load_config

        return load_config(config_name)
    from pretrain.config import parse_config_resolved

    return parse_config_resolved(meta_obj["config_resolved"])


def rebase_sources(data_cfg, data_root: str | Path):
    """A copy of ``data_cfg`` whose source paths point under ``data_root``.

    Each source keeps its basename (the layout ``fetch_audit_data`` writes) and
    every other field is carried over by ``model_copy``, so a field added to
    ``DataConfig`` later cannot be silently reset to its default here and pack a
    different stream than the run trained on.
    """
    out = data_cfg.model_copy(deep=True)
    for s in out.sources:
        s.path = str(Path(data_root) / Path(s.path).name)
    return out


# --------------------------------------------------------------------------- #
# Local layout helpers
# --------------------------------------------------------------------------- #


def _source_basename(src_path: str) -> str:
    """Per-source subdir name — the basename of the run's data-source path.

    A ``gsutil rsync`` of the PVC mirrors this exact name under the GCS root and
    we recreate it under ``<dest>/shards/`` so the layout round-trips.
    """
    return Path(src_path).name


def _local_source_dir(dest: Path, basename: str) -> Path:
    return dest / "shards" / basename


# --------------------------------------------------------------------------- #
# Phase 1: manifests + indices
# --------------------------------------------------------------------------- #


def _fetch_manifest_and_indices(
    mirror: GcsMirror,
    *,
    basename: str,
    local_dir: Path,
    verify: bool,
) -> SourceManifest:
    """Download a source's ``manifest.yaml`` + every shard ``.idx`` locally.

    The manifest's shard ``prefix`` is rewritten to a bare stem so the local
    readers resolve ``<local_dir>/<stem>.{idx,bin}`` regardless of whether the
    original (PVC) manifest stored absolute or relative prefixes. Returns the
    rewritten manifest.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = local_dir / "manifest.yaml"
    mirror.download(f"{basename}/manifest.yaml", manifest_path)
    manifest = SourceManifest.load(manifest_path)

    # Rewrite prefixes to local stems and save back, then fetch each .idx.
    for shard in manifest.shards:
        stem = Path(shard.prefix).name
        shard.prefix = stem
        idx_dest = local_dir / f"{stem}.idx"
        if not idx_dest.exists():
            mirror.download(f"{basename}/{stem}.idx", idx_dest)
        if verify and shard.idx_blake2b:
            got = blake2b_file(idx_dest)
            if got != shard.idx_blake2b:
                raise RuntimeError(
                    f"{idx_dest} blake2b mismatch: manifest={shard.idx_blake2b} "
                    f"got={got} — the GCS mirror is corrupt or stale for this shard."
                )
    manifest.save(manifest_path)
    return manifest


# --------------------------------------------------------------------------- #
# Phase 2: enumerate the interval's documents
# --------------------------------------------------------------------------- #


def enumerate_touched_shards(
    cfg,
    stream_state: GlobalStreamState | None,
    *,
    dest: Path,
    start_consumed_tokens: int,
    start_step: int,
    target_consumed_tokens: int | None,
    until_step: int | None,
    seed: int,
) -> dict[str, set[int]]:
    """Walk the canonical stream over the interval; return touched shard ids.

    Builds an ``index_only`` view against the already-downloaded manifests +
    ``.idx`` (no ``.bin`` needed) and drives
    :meth:`ShardedWindowView.walk_doc_refs`. Returns ``{source_name: {shard_id}}``.
    """
    from pretrain.config.schema import DataConfig, DataSourceConfig

    local_sources = [
        DataSourceConfig(
            name=s.name,
            path=str(_local_source_dir(dest, _source_basename(s.path))),
            weight=s.weight,
        )
        for s in cfg.data.sources
    ]
    local_data = DataConfig(
        sources=local_sources,
        seq_len=cfg.data.seq_len,
        document_separator_id=cfg.data.document_separator_id,
        pack_strategy=cfg.data.pack_strategy,
        weights_are_token_shares=cfg.data.weights_are_token_shares,
    )
    manifests, manifest_dirs, weights = _resolve_sources(local_data)

    view = ShardedWindowView(
        manifests,
        manifest_dirs,
        weights,
        cfg.train,
        seed=seed,
        rank=0,
        world_size=1,
        eos_id=cfg.data.document_separator_id,
        state=stream_state,
        start_consumed_tokens=start_consumed_tokens,
        start_step=start_step,
        index_only=True,
    )

    touched: dict[str, set[int]] = {}
    for name, shard_id, _local in view.walk_doc_refs(
        target_consumed_tokens=target_consumed_tokens, until_step=until_step
    ):
        touched.setdefault(name, set()).add(shard_id)
    return touched


# --------------------------------------------------------------------------- #
# Phase 3: download touched payloads
# --------------------------------------------------------------------------- #


def _fetch_bins(
    mirror: GcsMirror,
    *,
    basename: str,
    local_dir: Path,
    manifest: SourceManifest,
    shard_ids: Iterable[int],
    verify: bool,
) -> int:
    """Download the whole ``.bin`` for each touched shard; verify blake2b.

    Returns total bytes that landed on disk this call (0 for already-present,
    verified shards).
    """
    bytes_fetched = 0
    for sid in sorted(shard_ids):
        shard = manifest.shards[sid]
        stem = shard.prefix  # already a local stem (rewritten in phase 1)
        bin_dest = local_dir / f"{stem}.bin"
        if bin_dest.exists():
            # A real download OR a sparse placeholder from a prior smaller-interval
            # run (see _stub_untouched_bins) may be here; blake2b tells them apart.
            # A stub (zeros) fails the manifest hash and is re-downloaded, so a
            # shard that becomes touched on a later run always gets real bytes.
            if verify and shard.bin_blake2b and blake2b_file(bin_dest) != shard.bin_blake2b:
                LOG.warning("%s present but blake2b mismatched — re-downloading", bin_dest)
                bin_dest.unlink()
            else:
                continue
        mirror.download(f"{basename}/{stem}.bin", bin_dest)
        bytes_fetched += bin_dest.stat().st_size
        if verify and shard.bin_blake2b:
            got = blake2b_file(bin_dest)
            if got != shard.bin_blake2b:
                raise RuntimeError(
                    f"{bin_dest} blake2b mismatch: manifest={shard.bin_blake2b} "
                    f"got={got} — the GCS mirror is corrupt or stale for this shard."
                )
    return bytes_fetched


def _stub_untouched_bins(
    manifest: SourceManifest, local_dir: Path, touched_ids: set[int]
) -> int:
    """Create sparse placeholder ``.bin`` files for shards the interval did NOT
    consume. Returns the number created.

    The audit's loader (``_SourceWalker``) opens a reader for EVERY shard in the
    manifest at construction, and a full (payload) reader requires the ``.bin``
    to exist (``IndexedDatasetReader``). The interval only reads the touched
    shards, so we download just those and place a correctly-sized **sparse** file
    (zero disk blocks until written) for the rest. They are never read — the
    canonical walk only calls ``document()`` on touched shards — so their
    contents don't matter; if the walk ever did touch one, ``document()`` returns
    zeros, surfacing as a loud audit hash mismatch rather than silent corruption.
    This is an audit-only convenience and writes nothing outside ``dest``.
    """
    import numpy as np

    from pretrain.data.indexed_dataset import IndexedDatasetReader

    created = 0
    for sid, shard in enumerate(manifest.shards):
        if sid in touched_ids:
            continue
        stem = shard.prefix
        bin_dest = local_dir / f"{stem}.bin"
        if bin_dest.exists():
            continue  # real bytes from a prior run, or already stubbed
        # Size the placeholder to the shard's exact payload length so the reader's
        # mmap (whole-file / itemsize) is consistent with the .idx token offsets.
        r = IndexedDatasetReader(local_dir / stem, index_only=True)
        nbytes = int(r.token_count) * np.dtype(r.dtype).itemsize
        with open(bin_dest, "wb") as f:
            if nbytes > 0:
                f.truncate(nbytes)  # sparse: no disk used until written
        created += 1
    return created


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FetchResult:
    dest: str
    data_root: str            # pass this to the audit as the source-path parent
    start_step: int
    start_consumed_tokens: int
    target_consumed_tokens: int | None
    until_step: int | None
    touched_shards: dict[str, list[int]]
    bytes_fetched: int


def fetch_audit_interval(
    checkpoint: str | Path,
    gcs_root: str,
    dest: str | Path = "./data/audit_data",
    *,
    config_name: str | None = None,
    until_step: int | None = None,
    from_init: bool = False,
    verify: bool = True,
    mirror: "GcsMirror | None" = None,
) -> FetchResult:
    """Download exactly the shards the audit interval starting at ``checkpoint``
    consumes, from the ``gcs_root`` mirror, into ``dest``.

    The interval ends one ``ckpt_every_tokens`` after the checkpoint (the audit's
    default target) unless ``until_step`` is given. ``mirror`` lets a caller (or
    test) inject an object exposing ``download(rel, dest)``; by default a real
    :class:`GcsMirror` is built from ``gcs_root``.

    ``from_init=True`` matches ``audit_replay --from-init``: the replay starts at
    the canonical stream's ORIGIN (step 0), not the checkpoint's saved position,
    so we walk from a fresh stream up to ``until_step`` (defaulting to the
    checkpoint's own step) and fetch the shards the FIRST ``until_step`` steps
    consume. The checkpoint is then used only for its meta (seed/config/step) —
    not its stream position.
    """
    dest = Path(dest)
    meta_obj, stream_state = read_checkpoint_descriptor(checkpoint)
    cfg = load_run_config(meta_obj, config_name)

    seed = int(meta_obj["seed"])
    target_consumed = None
    if from_init:
        # Walk from the stream origin (fresh state, step 0) — NOT the checkpoint's
        # forward interval — so step 0's shards (e.g. *_w00_00000.bin) are fetched.
        stream_state = None
        start_step = 0
        start_consumed = 0
        if until_step is None:
            until_step = int(meta_obj["step"])
    else:
        start_step = int(meta_obj["step"])
        start_consumed = int(meta_obj["consumed_tokens"])
        # Interval target: an explicit ``until_step`` wins; otherwise default to one
        # checkpoint interval. Step-based when ckpt_every_steps > 0 (matches the loop
        # + audit_replay), so enumerate_touched_shards walks the batch schedule step
        # by step (phase-correct token count); else the legacy token interval.
        if until_step is None:
            if cfg.train.ckpt_every_steps > 0:
                until_step = start_step + cfg.train.ckpt_every_steps
            else:
                target_consumed = start_consumed + cfg.train.ckpt_every_tokens
    LOG.info(
        "fetch interval: start step=%d consumed=%d → %s (seed=%d)",
        start_step,
        start_consumed,
        f"step {until_step}" if until_step is not None else f"consumed {target_consumed}",
        seed,
    )

    if mirror is None:
        mirror = GcsMirror(gcs_root)

    # Phase 1: manifests + all .idx (cheap; needed to build the permutation).
    manifests_by_basename: dict[str, SourceManifest] = {}
    name_to_basename: dict[str, str] = {}
    for s in cfg.data.sources:
        basename = _source_basename(s.path)
        name_to_basename[s.name] = basename
        local_dir = _local_source_dir(dest, basename)
        LOG.info("source %s: fetching manifest + indices → %s", s.name, local_dir)
        manifests_by_basename[basename] = _fetch_manifest_and_indices(
            mirror, basename=basename, local_dir=local_dir, verify=verify
        )

    # Phase 2: walk the interval to find which shards it touches.
    touched = enumerate_touched_shards(
        cfg,
        stream_state,
        dest=dest,
        start_consumed_tokens=start_consumed,
        start_step=start_step,
        target_consumed_tokens=target_consumed,
        until_step=until_step,
        seed=seed,
    )
    n_shards = sum(len(v) for v in touched.values())
    LOG.info(
        "interval touches %d shard(s) across %d source(s): %s",
        n_shards,
        len(touched),
        {k: len(v) for k, v in touched.items()},
    )

    # Phase 3: download whole .bin for each touched shard (blake2b-verified).
    total_bytes = 0
    for name, shard_ids in touched.items():
        basename = name_to_basename[name]
        local_dir = _local_source_dir(dest, basename)
        total_bytes += _fetch_bins(
            mirror,
            basename=basename,
            local_dir=local_dir,
            manifest=manifests_by_basename[basename],
            shard_ids=shard_ids,
            verify=verify,
        )
    LOG.info("downloaded %.2f GB of shard payloads", total_bytes / 1e9)

    # Phase 3b: the audit loader opens a reader for EVERY shard of EVERY source at
    # construction, so each must have a .bin present even though the interval reads
    # only the touched ones. Place sparse placeholders for the untouched shards
    # (incl. whole sources the interval never picked). Audit-only; never read.
    n_stub = 0
    touched_by_basename: dict[str, set[int]] = {}
    for name, ids in touched.items():
        touched_by_basename.setdefault(name_to_basename[name], set()).update(ids)
    for basename, manifest in manifests_by_basename.items():
        local_dir = _local_source_dir(dest, basename)
        n_stub += _stub_untouched_bins(
            manifest, local_dir, touched_by_basename.get(basename, set())
        )
    if n_stub:
        LOG.info(
            "placed %d sparse placeholder .bin(s) for untouched shards so the "
            "loader can open all readers (never read; zero disk)", n_stub,
        )

    result = FetchResult(
        dest=str(dest),
        data_root=str(dest / "shards"),
        start_step=start_step,
        start_consumed_tokens=start_consumed,
        target_consumed_tokens=target_consumed,
        until_step=until_step,
        touched_shards={k: sorted(v) for k, v in touched.items()},
        bytes_fetched=total_bytes,
    )
    (dest / "fetch_manifest.json").write_text(
        json.dumps(dataclasses.asdict(result), indent=2)
    )
    return result
