#!/usr/bin/env python3
"""Build the pretraining corpus end-to-end.

Pulls the four sources from `plan/03 §1` (DCLM-Baseline, FineWeb-Edu,
Stack v2 permissive via SWH, Proof-Pile-2 across its three configs),
writes raw JSONL into ``data/raw/``, then tokenises into Megatron-style
indexed binary shards under ``data/shards/<src>_proxy/``.

Also hosts the midtraining source ``dolma3_dolmino_mix`` (the full
allenai/dolma3_dolmino_mix-100B-1125 annealing pool — INGREDIENT1, all 23
sources: math, code, QA, reasoning, instruction-tuning, natural web/PDF —
NOT a math-only subset). It is NOT part of the default pretraining plan or
presets — pull it explicitly with ``--only``:

    python scripts/build_corpus.py --shard-suffix '' \\
        --only dolma3_dolmino_mix --docs-dolma3 999999999999

(This repo publishes no per-source doc/token count, so there is no exact
"whole mixture" threshold to name — pass a target far above the true total
and the round-robin puller exhausts every source directory on its own,
which just means "take everything". A smaller target still samples
round-robin across every source directory in lockstep, one file at a time
per source per round, so a partial pull isn't skewed toward whichever
source glob-sorts first.)

Idempotent: each source is skipped if its shard manifest already exists.

Presets:
    100m_proxy  — sized for the 100M smoke / Spark run
    1b_proxy    — sized for the 1B proxy run on H100x8

Usage:
    huggingface-cli login                          # one-time, gated datasets
    python scripts/build_corpus.py --preset 100m_proxy
    python scripts/build_corpus.py --preset 1b_proxy
    python scripts/build_corpus.py --preset 100m_proxy --only stack_v2_permissive
    python scripts/build_corpus.py --docs-dclm 300000 --docs-fineweb 70000 \\
        --docs-stack 200000 --docs-proof 15000

Prereqs:
    - HF token with licence accepted for DCLM-Baseline + Stack v2 dedup.
    - tokenizer.json at --tokenizer (default: data/tokenizer.json).
    - boto3 (anonymous SWH S3 access for Stack v2 content).
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from pathlib import Path

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config
from datasets import load_dataset
from huggingface_hub import HfApi

LOG = logging.getLogger("build_corpus")

# Pinned HF dataset revisions for reproducibility. Resolved to a concrete
# commit SHA at pull time and recorded in a `.revision` sidecar so the
# manifest can prove which version produced the raw JSONL. To pin to a
# specific SHA: replace "main" with the SHA from a prior successful run's
# log line ("[stack] resolved … @ <sha>").
STACK_REPO = "bigcode/the-stack-v2-dedup"
# Pinned to the SHA the original 1T pull resolved (captured from the
# data/raw/*.revision sidecars before the shared volume was wiped 2026-06-02), so
# the scaled-down 0.5T re-pull streams a byte-identical metadata order.
STACK_REVISION = "94d47b4385264b30f228e28a5d63e9b2eee8c2c5"

PRESETS: dict[str, dict[str, int]] = {
    "100m_proxy": dict(dclm=300_000,   fineweb=70_000,  stack=200_000,   proof=15_000),
    "1b_proxy":   dict(dclm=3_000_000, fineweb=700_000, stack=2_000_000, proof=150_000),
}

# Proof-Pile-2 sub-corpus weights (research/01 §2.13: 29B + 15B + 11B = 55B tokens).
PROOF_CONFIGS: dict[str, float] = {
    "arxiv":           29 / 55,
    "open-web-math":   15 / 55,
    "algebraic-stack": 11 / 55,
}

# Dolma 3 Dolmino full annealing mixture (allenai/dolma3_dolmino_mix-100B-1125)
# — "the high-quality pool of data considered for the second stage of Olmo 3
# 32B" training. 23 sources spanning math (synthetic), code, QA (synthetic),
# reasoning (synthetic), instruction-tuning (synthetic), and natural web/PDF
# data, laid out as one directory per source under data/<ingredient>-<source>/
# (all .jsonl.zst). NOT the OLMo 2 dolmino-mix-1124 math-only subset this
# replaced — that repo's data/math/<component>/ layout and published
# per-component doc counts do not carry over.
#
# The repo ships TWO independently-curated ~100B-token "ingredients"
# (data/ingredient1-*/, data/ingredient2-*/). This pulls INGREDIENT1 only —
# it is the complete one: ingredient2 is missing the tinymath-pot source and
# has a couple of inconsistently-named directories (wiki_to_rcqa_part1 with
# an underscore vs ingredient1's wiki_to_rcqa-part1 with a hyphen).
#
# Unlike the old math-only subset (7 hand-weighted components reconstructed
# from the OLMo 2 paper's table), this ingredient needs no reweighting: it
# IS AI2's own pre-built, already-proportioned mixture, so pulling every
# file under data/ingredient1-*/ whole reproduces it directly.
DOLMA3_DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
# Resolved to a concrete commit SHA at pull time and recorded in a
# `.revision` sidecar (same audit pattern as Stack v2). To pin: replace
# "main" with the SHA from a prior run's log/sidecar.
DOLMA3_DOLMINO_REVISION = "main"
DOLMA3_DOLMINO_INGREDIENT = "ingredient1"

SOURCES = ("dclm_baseline", "fineweb_edu", "stack_v2_permissive", "proof_pile_2")

# Midtraining-only sources: selectable via --only but never part of the
# default all-sources run or the pretraining presets, so a preset corpus
# build cannot silently grow a midtraining source.
MIDTRAIN_SOURCES = ("dolma3_dolmino_mix",)

# Sources whose raw JSONL must be GLOBALLY SHUFFLED before sharding.
# Multi-component sources are pulled component-by-component into one JSONL,
# so the raw file is block-ordered; write_shards cuts it in order, making
# each ~1 GiB shard component-pure; and the sampler's per-epoch permutation
# is SHARD-granular (mix_sampler._build_global_perm: shard-level shuffle,
# then within-shard shuffle) — so without a raw-level shuffle the component
# blocks survive into the training stream as ~57-optimizer-step homogeneous
# stretches (the loss_ce square wave on the first midtrain anneal,
# 20260807). The four pretraining sources don't need this: each is
# internally homogeneous and the mix RNG interleaves across sources
# per-document. dolma3_dolmino_mix's own puller already interleaves its
# 146 source directories round-robin at pull time (see
# pull_dolma3_dolmino_mix), which is coarser-grained than a true global
# shuffle — keep it in SHUFFLED_SOURCES too so within-directory blocks
# don't survive into shard-granular sampling either.
SHUFFLED_SOURCES = ("dolma3_dolmino_mix",)

# Resume cadence for the streaming pullers (dclm, fineweb). After every
# _RESUME_CHUNK successful JSONL writes we flush the file handle and
# fsync the .scanned sidecar; on a crash mid-chunk we lose at most this
# many writes' worth of progress, capped at sub-percent duplicates on
# resume. Pairs with _resume_offset, which truncates the .part to the
# last chunk_size boundary so .part and .scanned stay mutually consistent.
_RESUME_CHUNK = 10_000


def _scanned_path(out: Path) -> Path:
    """Sidecar tracking raw HF-row cursor for a resumable puller."""
    return out.parent / (out.name + ".scanned")


def _revision_path(out: Path) -> Path:
    """Sidecar recording the HF dataset commit SHA used to produce ``out``."""
    return out.parent / (out.name + ".revision")


def _shuffled_path(out: Path) -> Path:
    """Sidecar marking that ``out`` has been globally shuffled (records the
    seed/buckets used). Its presence is the idempotency gate for
    ``shuffle_jsonl`` in the orchestration loop."""
    return out.parent / (out.name + ".shuffled")


def _resolve_revision(repo_id: str, revision: str) -> str:
    """Resolve a branch/tag/SHA to a concrete commit SHA via the HF API.

    Pass-through for literal SHAs; resolves "main"/branch names to the
    current HEAD SHA so even unpinned runs record an exact version.
    """
    return HfApi().dataset_info(repo_id, revision=revision).sha


def _read_scanned(out: Path) -> int:
    p = _scanned_path(out)
    if not p.is_file():
        return 0
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return 0


def _write_scanned(out: Path, n: int) -> None:
    p = _scanned_path(out)
    tmp = p.parent / (p.name + ".tmp")
    tmp.write_text(f"{n}\n")
    tmp.replace(p)


def _resume_offset(out: Path, chunk_size: int = _RESUME_CHUNK) -> int:
    """Valid JSONL line count in ``out`` after truncating to a chunk boundary.

    Resumable pullers call this on entry to discover how many docs the
    prior attempt managed to write. Each scanned line is JSON-validated;
    the first bad/partial line ends the count and the file is truncated
    there. We then floor to ``chunk_size`` so the written count is in
    lock-step with the periodically-checkpointed ``.scanned`` cursor.
    Pre-existing .part files written before this fix landed (no chunk
    alignment, no .scanned) are handled gracefully — they're rounded
    down to the last chunk_size boundary on first entry.
    """
    if not out.is_file() or out.stat().st_size == 0:
        return 0

    valid_lines = 0
    valid_bytes = 0
    with out.open("rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                json.loads(line)
            except json.JSONDecodeError:
                break
            valid_bytes += len(line)
            valid_lines += 1

    aligned = (valid_lines // chunk_size) * chunk_size

    # Already aligned and clean — no truncation needed.
    if aligned == valid_lines and valid_bytes == out.stat().st_size:
        return valid_lines

    # Re-scan to find the byte offset of the ``aligned``-th line, then
    # truncate. Cheaper than seeking line-by-line for the common case
    # where ``aligned`` is close to ``valid_lines``.
    target_bytes = 0
    n = 0
    with out.open("rb") as f:
        for line in f:
            if n >= aligned:
                break
            target_bytes += len(line)
            n += 1
    if target_bytes < out.stat().st_size:
        with out.open("rb+") as f:
            f.truncate(target_bytes)
    return aligned


@dataclass
class Paths:
    raw_dir: Path
    shard_dir: Path
    tokenizer: Path
    # Suffix appended to each per-source shard subdir name. Defaults to
    # ``_proxy`` for backwards compatibility with existing prep manifests +
    # configs/data/proxy.yaml. The 1T prep manifest passes ``""`` so the
    # shards land at ``data/shards/<src>/`` matching configs/data/recipe_v1.yaml.
    shard_suffix: str = "_proxy"

    def raw_jsonl(self, src: str) -> Path:
        return self.raw_dir / f"{src}.jsonl"

    def shard_subdir(self, src: str) -> Path:
        return self.shard_dir / f"{src}{self.shard_suffix}"

    def manifest(self, src: str) -> Path:
        return self.shard_subdir(src) / "manifest.yaml"


# ---------- Pullers ----------

def pull_dclm(n: int, out: Path) -> None:
    """Resumable streaming pull.

    On entry we count valid JSONL already in ``out`` (truncated to the
    last _RESUME_CHUNK boundary) and read the sidecar ``.scanned`` cursor
    to fast-forward past raw HF rows the prior attempt already consumed.
    The empty-text filter on this source is sub-1%, so written ≈ scanned;
    we still record both so resume is robust to filter drift.
    """
    written = _resume_offset(out)
    if written >= n:
        LOG.info("[dclm] already have %d >= %d docs in %s — done", written, n, out)
        return
    scanned = _read_scanned(out)
    LOG.info(
        "[dclm] streaming until %d docs in %s (resume: written=%d scanned=%d)",
        n, out, written, scanned,
    )
    ds = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)
    if scanned > 0:
        ds = ds.skip(scanned)
    i = scanned
    with out.open("a", encoding="utf-8") as f:
        for ex in ds:
            i += 1
            text = ex.get("text")
            if not text:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            written += 1
            if written % _RESUME_CHUNK == 0:
                f.flush()
                _write_scanned(out, i)
            if written >= n:
                break
        f.flush()
        _write_scanned(out, i)
    LOG.info("[dclm] %d docs in %s (scanned %d HF rows)", written, out, i)


def pull_fineweb(n: int, out: Path) -> None:
    """Resumable streaming pull with int_score>=3 filter (~25% keep rate).

    The high drop rate makes the .scanned sidecar load-bearing: a naive
    resume that re-skips by written-count would re-scan ~4x as many raw
    rows as needed. We persist the raw cursor so resume fast-forwards
    correctly past the filtered rows.
    """
    written = _resume_offset(out)
    if written >= n:
        LOG.info("[fineweb] already have %d >= %d docs in %s — done", written, n, out)
        return
    scanned = _read_scanned(out)
    LOG.info(
        "[fineweb] streaming until %d docs in %s (resume: written=%d scanned=%d)",
        n, out, written, scanned,
    )
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    if scanned > 0:
        ds = ds.skip(scanned)
    i = scanned
    with out.open("a", encoding="utf-8") as f:
        for ex in ds:
            i += 1
            if ex.get("int_score", 0) < 3:
                if i % 50_000 == 0:
                    LOG.info("[fineweb] scanned=%d written=%d", i, written)
                continue
            text = ex.get("text")
            if not text:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            written += 1
            if written % _RESUME_CHUNK == 0:
                f.flush()
                _write_scanned(out, i)
            if written >= n:
                break
        f.flush()
        _write_scanned(out, i)
    LOG.info("[fineweb] %d docs in %s (scanned %d HF rows)", written, out, i)


def _split_by_weight(total: int, weights: dict[str, float]) -> dict[str, int]:
    cfgs = list(weights.items())
    counts: dict[str, int] = {}
    for cfg, w in cfgs[:-1]:
        counts[cfg] = int(total * w)
    counts[cfgs[-1][0]] = total - sum(counts.values())
    return counts


def pull_proof(n: int, out: Path) -> None:
    """Stream proof-pile-2 .jsonl.zst directly from HF.

    The dataset's custom loader (``proof-pile-2.py`` on the Hub) raises
    ``zstd: Unknown frame descriptor`` in streaming mode under recent
    ``datasets`` releases. We bypass it: enumerate the .jsonl.zst files
    via ``HfFileSystem``, decompress with zstandard, and parse JSONL
    line by line.

    Not resumable: HfFileSystem.glob is not order-stable across runs, so
    a sidecar cursor wouldn't reliably point at the same byte stream
    after a restart. The outer caller leaves .part on exception; we
    truncate it here so a fresh attempt starts clean.
    """
    import io

    import zstandard as zstd
    from huggingface_hub import HfFileSystem

    if out.is_file():
        LOG.info("[proof] dropping partial .part from prior attempt (non-resumable)")
        out.unlink()
    _scanned_path(out).unlink(missing_ok=True)

    counts = _split_by_weight(n, PROOF_CONFIGS)
    LOG.info("[proof] config split: %s", counts)

    fs = HfFileSystem()
    total = 0
    with out.open("w", encoding="utf-8") as f_out:
        for cfg, target in counts.items():
            base = f"datasets/EleutherAI/proof-pile-2/{cfg}"
            files = sorted(fs.glob(f"{base}/**/*.jsonl.zst"))
            if not files:
                LOG.warning("[proof:%s] no .jsonl.zst under %s", cfg, base)
                continue
            LOG.info("[proof:%s] %d files; target %d docs", cfg, len(files), target)
            written = 0
            for path in files:
                if written >= target:
                    break
                try:
                    with fs.open(path, "rb") as raw:
                        dctx = zstd.ZstdDecompressor()
                        with dctx.stream_reader(raw) as reader:
                            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                                if not line.strip():
                                    continue
                                try:
                                    ex = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                text = ex.get("text")
                                if not text:
                                    continue
                                f_out.write(json.dumps({"text": text}) + "\n")
                                written += 1
                                if written >= target:
                                    break
                except Exception as e:
                    LOG.warning("[proof:%s] skipped %s: %s", cfg, path, e)
                    continue
            total += written
            LOG.info("[proof:%s] wrote %d / %d docs", cfg, written, target)
    LOG.info("[proof] wrote %d docs to %s", total, out)


def _is_eval_split(path: str) -> bool:
    """True if ``path`` looks like a held-out (test/valid/dev) file.

    Dolma3-Dolmino's math sources are flat — files like
    ``dolmino_math_gsm8k_socratic_test_0.jsonl.jsonl.zst`` embed the split
    name IN the filename, not as its own path segment, so a plain
    ``seg in ("test", ...) for seg in path.split("/")`` check (the old
    dolmino-mix-1124 puller's approach — that repo used one directory per
    split) silently lets them through. Match on a token boundary
    (``_``/``.``/``-``/``/``/start/end) across the whole path instead, so
    this still rejects a true ``.../test/...`` directory segment too.
    """
    return bool(re.search(r"(?:^|[_./\-])(test|valid|validation|dev)(?:[_./\-]|$)",
                           path, re.IGNORECASE))


def pull_dolma3_dolmino_mix(n: int, out: Path, concurrency: int = 24) -> None:
    """Stream the full Dolma3-Dolmino annealing mixture (ingredient1) from HF.

    Same shape as ``pull_proof``/the old dolmino-mix-1124 puller this
    replaces: enumerate files via ``HfFileSystem``, decompress, parse
    JSONL line by line, write ``{"text": ...}`` records. Differences:

    * No per-source weighting — this pulls every file under
      ``data/ingredient1-*/`` (146 source directories at last count: math,
      code, QA, reasoning, instruction-tuning, natural web/PDF), which is
      already AI2's own pre-proportioned 100B-token mixture.
    * CONCURRENT fetches (``concurrency`` in flight, ``pull_stack``'s
      ordered-window pattern: each file is submitted with a monotonic
      index, completions are buffered until they can be written in
      submission order, so output is deterministic regardless of fetch
      latency). Single-connection sequential fetching measured ~8 MB/s in
      practice — and worse, ONE stalled/503'd request blocks the entire
      pull (observed: a single 503 held up progress for ~50 minutes with
      nothing else in flight to make progress on). Concurrency fixes both:
      aggregate throughput scales with in-flight requests, and a slow
      straggler no longer blocks everything behind it.
    * Submission order is a ROUND-ROBIN across source directories
      (rotating the start each round), one file per directory per round,
      computed upfront as a flat list rather than exhausting each
      directory in turn — so a target smaller than the true total spreads
      across every source proportionally rather than draining whichever
      ones glob-sort first (the same "component blocking" failure
      SHUFFLED_SOURCES already guards against downstream, at directory
      instead of shard granularity). This is a per-ROUND guarantee, not
      per-file: a whole file is one unit of work, so if ``n`` is reached
      mid-round the dirs after that point in THAT round's (rotated) order
      get skipped for it — immaterial once a round covers many docs per
      dir (the intended default: ``n`` far above the unpublished true
      total, so no round ever gets cut short).
    * All files are ``.jsonl.zst`` (no mixed compression to dispatch on),
      but ``open_text`` still handles ``.gz``/plain for robustness.
    * ``_is_eval_split`` (filename-boundary match, not path-segment) excludes
      eval-contaminating splits — see its docstring.
    * The repo revision is resolved and recorded in a ``.revision`` sidecar
      (Stack v2 audit pattern); the resolved SHA is baked into the glob so
      every source lists one consistent snapshot.

    Not resumable (glob order is not stable across runs); truncates any
    prior .part on entry.
    """
    import io

    import zstandard as zstd
    from huggingface_hub import HfFileSystem

    if out.is_file():
        LOG.info("[dolma3] dropping partial .part from prior attempt (non-resumable)")
        out.unlink()
    _scanned_path(out).unlink(missing_ok=True)
    _revision_path(out).unlink(missing_ok=True)

    sha = _resolve_revision(DOLMA3_DOLMINO_REPO, DOLMA3_DOLMINO_REVISION)
    LOG.info("[dolma3] pulling %s @ %s (ingredient=%s, concurrency=%d)",
             DOLMA3_DOLMINO_REPO, sha, DOLMA3_DOLMINO_INGREDIENT, concurrency)
    _revision_path(out).write_text(sha + "\n")

    def open_text(fs: HfFileSystem, path: str):
        raw = fs.open(path, "rb")
        if path.endswith(".zst"):
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            return io.TextIOWrapper(reader, encoding="utf-8")
        if path.endswith(".gz"):
            return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8")
        return io.TextIOWrapper(raw, encoding="utf-8")

    fs = HfFileSystem()
    base = f"datasets/{DOLMA3_DOLMINO_REPO}@{sha}/data"
    src_dirs = sorted(fs.glob(f"{base}/{DOLMA3_DOLMINO_INGREDIENT}-*"))
    if not src_dirs:
        raise RuntimeError(f"[dolma3] no {DOLMA3_DOLMINO_INGREDIENT}-* dirs under {base}")

    # Per-source file queues, built upfront (not lazily) so the round-robin
    # below can pull one file per source per round without re-listing.
    # This listing pass is itself sequential (one glob per dir) but cheap
    # relative to the actual file fetches — ~162 metadata calls, not ~70k.
    queues: dict[str, list[str]] = {}
    for d in src_dirs:
        files = sorted(fs.glob(f"{d}/**/*.jsonl*"))
        files = [p for p in files if not _is_eval_split(p)]
        if files:
            queues[d] = list(files)
    LOG.info("[dolma3] %d/%d source dirs have files (%d files total)",
             len(queues), len(src_dirs), sum(len(v) for v in queues.values()))

    # Flatten into a single round-robin work order upfront (no I/O here —
    # just list bookkeeping), since concurrent submission needs a flat
    # queue rather than the sequential version's inline per-round loop.
    work: list[tuple[str, str]] = []
    remaining = {d: list(files) for d, files in queues.items()}
    active = list(remaining)
    round_num = 0
    while active:
        round_num += 1
        offset = (round_num - 1) % len(active)
        rotated = active[offset:] + active[:offset]
        next_active = []
        for d in rotated:
            work.append((d, remaining[d].pop(0)))
            if remaining[d]:
                next_active.append(d)
        active = next_active
    LOG.info("[dolma3] %d files queued in round-robin order", len(work))

    def fetch_one(d: str, path: str) -> list[str]:
        """Fetch+decode+parse one file; returns its JSON-encoded lines."""
        lines: list[str] = []
        try:
            with open_text(fs, path) as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = ex.get("text")
                    if not text:
                        continue
                    lines.append(json.dumps({"text": text}))
        except Exception as e:
            LOG.warning("[dolma3:%s] skipped %s: %s", d, path, e)
        return lines

    # Ordered-window drain (same pattern as pull_stack): each submitted
    # future is tagged with its monotonic submission idx; completions land
    # in `pending` keyed by idx; `next_write_idx` advances strictly in
    # submission (round-robin) order, so output is deterministic
    # regardless of which file happens to finish downloading first.
    fut_to_idx: dict[Future, int] = {}
    pending: dict[int, list[str]] = {}
    next_submit_idx = 0
    next_write_idx = 0
    max_inflight = concurrency * 2
    total = 0
    t0 = time.time()

    pool = ThreadPoolExecutor(max_workers=concurrency)
    f_out = out.open("w", encoding="utf-8")

    def harvest(done: set[Future]) -> None:
        for f in done:
            pending[fut_to_idx.pop(f)] = f.result()

    def drain_pending() -> bool:
        """Write ready-in-order results; True once `n` docs are written."""
        nonlocal total, next_write_idx
        while next_write_idx in pending:
            lines = pending.pop(next_write_idx)
            next_write_idx += 1
            for line in lines:
                f_out.write(line + "\n")
                total += 1
                if total >= n:
                    return True
        return False

    try:
        it = iter(work)
        for d, path in it:
            fut = pool.submit(fetch_one, d, path)
            fut_to_idx[fut] = next_submit_idx
            next_submit_idx += 1

            if len(fut_to_idx) >= max_inflight:
                done, _ = wait(list(fut_to_idx.keys()), return_when=FIRST_COMPLETED)
                harvest(done)
                if drain_pending():
                    break
                if next_submit_idx % 500 == 0:
                    rate = total / max(time.time() - t0, 1)
                    LOG.info("[dolma3] submitted=%d written=%d (%.0f docs/s, %.0fs elapsed)",
                             next_submit_idx, total, rate, time.time() - t0)

        if total < n and fut_to_idx:
            for fut in as_completed(list(fut_to_idx.keys())):
                harvest({fut})
                if drain_pending():
                    break
    finally:
        f_out.close()
        pool.shutdown(wait=False, cancel_futures=True)

    LOG.info("[dolma3] wrote %d docs to %s (%.0fs total)", total, out, time.time() - t0)


# ---------- Stack v2 (SWH S3 fetch) ----------

def _swh_client(pool_size: int):
    # `mode="adaptive"` switches boto3 to client-side rate-limit backoff:
    # on 503/SlowDown the client throttles itself with rapidly-growing
    # delays rather than retrying immediately. This is the right shape
    # for SWH's anonymous endpoint, which 503s aggressively when a single
    # IP submits a sudden burst of GETs (see the startup-burst pattern
    # in early build_corpus logs — failures concentrate in the first
    # few thousand fetches, then drop to zero). Paired with the reduced
    # default `--stack-concurrency 64`, the initial submission rate is
    # low enough that adaptive's first-second self-throttle absorbs the
    # burst without dropping blobs.
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=pool_size,
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )


def _fetch_blob(s3, blob_id: str, src_encoding: str | None) -> str | None:
    try:
        obj = s3.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
        body = gzip.decompress(obj["Body"].read())
    except Exception:
        return None
    return body.decode(src_encoding or "utf-8", errors="replace")


def pull_stack(
    n: int,
    out: Path,
    concurrency: int = 64,
    max_bytes: int = 1_000_000,
    worker_id: int = 0,
    num_workers: int = 1,
    log_tag: str = "stack",
) -> None:
    """Stream Stack v2 metadata + parallel SWH S3 blob fetches.

    Deterministic: each submitted blob fetch is tagged with its monotonic
    submission index, and completed fetches are buffered until they can be
    written in submission order. Same HF revision + same iterator + same
    submission filters → byte-identical JSONL across runs, regardless of
    SWH fetch-latency jitter. The pinned revision (``STACK_REVISION``) is
    resolved to a concrete commit SHA and written to a ``.revision``
    sidecar for audit.

    Not resumable: the outer caller leaves .part on exception; we truncate
    it here so a fresh attempt starts clean. (Resume could be added by
    checkpointing the next-write index, since ordering is now stable.)

    Fan-out: pass ``worker_id`` and ``num_workers > 1`` to restrict this
    worker to its share of the metadata via ``IterableDataset.shard()``,
    which partitions the underlying parquet *files* across workers (so
    each worker only downloads ~1/N of the metadata, not all of it).
    ``IterableDataset.skip()`` would be O(rows) — useless here since
    worker K would have to download and parse K/N of the dataset before
    fetching its first blob. Each worker still caps at ``n`` keepers
    (typically docs_stack / num_workers). ``log_tag`` distinguishes
    worker pods in shared logs (e.g. ``"stack:w03"``).
    """
    if out.is_file():
        LOG.info("[%s] dropping partial .part from prior attempt (non-resumable)", log_tag)
        out.unlink()
    _scanned_path(out).unlink(missing_ok=True)
    _revision_path(out).unlink(missing_ok=True)

    sha = _resolve_revision(STACK_REPO, STACK_REVISION)
    LOG.info(
        "[%s] streaming %s @ %s (shard %d/%d, target up to %d kept, concurrency=%d)",
        log_tag, STACK_REPO, sha, worker_id, num_workers, n, concurrency,
    )
    _revision_path(out).write_text(sha + "\n")

    s3 = _swh_client(pool_size=concurrency * 2)
    ds = load_dataset(STACK_REPO, split="train", streaming=True, revision=sha)
    if num_workers > 1:
        # IterableDataset.shard() was added in datasets 3.2.0; we pin
        # 3.0.1. split_dataset_by_node has been in datasets since 2.8.0
        # and does shard-level (file-level) partitioning when
        # n_shards >= world_size — which is the case here (the_stack-v2-
        # dedup ships thousands of parquet shards). Only falls back to
        # O(n) round-robin example partitioning if n_shards < world_size.
        from datasets.distributed import split_dataset_by_node
        ds = split_dataset_by_node(ds, rank=worker_id, world_size=num_workers)

    # Ordered-window drain: each submitted future is tagged with its
    # monotonic submission idx. Completed results land in `pending` keyed
    # by idx; `next_write_idx` advances strictly in submission order. The
    # writer is allowed to stall on a head-of-line straggler — by the
    # time it stalls there are already up to max_inflight other fetches
    # in flight, so throughput tracks the previous as_completed loop
    # closely while output ordering becomes deterministic.
    fut_to_idx: dict[Future, int] = {}
    pending: dict[int, str | None] = {}
    next_submit_idx = 0
    next_write_idx = 0
    max_inflight = concurrency * 4
    scanned = kept = fetch_failed = 0
    t0 = time.time()

    pool = ThreadPoolExecutor(max_workers=concurrency)
    f_out = out.open("w", encoding="utf-8")

    def harvest(done: set[Future]) -> None:
        for f in done:
            pending[fut_to_idx.pop(f)] = f.result()

    def drain_pending() -> bool:
        nonlocal kept, fetch_failed, next_write_idx
        while next_write_idx in pending:
            text = pending.pop(next_write_idx)
            next_write_idx += 1
            if text is None:
                fetch_failed += 1
                continue
            if not text.strip():
                continue
            f_out.write(json.dumps({"text": text}) + "\n")
            kept += 1
            if kept >= n:
                return True
        return False

    try:
        for ex in ds:
            scanned += 1
            if scanned % 10_000 == 0:
                rate = kept / max(time.time() - t0, 1)
                LOG.info(
                    "[%s] scanned=%d kept=%d fail=%d (%.0f kept/s)",
                    log_tag, scanned, kept, fetch_failed, rate,
                )
            if ex.get("license_type") != "permissive":
                continue
            blob_id = ex.get("blob_id")
            if not blob_id:
                continue
            length = ex.get("length_bytes") or 0
            if length and length > max_bytes:
                continue

            fut = pool.submit(_fetch_blob, s3, blob_id, ex.get("src_encoding"))
            fut_to_idx[fut] = next_submit_idx
            next_submit_idx += 1

            if len(fut_to_idx) >= max_inflight:
                done, _ = wait(list(fut_to_idx.keys()), return_when=FIRST_COMPLETED)
                harvest(done)
                if drain_pending():
                    break

        if kept < n and fut_to_idx:
            for fut in as_completed(list(fut_to_idx.keys())):
                harvest({fut})
                if drain_pending():
                    break
    finally:
        f_out.close()
        pool.shutdown(wait=False, cancel_futures=True)

    LOG.info(
        "[%s] wrote %d docs (scanned=%d, fail=%d, %.0fs) to %s",
        log_tag, kept, scanned, fetch_failed, time.time() - t0, out,
    )


# ---------- Global raw-level shuffle ----------

# Bucket count for shuffle_jsonl: peak memory during pass 2 is
# ~file_size/num_buckets (one bucket's lines in memory). The old
# dolmino_math math-only pull was ~60 GB raw; 64 buckets kept that at ~1 GB
# per bucket. dolma3_dolmino_mix is the FULL annealing pool — the dataset
# card lists 340 GB compressed .zst for BOTH ingredients, so ingredient1
# alone is on the order of ~170 GB compressed / very roughly ~500-700 GB
# raw JSONL (zst text ratio ~3-4x) — 10x-ish the old pull. 512 buckets
# keeps that back down to ~1-1.5 GB/bucket; bump further if the prep pod's
# memory limit can't cover it plus JSON-parsing overhead.
_SHUFFLE_BUCKETS = 512


def shuffle_jsonl(path: Path, *, seed: int, num_buckets: int = _SHUFFLE_BUCKETS) -> None:
    """Globally shuffle a JSONL file in place — deterministic, memory-bounded.

    Two-pass bucket shuffle. Pass 1 streams the file once, appending each
    line to one of ``num_buckets`` temp bucket files chosen by a seeded RNG
    (a uniformly-random bucket assignment). Pass 2 loads each bucket
    (~1/num_buckets of the file) fully into memory, permutes it with a
    per-bucket seeded RNG, and appends it to the output. Random assignment
    + within-bucket permutation compose to a uniform global permutation.
    Both passes are sequential I/O; peak memory is one bucket; peak disk is
    ~2x the input while the buckets exist. Deterministic for a given
    ``(seed, num_buckets)`` — CPython guarantees ``random.Random`` sequence
    stability for int seeds.

    Crash-safe, not resumable: output is built at ``<path>.shuffled.part``
    and atomically replaces ``path`` only on success, so an interrupted
    shuffle leaves the input untouched and a rerun starts clean (any stale
    ``.shuffle_tmp``/``.part`` from a prior attempt is removed on entry).

    Why this exists: see ``SHUFFLED_SOURCES``.
    """
    tmp_dir = path.parent / (path.name + ".shuffle_tmp")
    out_tmp = path.parent / (path.name + ".shuffled.part")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    out_tmp.unlink(missing_ok=True)
    tmp_dir.mkdir()

    t0 = time.time()
    rng_assign = random.Random(seed)
    bucket_paths = [tmp_dir / f"bucket_{i:04d}.jsonl" for i in range(num_buckets)]
    handles = [p.open("w", encoding="utf-8") for p in bucket_paths]
    lines_in = 0
    try:
        with path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                handles[rng_assign.randrange(num_buckets)].write(line)
                lines_in += 1
    finally:
        for h in handles:
            h.close()
    LOG.info(
        "[shuffle] %s: pass 1 bucketed %d lines into %d buckets (%.0fs)",
        path.name, lines_in, num_buckets, time.time() - t0,
    )

    lines_out = 0
    with out_tmp.open("w", encoding="utf-8") as f_out:
        for i, bpath in enumerate(bucket_paths):
            with bpath.open("r", encoding="utf-8") as f_b:
                lines = f_b.readlines()
            # Distinct deterministic stream per bucket (int-keyed, stable).
            random.Random(seed * 1_000_003 + i + 1).shuffle(lines)
            f_out.writelines(lines)
            lines_out += len(lines)
            bpath.unlink()
    shutil.rmtree(tmp_dir)

    if lines_out != lines_in:
        out_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"shuffle_jsonl: line count changed ({lines_in} in, {lines_out} "
            f"out) — refusing to replace {path}"
        )
    out_tmp.replace(path)
    LOG.info(
        "[shuffle] %s: pass 2 done — %d lines globally shuffled (%.0fs total)",
        path.name, lines_out, time.time() - t0,
    )


# ---------- Sharding ----------

def _merge_worker_manifests(shard_dir: Path, num_workers: int, out_manifest: Path) -> None:
    """Combine ``manifest.w00.yaml`` .. ``manifest.w<N-1>.yaml`` into one.

    Plain-YAML merge (not ``pretrain.data.manifest.SourceManifest`` — this
    script deliberately never imports the ``pretrain`` package, only
    shells out to it) that reproduces the exact schema
    ``SourceManifest.to_yaml()``/``from_yaml()`` use: same ``name`` /
    ``tokenizer_hash`` / ``dtype`` / ``filter_version`` across every
    worker (they tokenized disjoint byte-ranges of the SAME raw file with
    the SAME tokenizer, so these fields must agree — asserted below),
    ``shards`` concatenated (each worker's shard filenames are already
    ``_w<K>``-tagged so there's no collision), ``raw_jsonl_blake2b`` dicts
    merged (each worker hashes the same input file under its own
    ``{filename}`` key from ``prepare_data.py``'s ``_cmd_shard`` — same
    key from every worker, so this keeps whichever value is present; they
    are byte-identical since it's the same file).
    """
    manifests = []
    for k in range(num_workers):
        p = shard_dir / f"manifest.w{k:02d}.yaml"
        manifests.append(yaml.safe_load(p.read_text(encoding="utf-8")))

    for field in ("name", "tokenizer_hash", "dtype", "filter_version"):
        values = {m.get(field) for m in manifests}
        if len(values) > 1:
            raise RuntimeError(
                f"[shard] worker manifests disagree on {field!r}: {values} — "
                f"workers didn't all tokenize with the same config"
            )

    merged = {
        "name": manifests[0]["name"],
        "tokenizer_hash": manifests[0]["tokenizer_hash"],
        "dtype": manifests[0]["dtype"],
        "filter_version": manifests[0].get("filter_version", ""),
        "shards": [s for m in manifests for s in m["shards"]],
        "raw_jsonl_blake2b": {
            key: val for m in manifests for key, val in (m.get("raw_jsonl_blake2b") or {}).items()
        },
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

    total_docs = sum(s["num_documents"] for s in merged["shards"])
    total_tokens = sum(s["token_count"] for s in merged["shards"])
    LOG.info("[shard] merged %d worker manifests -> %s (%d shards, %d docs, %d tokens)",
             num_workers, out_manifest, len(merged["shards"]), total_docs, total_tokens)

    # Worker manifests are now redundant with the merged one; leave them
    # (cheap, and useful if the merge itself needs debugging) but drop
    # nothing else — the .bin/.idx shard files they reference stay in place.


def shard_source(
    src: str,
    paths: Paths,
    *,
    input_glob: str | None = None,
    shard_tag: str = "",
    parallel_workers: int = 1,
) -> None:
    """Shard a source's JSONL into Megatron-style indexed binary shards.

    ``input_glob`` overrides the default ``<src>.jsonl`` glob (used by
    the fan-out pattern where each worker reads its own per-worker
    JSONL like ``<src>_w<K>.jsonl``). ``shard_tag`` adds a suffix to the
    shard prefix and manifest filename so multiple workers contributing
    to the same source's shard directory don't collide.

    ``parallel_workers`` > 1 fans the SAME input file out across that many
    subprocesses IN THIS POD, each tokenizing a contiguous byte-range via
    ``prepare_data.py shard``'s own ``--num-workers``/``--worker-id``
    (``iter_text_from_jsonl_segment`` — already built for the Stack v2
    cross-POD fan-out; this reuses it for cross-PROCESS fan-out within one
    pod instead). Mutually exclusive with ``shard_tag`` — same constraint
    ``prepare_data.py shard`` enforces, since both are "tag the output"
    mechanisms and this source's outer orchestration never combines them
    (only Stack v2 uses the outer ``shard_tag`` for cross-pod naming, and
    it doesn't request ``parallel_workers`` here). Each subprocess's
    ``tokenizer.encode_batch`` already parallelizes internally (the HF
    ``tokenizers`` Rust lib's own thread pool) — ``RAYON_NUM_THREADS`` is
    capped per-subprocess to ``cores / parallel_workers`` so N subprocesses
    don't each try to claim every core and thrash. Blocks until every
    worker exits, then merges their manifests into the canonical one.
    """
    if parallel_workers <= 1:
        LOG.info("[shard] %s → %s (tag=%r)", src, paths.shard_subdir(src), shard_tag)
        cmd = [
            sys.executable, "-m", "pretrain.cli.prepare_data", "shard",
            "--input-dir", str(paths.raw_dir),
            "--glob", input_glob if input_glob else f"{src}.jsonl",
            "--format", "jsonl",
            "--tokenizer", str(paths.tokenizer),
            "--output-dir", str(paths.shard_subdir(src)),
            "--source-name", src,
        ]
        if shard_tag:
            cmd.extend(["--shard-tag", shard_tag])
        subprocess.check_call(cmd)
        return

    if shard_tag:
        raise ValueError(
            "shard_source: parallel_workers>1 and shard_tag are mutually "
            "exclusive (prepare_data.py shard enforces the same constraint "
            "between its own --num-workers and --shard-tag)"
        )
    LOG.info("[shard] %s → %s (%d parallel in-pod workers)",
              src, paths.shard_subdir(src), parallel_workers)

    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or parallel_workers
    per_worker_threads = max(1, cores // parallel_workers)

    procs: list[subprocess.Popen] = []
    for k in range(parallel_workers):
        cmd = [
            sys.executable, "-m", "pretrain.cli.prepare_data", "shard",
            "--input-dir", str(paths.raw_dir),
            "--glob", input_glob if input_glob else f"{src}.jsonl",
            "--format", "jsonl",
            "--tokenizer", str(paths.tokenizer),
            "--output-dir", str(paths.shard_subdir(src)),
            "--source-name", src,
            "--num-workers", str(parallel_workers),
            "--worker-id", str(k),
        ]
        env = dict(os.environ)
        env["RAYON_NUM_THREADS"] = str(per_worker_threads)
        env["TOKENIZERS_PARALLELISM"] = "true"
        procs.append(subprocess.Popen(cmd, env=env))

    failed = [k for k, p in enumerate(procs) if p.wait() != 0]
    if failed:
        raise RuntimeError(
            f"[shard] {src}: worker(s) {failed} exited non-zero — see their "
            f"subprocess output above for the actual error"
        )
    _merge_worker_manifests(paths.shard_subdir(src), parallel_workers, paths.manifest(src))


# ---------- Orchestration ----------

PULLERS = {
    "dclm_baseline":       pull_dclm,
    "fineweb_edu":         pull_fineweb,
    "stack_v2_permissive": pull_stack,
    "proof_pile_2":        pull_proof,
    "dolma3_dolmino_mix":  pull_dolma3_dolmino_mix,
}


_SRC_TO_FLAG_KEY = {
    "dclm_baseline":       "dclm",
    "fineweb_edu":         "fineweb",
    "stack_v2_permissive": "stack",
    "proof_pile_2":        "proof",
    "dolma3_dolmino_mix":  "dolma3",
}


def _resolve_counts(args: argparse.Namespace) -> dict[str, int]:
    counts = dict(PRESETS[args.preset]) if args.preset else {}
    overrides = {
        "dclm": args.docs_dclm,
        "fineweb": args.docs_fineweb,
        "stack": args.docs_stack,
        "proof": args.docs_proof,
        "dolma3": args.docs_dolma3,
    }
    for k, v in overrides.items():
        if v is not None:
            counts[k] = v
    # Only require counts for sources we'll actually run. Under --only, the
    # per-source parallel Job pattern (one Job per source YAML) passes
    # one --docs-* flag and expects the rest to be irrelevant.
    selected_srcs = args.only or list(SOURCES)
    required = {_SRC_TO_FLAG_KEY[s] for s in selected_srcs}
    missing = required - counts.keys()
    if missing:
        raise SystemExit(
            f"missing doc counts for {sorted(missing)} "
            f"(selected sources: {sorted(selected_srcs)}); "
            f"pass --preset or --docs-* flags"
        )
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--preset", choices=list(PRESETS))
    p.add_argument("--docs-dclm", type=int)
    p.add_argument("--docs-fineweb", type=int)
    p.add_argument("--docs-stack", type=int)
    p.add_argument("--docs-proof", type=int)
    p.add_argument("--docs-dolma3", type=int,
                   help="dolma3_dolmino_mix doc target (midtraining only, "
                        "requires --only dolma3_dolmino_mix); this repo "
                        "publishes no total doc/token count, so pass a "
                        "target far above the true total (e.g. 1000000000000) to "
                        "take the whole ~100B-token mixture — the "
                        "round-robin puller just exhausts every source "
                        "directory on its own and stops")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--shard-dir", default="data/shards")
    p.add_argument("--shard-suffix", default="_proxy",
                   help="suffix on per-source shard subdir name "
                        "(default '_proxy' for backwards compat; pass '' for the "
                        "recipe_v1 full-corpus path layout)")
    p.add_argument("--delete-raw", action="store_true",
                   help="delete the per-source JSONL after sharding "
                        "(default: keep raw JSONL for auditability/reproducibility)")
    p.add_argument("--shuffle-seed", type=int, default=0,
                   help="seed for the raw-level global shuffle applied to "
                        "SHUFFLED_SOURCES between pull and shard (recorded "
                        "in the .shuffled sidecar)")
    p.add_argument("--only", nargs="+", choices=SOURCES + MIDTRAIN_SOURCES,
                   help="restrict to these sources (default: the four "
                        "pretraining sources; midtraining sources like "
                        "dolma3_dolmino_mix run ONLY when named here)")
    p.add_argument("--stack-concurrency", type=int, default=64,
                   help="parallel SWH S3 fetches for Stack v2")
    p.add_argument("--dolma3-concurrency", type=int, default=24,
                   help="parallel HF file fetches for dolma3_dolmino_mix "
                        "(sequential fetching measured ~8 MB/s and lets a "
                        "single stalled/503'd request block the whole pull)")
    p.add_argument("--dolma3-shard-workers", type=int, default=8,
                   help="parallel IN-POD tokenizer subprocesses for "
                        "dolma3_dolmino_mix's shard step (byte-range "
                        "fan-out over the same raw JSONL, merged after; "
                        "see shard_source's parallel_workers)")
    # ---- Stack fan-out (multi-pod parallel pull+shard) ----
    # Each worker pod handles a contiguous metadata-row slice of
    # bigcode/the-stack-v2-dedup. Output JSONL and shards are tagged
    # with the worker id so siblings don't collide; a separate
    # merge-manifests step combines per-worker manifests into the
    # canonical data/shards/stack_v2_permissive/manifest.yaml.
    # Today only stack_v2_permissive supports fan-out via this script
    # (DCLM/FW pulls are streaming-resumable and already adequately
    # fast; Proof is small enough that a single pod is fine).
    p.add_argument("--worker-id", type=int, default=0,
                   help="zero-based worker index for Stack fan-out pull")
    p.add_argument("--num-workers", type=int, default=1,
                   help="total fan-out workers (1 = single-worker, no slicing)")
    args = p.parse_args()

    if args.num_workers > 1:
        if not (0 <= args.worker_id < args.num_workers):
            raise SystemExit(
                f"--worker-id={args.worker_id} out of range for "
                f"--num-workers={args.num_workers}"
            )
        if args.only != ["stack_v2_permissive"]:
            raise SystemExit(
                "--num-workers > 1 is only supported for "
                "--only stack_v2_permissive (other sources have their own "
                "fan-out path or don't need one)"
            )

    log_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    logging.basicConfig(level=logging.INFO)
    for h in logging.getLogger().handlers:
        h.setFormatter(log_fmt)

    counts = _resolve_counts(args)
    paths = Paths(
        raw_dir=Path(args.raw_dir),
        shard_dir=Path(args.shard_dir),
        tokenizer=Path(args.tokenizer),
        shard_suffix=args.shard_suffix,
    )
    if not paths.tokenizer.is_file():
        raise SystemExit(
            f"tokenizer not found at {paths.tokenizer} — train one first "
            f"(python -m pretrain.cli.prepare_data train-tokenizer ...)"
        )
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.shard_dir.mkdir(parents=True, exist_ok=True)

    # Persist logs to the shared volume so they survive pod termination —
    # pod stdout is gone when the worker is reaped, and a 30-day Stack pull
    # produces logs that are the only post-hoc audit trail. Per-worker
    # filename mirrors the JSONL suffix so fanout siblings don't collide.
    log_suffix = f"_w{args.worker_id:02d}" if args.num_workers > 1 else ""
    log_path = paths.raw_dir / f"build_corpus{log_suffix}.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_fmt)
    logging.getLogger().addHandler(file_handler)
    LOG.info("logging to %s", log_path)

    selected = args.only or list(SOURCES)
    # Build src_to_count only for sources we'll actually run — under --only,
    # the counts dict may be sparse (one source's flag passed, others absent).
    src_to_count = {s: counts[_SRC_TO_FLAG_KEY[s]] for s in selected}
    LOG.info("plan: %s", src_to_count)

    # Fan-out: each worker writes to its own per-worker JSONL and shards
    # under <source><shard_suffix>/<source>_w<K>_*.bin with manifest.w<K>.yaml.
    # Single-worker (num_workers == 1) preserves the existing path layout.
    is_fanout = args.num_workers > 1
    shard_tag = f"w{args.worker_id:02d}" if is_fanout else ""
    raw_suffix = f"_w{args.worker_id:02d}" if is_fanout else ""

    for src in selected:
        # Whole-source gate (final merged manifest already present).
        if paths.manifest(src).is_file():
            LOG.info("[%s] manifest already at %s — skipping", src, paths.manifest(src))
            continue
        # Per-worker manifest gate (this worker's slice already sharded
        # in a prior pod attempt — Job backoffLimit retry should resume,
        # not redo).
        if is_fanout:
            worker_manifest = paths.shard_subdir(src) / f"manifest.{shard_tag}.yaml"
            if worker_manifest.is_file():
                LOG.info(
                    "[%s] worker manifest already at %s — skipping",
                    src, worker_manifest,
                )
                continue

        # Worker-tagged raw JSONL keeps sibling workers' pulls from
        # colliding on the same .part file.
        out_jsonl = paths.raw_dir / f"{src}{raw_suffix}.jsonl"
        if out_jsonl.is_file():
            LOG.info("[%s] raw JSONL already present at %s — using as-is", src, out_jsonl)
        else:
            # Write to .part and rename on success. We deliberately do NOT
            # unlink .part on entry or on exception — the puller is
            # resume-aware (dclm, fineweb) or handles its own truncate
            # (stack, proof). Leaving the .part on a transient HF / SWH
            # failure lets the Job's backoffLimit retry pick up where the
            # previous pod left off rather than losing hours of pull work.
            out_part = Path(str(out_jsonl) + ".part")
            puller = PULLERS[src]
            if src == "stack_v2_permissive":
                if is_fanout:
                    n_per_worker = src_to_count[src] // args.num_workers
                    # Remainder lands on the last worker so sum-of-targets
                    # equals docs_stack regardless of N.
                    if args.worker_id == args.num_workers - 1:
                        n_per_worker += src_to_count[src] % args.num_workers
                    puller(
                        n_per_worker, out_part,
                        concurrency=args.stack_concurrency,
                        worker_id=args.worker_id,
                        num_workers=args.num_workers,
                        log_tag=f"stack:{shard_tag}",
                    )
                else:
                    puller(src_to_count[src], out_part, concurrency=args.stack_concurrency)
            elif src == "dolma3_dolmino_mix":
                puller(src_to_count[src], out_part, concurrency=args.dolma3_concurrency)
            else:
                puller(src_to_count[src], out_part)
            out_part.rename(out_jsonl)
            # Sidecar is no longer needed once the source is final.
            _scanned_path(out_part).unlink(missing_ok=True)

        # Global raw-level shuffle (see SHUFFLED_SOURCES). Placed between
        # pull and shard so it applies exactly once per raw file (marker
        # sidecar), INCLUDING a pre-existing raw pulled before this step
        # existed — that file converges to shuffled on the next run without
        # a re-pull. An interrupted shuffle leaves the raw untouched.
        if src in SHUFFLED_SOURCES and not _shuffled_path(out_jsonl).is_file():
            LOG.info(
                "[%s] globally shuffling %s (seed=%d)",
                src, out_jsonl, args.shuffle_seed,
            )
            shuffle_jsonl(out_jsonl, seed=args.shuffle_seed)
            _shuffled_path(out_jsonl).write_text(
                f"seed={args.shuffle_seed} buckets={_SHUFFLE_BUCKETS}\n"
            )

        shard_source(
            src, paths,
            input_glob=out_jsonl.name,
            shard_tag=shard_tag,
            parallel_workers=args.dolma3_shard_workers if src == "dolma3_dolmino_mix" else 1,
        )

        if args.delete_raw:
            out_jsonl.unlink(missing_ok=True)
            LOG.info("[%s] removed raw JSONL", src)

    LOG.info("DONE — shards under %s", paths.shard_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
