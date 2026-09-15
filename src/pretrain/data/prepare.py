"""Prepare a source's text into indexed binary shards.

Workflow per source:
  text iterator → tokenizer.encode → indexed binary shards (1 GB each) →
  ``manifest.yaml`` listing shards + token counts.

The CLI front-end is in ``pretrain.cli.prepare_data``. This module exposes
the building blocks that CLI uses, so they're individually testable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from pretrain.data.indexed_dataset import IndexedDatasetWriter
from pretrain.data.manifest import ShardInfo, SourceManifest, blake2b_file
from pretrain.data.tokenizer import Tokenizer

LOG = logging.getLogger(__name__)


def iter_text_from_parquet(
    parquet_paths: Iterable[str | Path],
    text_column: str = "text",
) -> Iterator[str]:
    """Iterate documents from a list of parquet files. Streamed; no full
    materialisation.
    """
    import pyarrow.parquet as pq

    for path in parquet_paths:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(columns=[text_column], batch_size=4096):
            for v in batch.column(text_column).to_pylist():
                if v is None:
                    continue
                yield v


def iter_text_from_jsonl(
    jsonl_paths: Iterable[str | Path],
    text_field: str = "text",
) -> Iterator[str]:
    """Iterate documents from JSON-lines files (one JSON object per line).

    Tolerant of malformed lines: when the stop-early-and-shard workflow
    renames a still-being-written .part to .jsonl, the last line may be
    a truncated half-write like ``{"text": "abc...`` with no closing
    brace. We skip JSON-decode failures rather than crashing the shard,
    and log a summary per file so the count is visible but not noisy.
    """
    import json

    for path in jsonl_paths:
        bad = 0
        with open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                v = obj.get(text_field)
                if v:
                    yield v
        if bad:
            LOG.warning("[%s] skipped %d malformed JSON lines", path, bad)


def iter_text_from_jsonl_segment(
    jsonl_paths: Iterable[str | Path],
    worker_id: int,
    num_workers: int,
    text_field: str = "text",
) -> Iterator[str]:
    """Iterate the worker_id'th byte-segment of each JSONL file.

    Splits each file into ``num_workers`` contiguous byte ranges; worker
    ``k`` reads the ``[size*k/N, size*(k+1)/N)`` range. The partition
    boundary may land mid-line, so we use the standard split-at-newline
    convention: the line that *contains* the start byte belongs to the
    previous worker. Concretely:

      - if the byte at ``start-1`` is ``\\n``, ``start`` is the first byte
        of a fresh line — we keep it.
      - otherwise ``start`` is inside a line that the previous worker has
        already emitted — we discard up to the next newline.

    A line that *starts* inside our range but extends past ``end`` is
    fully emitted by us; the next worker's pre-newline check will skip
    it. So every line is emitted exactly once across the N workers.
    """
    if num_workers <= 0 or not (0 <= worker_id < num_workers):
        raise ValueError(
            f"invalid worker_id={worker_id} for num_workers={num_workers}"
        )
    import json

    for path in jsonl_paths:
        size = Path(path).stat().st_size
        start = (size * worker_id) // num_workers
        end = (size * (worker_id + 1)) // num_workers
        if start >= end:
            continue
        bad = 0
        with open(path, "rb") as f:
            if start > 0:
                f.seek(start - 1)
                if f.read(1) != b"\n":
                    f.readline()
            while f.tell() < end:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                v = obj.get(text_field)
                if v:
                    yield v
        if bad:
            LOG.warning("[%s w=%d/%d] skipped %d malformed JSON lines",
                        path, worker_id, num_workers, bad)


def write_shards(
    text_iter: Iterable[str],
    tokenizer: Tokenizer,
    output_dir: str | Path,
    source_name: str,
    target_shard_bytes: int = 1 << 30,    # 1 GiB
    dtype: np.dtype | str = np.uint32,
    progress_every: int = 100_000,
    shard_prefix_tag: str = "",
    manifest_filename: str = "manifest.yaml",
    raw_jsonl_blake2b: dict[str, str] | None = None,
) -> SourceManifest:
    """Tokenize ``text_iter`` and write 1 GB-target shards.

    Returns a freshly-built manifest. The on-disk layout is

        <output_dir>/<source><tag>_00000.bin
        <output_dir>/<source><tag>_00000.idx
        <output_dir>/<source><tag>_00001.bin
        ...
        <output_dir>/<manifest_filename>

    ``shard_prefix_tag`` and ``manifest_filename`` exist for the fan-out
    sharder (see ``--worker-id`` / ``--num-workers`` on the shard CLI):
    each parallel worker writes its own non-colliding shard files and
    its own per-worker manifest, which a later merge step concatenates
    into the canonical ``manifest.yaml``. Default empty tag preserves
    the single-worker on-disk layout.

    ``raw_jsonl_blake2b`` is an audit field threaded into the manifest
    unchanged — keyed by raw JSONL filename, hashed by the caller
    before the iterator is consumed. blake2b-32 of each shard's .bin
    and .idx is recorded automatically into the per-shard ``ShardInfo``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize

    shards: list[ShardInfo] = []
    shard_idx = 0
    bytes_written = 0
    docs_in_shard = 0
    tokens_in_shard = 0
    docs_total = 0
    tokens_total = 0

    def open_shard(i: int) -> tuple[Path, IndexedDatasetWriter]:
        prefix = output_dir / f"{source_name}{shard_prefix_tag}_{i:05d}"
        return prefix, IndexedDatasetWriter(prefix, dtype=dtype)

    prefix, writer = open_shard(shard_idx)

    try:
        for ids in tokenizer.encode_iter(text_iter):
            if not ids:
                continue
            writer.add_document(np.asarray(ids, dtype=dtype))
            n = len(ids)
            bytes_written += n * itemsize
            docs_in_shard += 1
            tokens_in_shard += n
            docs_total += 1
            tokens_total += n

            if docs_total % progress_every == 0:
                LOG.info(
                    "[%s] docs=%d tokens=%d shards=%d",
                    source_name, docs_total, tokens_total, shard_idx + 1,
                )

            if bytes_written >= target_shard_bytes:
                writer.close()
                shards.append(
                    ShardInfo(
                        prefix=str(prefix.relative_to(output_dir)),
                        num_documents=docs_in_shard,
                        token_count=tokens_in_shard,
                        bin_blake2b=blake2b_file(prefix.with_suffix(".bin")),
                        idx_blake2b=blake2b_file(prefix.with_suffix(".idx")),
                    )
                )
                shard_idx += 1
                bytes_written = docs_in_shard = tokens_in_shard = 0
                prefix, writer = open_shard(shard_idx)

        # Flush trailing shard if non-empty.
        if docs_in_shard > 0:
            writer.close()
            shards.append(
                ShardInfo(
                    prefix=str(prefix.relative_to(output_dir)),
                    num_documents=docs_in_shard,
                    token_count=tokens_in_shard,
                    bin_blake2b=blake2b_file(prefix.with_suffix(".bin")),
                    idx_blake2b=blake2b_file(prefix.with_suffix(".idx")),
                )
            )
        else:
            writer.close()
            # Remove the empty trailing files.
            prefix.with_suffix(".bin").unlink(missing_ok=True)
            prefix.with_suffix(".idx").unlink(missing_ok=True)
    finally:
        try:
            writer.close()
        except Exception:
            pass

    manifest = SourceManifest(
        name=source_name,
        tokenizer_hash=tokenizer.hash,
        dtype=dtype.name,
        shards=shards,
        raw_jsonl_blake2b=dict(raw_jsonl_blake2b) if raw_jsonl_blake2b else {},
    )
    manifest.save(output_dir / manifest_filename)
    LOG.info(
        "[%s] DONE docs=%d tokens=%d shards=%d",
        source_name, docs_total, tokens_total, len(shards),
    )
    return manifest


def inspect_shard(
    shard_prefix: str | Path,
    tokenizer: Tokenizer,
    n: int = 3,
    write: Callable[[str], None] = print,
) -> None:
    """Decode and dump up to ``n`` documents from a shard."""
    from pretrain.data.indexed_dataset import IndexedDatasetReader

    reader = IndexedDatasetReader(shard_prefix)
    write(f"== shard {shard_prefix} (docs={reader.num_documents}, tokens={reader.token_count})")
    for i in range(min(n, reader.num_documents)):
        ids = [int(t) for t in reader.document(i)]
        text = tokenizer.decode(ids)
        write(f"-- doc {i} (len={len(ids)}):\n{text[:1024]}")
