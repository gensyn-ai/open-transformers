"""CLI: detokenize every shard of a prepared dataset into Parquet documents.

One output file per shard, one row per document::

    source, shard_id, local_doc, n_tokens, text

``(source, shard_id, local_doc)`` matches the identity the step -> document map
uses (:mod:`pretrain.cli.dump_doc_map`), so the two tables join directly.
Shards are processed in parallel; existing outputs are skipped so the job
resumes after interruption.

Usage:
    python -m pretrain.cli.dump_documents \\
        --data-root data/shards --tokenizer data/tokenizer.json \\
        --out-dir /path/to/documents [--sources dclm_baseline fineweb_edu] \\
        [--workers 24] [--batch-docs 2048]
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

log = logging.getLogger("pretrain.dump_documents")

_TOK = None


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="pretrain.cli.dump_documents")
    p.add_argument("--data-root", required=True, help="directory holding <source>/manifest.yaml + shards")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json used to prepare the shards")
    p.add_argument("--out-dir", required=True, help="output root; files land at <out>/<source>/<shard_id>_<prefix>.parquet")
    p.add_argument("--sources", nargs="*", default=None, help="subset of source names (default: all)")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--batch-docs", type=int, default=2048, help="documents decoded per batch / row group")
    p.add_argument("--limit-shards", type=int, default=None, help="debug: only the first N shards per source")
    return p.parse_args(argv)


def _init_worker(tokenizer_path: str) -> None:
    global _TOK
    from pretrain.data.tokenizer import Tokenizer

    _TOK = Tokenizer(tokenizer_path)


def dump_shard(task: tuple[str, int, str, str, int]) -> tuple[str, int, int, float]:
    """Decode one shard to Parquet. Returns (source, shard_id, n_docs, seconds)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pretrain.data.indexed_dataset import IndexedDatasetReader

    source, shard_id, prefix, out_path, batch_docs = task
    out = Path(out_path)
    if out.exists():
        return source, shard_id, -1, 0.0
    t0 = time.time()
    reader = IndexedDatasetReader(prefix)
    schema = pa.schema(
        [
            ("source", pa.string()),
            ("shard_id", pa.int32()),
            ("local_doc", pa.int32()),
            ("n_tokens", pa.int32()),
            ("text", pa.large_string()),
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    n = reader.num_documents
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for start in range(0, n, batch_docs):
            stop = min(n, start + batch_docs)
            ids = [reader.document(i).tolist() for i in range(start, stop)]
            # A scraped document can contain the literal "<|endoftext|>", which
            # the tokenizer stores as the special id mid-document. Keep it so
            # the text stays faithful to the stored tokens and to n_tokens.
            texts = _TOK.decode_batch(ids, skip_special_tokens=False)
            w.write_table(
                pa.table(
                    {
                        "source": pa.array([source] * (stop - start), pa.string()),
                        "shard_id": pa.array([shard_id] * (stop - start), pa.int32()),
                        "local_doc": pa.array(range(start, stop), pa.int32()),
                        "n_tokens": pa.array([len(x) for x in ids], pa.int32()),
                        "text": pa.array(texts, pa.large_string()),
                    },
                    schema=schema,
                )
            )
    tmp.replace(out)
    return source, shard_id, n, time.time() - t0


def build_tasks(data_root: Path, out_dir: Path, sources: list[str] | None, batch_docs: int, limit: int | None):
    from pretrain.data.manifest import SourceManifest

    tasks = []
    seen: dict[str, str] = {}
    for mpath in sorted(data_root.glob("*/manifest.yaml")):
        manifest = SourceManifest.load(mpath)
        if sources and manifest.name not in sources:
            continue
        for sid, shard in enumerate(manifest.shards):
            if limit is not None and sid >= limit:
                break
            pfx = Path(shard.prefix)
            prefix = pfx if pfx.is_absolute() else mpath.parent / pfx
            # Keyed on shard_id, not the prefix basename: shard prefixes may
            # live in different parent dirs and share a basename.
            out = out_dir / manifest.name / f"{sid:05d}_{pfx.name}.parquet"
            # Two manifests with the same source name would still collide, and
            # the skip-existing resume logic would then silently drop one shard.
            if str(out) in seen:
                raise ValueError(
                    f"duplicate output {out} for shards {seen[str(out)]} and {prefix}"
                )
            seen[str(out)] = str(prefix)
            tasks.append((manifest.name, sid, str(prefix), str(out), batch_docs))
    return tasks


def run(args) -> int:
    data_root, out_dir = Path(args.data_root), Path(args.out_dir)
    tasks = build_tasks(data_root, out_dir, args.sources, args.batch_docs, args.limit_shards)
    if not tasks:
        log.error("no shards found under %s", data_root)
        return 1
    log.info("%d shards, %d workers, out=%s", len(tasks), args.workers, out_dir)
    t0 = time.time()
    done = skipped = 0
    docs = 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for source, sid, n, dt in pool.imap_unordered(dump_shard, tasks):
            if n < 0:
                skipped += 1
                continue
            done += 1
            docs += n
            if done % 10 == 0 or dt > 300:
                log.info(
                    "%d/%d shards (%d skipped), %d docs, last %s/%d %.0fs, elapsed %.0fs",
                    done + skipped, len(tasks), skipped, docs, source, sid, dt, time.time() - t0,
                )
    log.info("finished: %d written, %d skipped, %d docs, %.0fs", done, skipped, docs, time.time() - t0)
    return 0


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    sys.exit(run(_parse_args(argv)))


if __name__ == "__main__":
    main()
