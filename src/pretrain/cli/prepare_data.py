"""Data prep CLI.

Subcommands:
    train-tokenizer  : train a fresh BPE tokenizer
    shard            : tokenise a source's parquet/jsonl files into
                       indexed binary shards
    inspect          : decode N documents from a shard
    token-stats      : per-source bytes/token diagnostic
    replay-step      : emit the documents the loader would produce at
                       step N with seed S (debugging tool)

The full-fat dataset-specific runners live under ``scripts/`` and call
into this module for the heavy lifting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pretrain.data.indexed_dataset import IndexedDatasetReader, IndexedDatasetWriter
from pretrain.data.manifest import ShardInfo, SourceManifest, blake2b_file
from pretrain.data.prepare import (
    inspect_shard,
    iter_text_from_jsonl,
    iter_text_from_jsonl_segment,
    iter_text_from_parquet,
    write_shards,
)
from pretrain.data.tokenizer import Tokenizer, bytes_per_token, train_tokenizer

LOG = logging.getLogger("pretrain.prepare")


def _cmd_train_tokenizer(args: argparse.Namespace) -> None:
    paths = list(Path(args.input_dir).glob(args.glob))
    LOG.info("training tokenizer on %d files (glob=%s)", len(paths), args.glob)

    if args.format == "parquet":
        text_iter = iter_text_from_parquet(paths, text_column=args.text_field)
    elif args.format == "jsonl":
        text_iter = iter_text_from_jsonl(paths, text_field=args.text_field)
    else:
        raise SystemExit(f"unknown --format {args.format}")

    out = train_tokenizer(
        text_iterator=text_iter,
        output_path=args.output,
        vocab_size=args.vocab_size,
    )
    LOG.info("wrote %s", out)


def _cmd_shard(args: argparse.Namespace) -> None:
    if args.num_workers > 1 and args.shard_tag:
        raise SystemExit(
            "pass --num-workers (input segmenting + auto-tagging) OR "
            "--shard-tag (output-only tagging), not both"
        )
    tok = Tokenizer(args.tokenizer)
    paths = list(Path(args.input_dir).glob(args.glob))
    if not paths:
        raise SystemExit(f"no input files matched {args.input_dir}/{args.glob}")
    if args.num_workers > 1:
        if args.format != "jsonl":
            raise SystemExit(
                "--num-workers > 1 only supported for --format jsonl "
                "(parquet sharding is already per-file parallel via --glob)"
            )
        if not (0 <= args.worker_id < args.num_workers):
            raise SystemExit(
                f"--worker-id={args.worker_id} out of range for "
                f"--num-workers={args.num_workers}"
            )
        text_iter = iter_text_from_jsonl_segment(
            paths,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            text_field=args.text_field,
        )
        shard_prefix_tag = f"_w{args.worker_id:02d}"
        manifest_filename = f"manifest.w{args.worker_id:02d}.yaml"
        LOG.info(
            "sharding %d files (worker %d/%d) into %s — shards=%s%s_*, manifest=%s",
            len(paths), args.worker_id, args.num_workers, args.output_dir,
            args.source_name, shard_prefix_tag, manifest_filename,
        )
    else:
        if args.format == "parquet":
            text_iter = iter_text_from_parquet(paths, text_column=args.text_field)
        elif args.format == "jsonl":
            text_iter = iter_text_from_jsonl(paths, text_field=args.text_field)
        else:
            raise SystemExit(f"unknown --format {args.format}")
        if args.shard_tag:
            shard_prefix_tag = f"_{args.shard_tag}"
            manifest_filename = f"manifest.{args.shard_tag}.yaml"
            LOG.info(
                "sharding %d files into %s — shards=%s%s_*, manifest=%s",
                len(paths), args.output_dir,
                args.source_name, shard_prefix_tag, manifest_filename,
            )
        else:
            shard_prefix_tag = ""
            manifest_filename = "manifest.yaml"
            LOG.info("sharding %d files into %s", len(paths), args.output_dir)

    # Audit hash of each raw JSONL fed to the sharder. Recorded into the
    # manifest so a future run can verify "we sharded the same bytes."
    # For fanout (workers reading byte slices of the same file), every
    # worker records the same {filename: hash} entry; the merge step
    # deduplicates identical-key/identical-value pairs and fails loudly
    # on key/value conflicts. Skipped for parquet inputs.
    if args.format == "jsonl":
        raw_jsonl_blake2b = {p.name: blake2b_file(p) for p in paths}
        LOG.info("hashed %d raw JSONL input(s) for manifest audit", len(paths))
    else:
        raw_jsonl_blake2b = None

    manifest = write_shards(
        text_iter=text_iter,
        tokenizer=tok,
        output_dir=args.output_dir,
        source_name=args.source_name,
        target_shard_bytes=args.shard_bytes,
        dtype=np.uint32,
        shard_prefix_tag=shard_prefix_tag,
        manifest_filename=manifest_filename,
        raw_jsonl_blake2b=raw_jsonl_blake2b,
    )
    LOG.info(
        "manifest: %d shards / %d docs / %d tokens",
        len(manifest.shards), manifest.total_documents, manifest.total_tokens,
    )


def _cmd_merge_manifests(args: argparse.Namespace) -> None:
    """Merge per-worker shard manifests into the canonical manifest.yaml.

    The fan-out sharder (``shard --worker-id K --num-workers N``) writes
    ``manifest.w<K>.yaml`` per worker. Once all workers have finished,
    this command concatenates their shard lists (in worker-id order, so
    the on-disk byte ordering is reproducible) into a single
    ``<output-dir>/manifest.yaml`` that the loader can consume.
    """
    out_dir = Path(args.output_dir)
    parts = sorted(out_dir.glob("manifest.w*.yaml"))
    if not parts:
        raise SystemExit(
            f"no manifest.w*.yaml under {out_dir} — did the fan-out workers run?"
        )
    if args.expected_workers and len(parts) != args.expected_workers:
        raise SystemExit(
            f"expected {args.expected_workers} per-worker manifests under "
            f"{out_dir}, found {len(parts)}: {[p.name for p in parts]}"
        )

    manifests = [SourceManifest.load(p) for p in parts]
    first = manifests[0]
    for p, m in zip(parts, manifests):
        if m.name != first.name:
            raise SystemExit(f"name mismatch in {p}: {m.name} vs {first.name}")
        if m.tokenizer_hash != first.tokenizer_hash:
            raise SystemExit(
                f"tokenizer_hash mismatch in {p} — workers used different tokenizers?"
            )
        if m.dtype != first.dtype:
            raise SystemExit(f"dtype mismatch in {p}: {m.dtype} vs {first.dtype}")

    merged_shards: list[ShardInfo] = []
    for m in manifests:
        merged_shards.extend(m.shards)

    # Union per-worker raw_jsonl_blake2b dicts. Two workers may legitimately
    # record the same filename (DCLM shard fanout, where N workers read
    # byte slices of one JSONL); identical values collapse cleanly. Any
    # disagreement on the same filename is a real bug — fail loudly.
    merged_raw: dict[str, str] = {}
    for p, m in zip(parts, manifests):
        for fname, h in m.raw_jsonl_blake2b.items():
            existing = merged_raw.get(fname)
            if existing is not None and existing != h:
                raise SystemExit(
                    f"raw_jsonl_blake2b conflict for {fname!r} in {p}: "
                    f"{h} vs already-seen {existing}"
                )
            merged_raw[fname] = h

    merged = SourceManifest(
        name=first.name,
        tokenizer_hash=first.tokenizer_hash,
        dtype=first.dtype,
        shards=merged_shards,
        filter_version=first.filter_version,
        raw_jsonl_blake2b=merged_raw,
    )
    out_path = out_dir / "manifest.yaml"
    merged.save(out_path)
    LOG.info(
        "merged %d worker manifests → %s: %d shards / %d docs / %d tokens",
        len(parts), out_path,
        len(merged.shards), merged.total_documents, merged.total_tokens,
    )
    if args.remove_worker_manifests:
        for p in parts:
            p.unlink()
        LOG.info("removed %d per-worker manifest files", len(parts))


def _cmd_inspect(args: argparse.Namespace) -> None:
    tok = Tokenizer(args.tokenizer)
    inspect_shard(args.shard, tok, n=args.n)


def _cmd_token_stats(args: argparse.Namespace) -> None:
    tok = Tokenizer(args.tokenizer)
    sample = Path(args.sample).read_text(encoding="utf-8")
    bpt = bytes_per_token(tok, sample)
    print(json.dumps({"bytes_per_token": bpt}, indent=2))


def _cmd_synthetic(args: argparse.Namespace) -> None:
    """Generate a synthetic corpus for the M0 smoke test.

    No real data, no tokenizer required. Two "sources" matching
    `configs/data/synthetic.yaml` are written: synth_a and synth_b.
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for name, n_docs in [("synth_a", args.docs_a), ("synth_b", args.docs_b)]:
        src_dir = out_dir / name
        src_dir.mkdir(parents=True, exist_ok=True)
        prefix = src_dir / f"{name}_00000"
        with IndexedDatasetWriter(prefix, dtype=np.uint32) as w:
            for _ in range(n_docs):
                doc_len = int(rng.integers(args.min_doc_len, args.max_doc_len + 1))
                w.add_document(
                    rng.integers(1, args.vocab_size, size=doc_len, dtype=np.uint32)
                )
        manifest = SourceManifest(
            name=name,
            tokenizer_hash="synthetic-no-tokenizer",
            dtype="uint32",
            shards=[
                ShardInfo(
                    prefix=f"{name}_00000",
                    num_documents=n_docs,
                    token_count=-1,    # filled below from the reader
                )
            ],
        )
        # Read back the actual token count rather than tracking it in the loop.
        reader = IndexedDatasetReader(prefix)
        manifest.shards[0].token_count = reader.token_count
        manifest.save(src_dir / "manifest.yaml")
        LOG.info(
            "synthetic %s: %d docs / %d tokens at %s",
            name, n_docs, reader.token_count, src_dir,
        )


def _cmd_replay_step(args: argparse.Namespace) -> None:
    from pretrain.config import load_config
    from pretrain.data.loader import build_loader

    cfg = load_config(args.config_name, overrides=args.override or [])
    loader, _ = build_loader(
        cfg.data,
        micro_batch_size=cfg.train.micro_batch_size,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.seed,
        eos_id=cfg.data.document_separator_id,
    )
    it = iter(loader)
    for _ in range(args.step):
        next(it)
    batch = next(it)
    print(json.dumps({
        "input_ids_shape": list(batch["input_ids"].shape),
        "first_tokens": batch["input_ids"][0, :32].tolist(),
    }, indent=2))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="pretrain.cli.prepare_data")
    sp = parser.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("train-tokenizer", help="train a fresh BPE tokenizer")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--glob", default="*.parquet")
    p.add_argument("--format", choices=["parquet", "jsonl"], default="parquet")
    p.add_argument("--text-field", default="text")
    p.add_argument("--output", required=True)
    p.add_argument("--vocab-size", type=int, default=128_256)
    p.set_defaults(fn=_cmd_train_tokenizer)

    p = sp.add_parser("shard", help="parquet/jsonl → indexed binary shards")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--glob", default="*.parquet")
    p.add_argument("--format", choices=["parquet", "jsonl"], default="parquet")
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source-name", required=True)
    p.add_argument("--shard-bytes", type=int, default=1 << 30)
    # Fan-out sharder: each worker reads a byte-segment of the JSONL and
    # writes shards named <source>_w<K>_<idx>.bin plus manifest.w<K>.yaml.
    # Defaults reproduce the single-worker layout. Run `merge-manifests`
    # after all workers finish to write the canonical manifest.yaml.
    p.add_argument("--worker-id", type=int, default=0,
                   help="zero-based worker index for fan-out sharding")
    p.add_argument("--num-workers", type=int, default=1,
                   help="total number of fan-out workers (1 = single worker)")
    p.add_argument("--shard-tag", default="",
                   help="output-only naming tag (e.g. 'w03'): emits "
                        "<source>_<tag>_*.bin and manifest.<tag>.yaml. Mutually "
                        "exclusive with --num-workers; use this when each worker "
                        "has its own input JSONL (e.g. Stack fan-out) rather than "
                        "byte-slicing a shared one (e.g. DCLM fan-out).")
    p.set_defaults(fn=_cmd_shard)

    p = sp.add_parser(
        "merge-manifests",
        help="combine per-worker manifest.w*.yaml into manifest.yaml",
    )
    p.add_argument("--output-dir", required=True,
                   help="directory containing manifest.w*.yaml from fan-out workers")
    p.add_argument("--expected-workers", type=int, default=0,
                   help="if non-zero, fail unless exactly this many per-worker "
                        "manifests are found")
    p.add_argument("--remove-worker-manifests", action="store_true",
                   help="delete manifest.w*.yaml after a successful merge")
    p.set_defaults(fn=_cmd_merge_manifests)

    p = sp.add_parser("inspect", help="decode N documents from a shard")
    p.add_argument("--shard", required=True, help="shard prefix (no .bin/.idx)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--n", type=int, default=3)
    p.set_defaults(fn=_cmd_inspect)

    p = sp.add_parser("token-stats", help="bytes/token sanity")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--sample", required=True, help="path to a UTF-8 text file")
    p.set_defaults(fn=_cmd_token_stats)

    p = sp.add_parser(
        "synthetic",
        help="generate a synthetic corpus for the M0 smoke test (no tokenizer needed)",
    )
    p.add_argument("--output-dir", default="data/shards/synthetic")
    p.add_argument("--docs-a", type=int, default=2000)
    p.add_argument("--docs-b", type=int, default=1000)
    p.add_argument("--min-doc-len", type=int, default=64)
    p.add_argument("--max-doc-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=128_256)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_cmd_synthetic)

    p = sp.add_parser("replay-step", help="emit the loader's batch for step N")
    p.add_argument("--config-name", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--override", nargs="*", default=[])
    p.set_defaults(fn=_cmd_replay_step)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
