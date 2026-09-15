"""Step -> document map for the training data explorer.

Walks the canonical global stream index-only (``.idx`` + ``manifest.yaml``, no
``.bin``) and records, for every optimizer step, the document fragments packed
into each of the step's windows. One row per fragment::

    step, global_window, microbatch, slot, frag_idx, dp_rank,
    source, shard_id, local_doc, tok_start, tok_end

``(source, shard_id, local_doc)`` is the corpus-wide document identity used by
the ``documents`` table (see :mod:`pretrain.cli.dump_documents`). The
``dp_rank`` column comes from :func:`pretrain.train.batch_schedule.owner_rank`,
the same function the training loop uses to pick a micro-batch's rank.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from pretrain.data.global_stream import ShardedWindowView
from pretrain.train.batch_schedule import StepPlan, owner_rank, window_position

SCHEMA = pa.schema(
    [
        ("step", pa.int32()),
        ("global_window", pa.int64()),
        ("microbatch", pa.int32()),
        ("slot", pa.int16()),
        ("frag_idx", pa.int16()),
        ("dp_rank", pa.int16()),
        ("source", pa.string()),
        ("shard_id", pa.int32()),
        ("local_doc", pa.int32()),
        ("tok_start", pa.int32()),
        ("tok_end", pa.int32()),
    ]
)

_COLUMNS = [f.name for f in SCHEMA]


def step_rows(
    plan: StepPlan,
    spans: list[tuple[int, str, int, int, int, int]],
    *,
    micro_batch_size: int,
    dp_world_size: int,
) -> dict[str, list]:
    """Turn one step's spans into columnar rows (see module docstring)."""
    cols: dict[str, list] = {c: [] for c in _COLUMNS}
    frag_idx = 0
    prev_window = -1
    for gw, source, shard_id, local, tok_start, tok_end in spans:
        if gw != prev_window:
            frag_idx = 0
            prev_window = gw
        microbatch, slot = window_position(gw, plan, micro_batch_size=micro_batch_size)
        cols["step"].append(plan.step)
        cols["global_window"].append(gw)
        cols["microbatch"].append(microbatch)
        cols["slot"].append(slot)
        cols["frag_idx"].append(frag_idx)
        cols["dp_rank"].append(owner_rank(microbatch, world_size=dp_world_size))
        cols["source"].append(source)
        cols["shard_id"].append(shard_id)
        cols["local_doc"].append(local)
        cols["tok_start"].append(tok_start)
        cols["tok_end"].append(tok_end)
        frag_idx += 1
    return cols


def iter_doc_map(
    view: ShardedWindowView,
    *,
    until_step: int,
    micro_batch_size: int,
    dp_world_size: int,
) -> Iterator[tuple[StepPlan, dict[str, list]]]:
    """Yield ``(plan, columns)`` per step from an ``index_only`` view."""
    for plan, spans in view.walk_step_spans(until_step=until_step):
        yield plan, step_rows(
            plan, spans, micro_batch_size=micro_batch_size, dp_world_size=dp_world_size
        )


def _extend(acc: dict[str, list], cols: dict[str, list]) -> None:
    for k, v in cols.items():
        acc[k].extend(v)


def _flush(acc: dict[str, list], out_dir: Path, first_step: int, last_step: int) -> Path:
    table = pa.table({k: pa.array(v, type=SCHEMA.field(k).type) for k, v in acc.items()})
    path = out_dir / f"step_docs-{first_step:06d}-{last_step:06d}.parquet"
    tmp = path.with_suffix(".parquet.part")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    for v in acc.values():
        v.clear()
    return path


def write_doc_map(
    view: ShardedWindowView,
    out_dir: str | Path,
    *,
    until_step: int,
    micro_batch_size: int,
    dp_world_size: int,
    chunk_steps: int = 1000,
    progress=None,
) -> dict:
    """Walk ``[view.start_step, until_step)`` and write Parquet chunks to ``out_dir``.

    Returns a summary with per-source fragment counts, totals, the written
    files and the stream ``position`` at the end of the walk (for comparison
    against a checkpoint's ``global_stream.json``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list] = {c: [] for c in _COLUMNS}
    files: list[str] = []
    per_source: dict[str, int] = {}
    total_rows = 0
    total_windows = 0
    first_step: int | None = None
    last_step = -1
    steps_walked = 0
    for plan, cols in iter_doc_map(
        view, until_step=until_step, micro_batch_size=micro_batch_size, dp_world_size=dp_world_size
    ):
        if first_step is None:
            first_step = plan.step
        last_step = plan.step
        steps_walked += 1
        n = len(cols["step"])
        total_rows += n
        total_windows += plan.microbatches * micro_batch_size
        for s in cols["source"]:
            per_source[s] = per_source.get(s, 0) + 1
        _extend(acc, cols)
        if (plan.step + 1) % chunk_steps == 0:
            files.append(str(_flush(acc, out_dir, first_step, last_step)))
            first_step = None
            if progress is not None:
                progress(plan.step + 1, total_rows)
    if first_step is not None and acc["step"]:
        files.append(str(_flush(acc, out_dir, first_step, last_step)))
    return {
        "until_step": until_step,
        "steps_walked": steps_walked,
        "windows": total_windows,
        "rows": total_rows,
        "rows_per_source": per_source,
        "files": files,
        "position": view.position(),
    }


def compare_position(position: dict, global_stream_json: dict) -> list[str]:
    """Return human-readable mismatches between a walk's ``position`` and a
    checkpoint's ``global_stream.json``; empty list means they agree."""
    problems: list[str] = []
    want = global_stream_json.get("consumed_documents_per_source", {})
    got = position["consumed_documents_per_source"]
    for name in sorted(set(want) | set(got)):
        if want.get(name) != got.get(name):
            problems.append(
                f"consumed_documents_per_source[{name}]: walk={got.get(name)} ckpt={want.get(name)}"
            )
    # A source missing on either side is a mismatch, same as for the document
    # counts: an older global_stream.json that names fewer sources must not pass
    # just because the walk happens to sit in epoch 0.
    want_ep = global_stream_json.get("epoch_per_source", {})
    got_ep = position["epoch_per_source"]
    for name in sorted(set(want_ep) | set(got_ep)):
        if want_ep.get(name) != got_ep.get(name):
            problems.append(f"epoch_per_source[{name}]: walk={got_ep.get(name)} ckpt={want_ep.get(name)}")
    if global_stream_json.get("windows_emitted") != position["windows_emitted"]:
        problems.append(
            f"windows_emitted: walk={position['windows_emitted']} "
            f"ckpt={global_stream_json.get('windows_emitted')}"
        )
    return problems


def write_manifest(out_dir: str | Path, summary: dict, extra: dict | None = None) -> Path:
    path = Path(out_dir) / "step_docs_manifest.json"
    blob = dict(summary)
    if extra:
        blob.update(extra)
    path.write_text(json.dumps(blob, indent=2))
    return path
