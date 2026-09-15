"""Spot-check a step_docs dump against the materialised stream (gate 3).

For each requested step, materialise the step's windows with a full (``.bin``
backed) ``ShardedWindowView`` and check that the dump's fragments, read from
the shards at ``[tok_start, tok_end)`` and interleaved with the packer's EOS
separators, reproduce every window token for token. Exit 1 on any mismatch.

    PYTHONPATH=src python scripts/explorer/spot_check_doc_map.py \
        --checkpoint runs/<id>/checkpoints/step_000080957 \
        --step-docs /path/to/step_docs --steps 0 1 19 [--data-root data/shards]
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pretrain.data.fetch_interval import load_run_config, read_checkpoint_descriptor, rebase_sources
from pretrain.data.global_stream import _EOS, ShardedWindowView
from pretrain.data.indexed_dataset import IndexedDatasetReader
from pretrain.data.loader import _resolve_sources
from pretrain.train.batch_schedule import iter_step_plans


def load_rows(step_docs: str | Path, steps: set[int]) -> dict[int, list[dict]]:
    """Doc-map rows for ``steps``, grouped by global window and ordered by frag_idx."""
    rows: dict[int, list[dict]] = defaultdict(list)
    for f in sorted(glob.glob(str(Path(step_docs) / "step_docs-*.parquet"))):
        t = pq.read_table(f, filters=[("step", "in", list(steps))])
        for r in t.to_pylist():
            rows[r["global_window"]].append(r)
    for frs in rows.values():
        frs.sort(key=lambda r: r["frag_idx"])
    return rows


class _Readers:
    """Open a shard's reader on first use only. Every reader copies its two
    int64 offset tables at construction, which at corpus scale is tens of GB
    if done for all shards up front; a spot check touches a handful."""

    def __init__(self, manifests, dirs):
        self._prefix = {}
        for m, d in zip(manifests, dirs):
            for sid, sh in enumerate(m.shards):
                p = Path(sh.prefix)
                self._prefix[(m.name, sid)] = p if p.is_absolute() else Path(d) / p
        self._open: dict[tuple[str, int], IndexedDatasetReader] = {}

    def __call__(self, source: str, shard_id: int) -> IndexedDatasetReader:
        key = (source, shard_id)
        if key not in self._open:
            self._open[key] = IndexedDatasetReader(self._prefix[key])
        return self._open[key]


def check_window(win: np.ndarray, frags: list[dict], readers, *, eos: int, leading_eos: bool) -> str | None:
    """Rebuild ``win`` from its fragments; return a message on the first mismatch.

    Packing rule (``ShardedWindowView._refill`` / ``_take``): the stream is
    ``doc, EOS, doc, EOS, ...`` cut into fixed ``seq_len + 1`` windows. So a
    fragment that reaches the end of its document is followed by exactly one
    EOS, which lands in this window if there is room and otherwise opens the
    next one (``leading_eos``). A fragment that stops short of its document's
    end was cut by the window boundary and must be the window's last token.
    A document's own tokens may legitimately equal ``eos``, so the walk never
    classifies tokens by value; it only checks them at the offsets the rule
    dictates.
    """
    n = win.size
    cur = 0
    if leading_eos:
        if win[0] != eos:
            return f"offset 0: expected spilled EOS {eos}, got {int(win[0])}"
        cur = 1
    for i, r in enumerate(frags):
        if r["frag_idx"] != i:
            return f"frag {i}: frag_idx {r['frag_idx']} out of sequence"
        doc = readers(r["source"], r["shard_id"]).document(r["local_doc"])
        s, e = r["tok_start"], r["tok_end"]
        if not 0 <= s <= e <= doc.size:
            return f"frag {i}: [{s}, {e}) outside document of {doc.size} tokens"
        if cur + (e - s) > n:
            return f"frag {i}: {e - s} tokens at offset {cur} overrun the window"
        if not np.array_equal(win[cur : cur + (e - s)], doc[s:e]):
            return f"frag {i}: tokens at offset {cur} differ from {r['source']}/{r['shard_id']}/{r['local_doc']}[{s}:{e}]"
        cur += e - s
        if cur == n:
            if i != len(frags) - 1:
                return f"frag {i}: window full but {len(frags) - 1 - i} fragments remain"
            break
        if e != doc.size:
            return f"frag {i}: document cut short at offset {cur} but window continues"
        if win[cur] != eos:
            return f"offset {cur}: expected EOS {eos} after frag {i}, got {int(win[cur])}"
        cur += 1
    if cur != n:
        return f"fragments and separators cover {cur} of {n} tokens"
    return None


def check_steps(
    manifests, dirs, weights, train, *, seed: int, eos: int, rows: dict[int, list[dict]], steps: set[int]
) -> tuple[int, int]:
    """Walk the stream from step 0 and check every window of ``steps``.

    Returns ``(checked, bad)``. Only the requested steps' windows are
    materialised; every other window is a length-only cursor advance.
    """
    view = ShardedWindowView(manifests, dirs, weights, train, seed=seed, rank=0, world_size=1, eos_id=eos)
    readers = _Readers(manifests, dirs)
    mb = train.micro_batch_size
    max_step = max(steps)
    checked = bad = 0
    for plan in iter_step_plans(0, 0, train):
        if plan.step > max_step:
            break
        want = plan.step in steps
        for m in range(plan.microbatches):
            for s in range(mb):
                # The buffer front is an EOS segment exactly when the previous
                # window ended on a document's last token and its separator
                # spilled over.
                leading_eos = bool(view._segs) and view._segs[0][0] == _EOS
                win = view._take_window(materialize=want)
                if not want:
                    continue
                gw = plan.base_window + m * mb + s
                frs = rows.get(gw, [])
                problem = check_window(np.asarray(win, dtype=np.int64), frs, readers, eos=eos, leading_eos=leading_eos)
                if problem is None:
                    for r in frs:
                        if r["microbatch"] != m or r["slot"] != s:
                            problem = f"frag {r['frag_idx']}: microbatch/slot {r['microbatch']}/{r['slot']} != {m}/{s}"
                            break
                checked += 1
                if problem is not None:
                    bad += 1
                    print(f"MISMATCH step={plan.step} window={gw} frags={len(frs)}: {problem}")
    return checked, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--step-docs", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--config-name", default=None)
    a = ap.parse_args()

    meta, _ = read_checkpoint_descriptor(a.checkpoint)
    cfg = load_run_config(meta, a.config_name)
    data_cfg = rebase_sources(cfg.data, a.data_root) if a.data_root else cfg.data
    manifests, dirs, weights = _resolve_sources(data_cfg)

    steps = set(a.steps)
    rows = load_rows(a.step_docs, steps)
    if not rows:
        print("no rows for the requested steps in", a.step_docs)
        return 1
    checked, bad = check_steps(
        manifests, dirs, weights, cfg.train,
        seed=int(meta["seed"]), eos=data_cfg.document_separator_id, rows=rows, steps=steps,
    )
    print(f"checked {checked} windows across steps {sorted(steps)}: {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
