"""Plot DCLM-CORE eval results from one or more run directories.

After copying ``runs/<RUN_ID>/evals/`` from the training host:

    python scripts/plot_evals.py runs/<RUN_ID>
    python scripts/plot_evals.py runs/<a> runs/<b> --labels torch repop
    python scripts/plot_evals.py runs/<a> -o evals.png
    python scripts/plot_evals.py runs/<a> runs/<a-resume1> runs/<a-resume2> --stitch

With ``--stitch`` the run directories are treated as consecutive
segments of one resumed run (given in chronological order) and merged
into a single series; where segments overlap in step, the later
segment wins.

Reads each ``step_<N>/step_*.json`` produced by ``pretrain.cli.eval`` and
draws, vs training step:

  - the overall macro-average,
  - one panel per category (a category with a single task is relabeled
    to that task, e.g. ``world_knowledge`` -> ``mmlu``),
  - one panel per individual task (its ``primary`` metric), so the
    per-task breakdown inside multi-task categories is visible too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

CAT_PREFIX = "cat:"
TASK_PREFIX = "task:"


def load_run(run_dir: Path) -> tuple[list[dict], dict[str, set[str]]]:
    """Return (points, category->tasks mapping) for a single run.

    Each point holds ``step``, ``macro`` (overall macro-average),
    ``cat:<name>`` per category macro, and ``task:<name>`` per-task
    primary score. The mapping is populated from ``category_tasks`` when
    present (eval outputs since the relabel change); older outputs leave
    it empty and the caller infers single-task categories by value.
    """
    evals_dir = run_dir / "evals"
    if not evals_dir.exists():
        raise SystemExit(f"no evals dir at {evals_dir}")

    points: list[dict] = []
    cat_tasks: dict[str, set[str]] = {}
    for d in sorted(evals_dir.glob("step_*")):
        if not d.is_dir():
            continue
        json_files = list(d.glob("step_*.json"))
        if not json_files:
            continue
        result = json.loads(json_files[0].read_text())
        step = int(d.name.removeprefix("step_"))

        pt: dict = {"step": step, "macro": result.get("macro_average")}
        for cat, val in (result.get("categories") or {}).items():
            pt[CAT_PREFIX + cat] = val
        for name, info in (result.get("tasks") or {}).items():
            primary = info.get("primary") if isinstance(info, dict) else None
            if primary is not None:
                pt[TASK_PREFIX + name] = primary
        points.append(pt)

        for cat, names in (result.get("category_tasks") or {}).items():
            cat_tasks.setdefault(cat, set()).update(names)

    if not points:
        raise SystemExit(f"no eval results found under {evals_dir}")
    points.sort(key=lambda pt: pt["step"])
    return points, cat_tasks


def stitch_points(segments: list[list[dict]]) -> list[dict]:
    """Merge resumed-run segments into one series ordered by step.

    Segments are given in chronological order; where two segments both
    evaluated the same step (e.g. a re-eval just before the resume
    point), the later segment's point wins.
    """
    by_step: dict[int, dict] = {}
    for points in segments:
        for pt in points:
            by_step[pt["step"]] = pt
    return [by_step[s] for s in sorted(by_step)]


def _infer_single_task(points: list[dict], cat: str) -> str | None:
    """For legacy outputs lacking ``category_tasks``: a category is a
    single-task category iff exactly one task's primary equals its macro
    at every step it appears. A multi-task macro is an average, so it
    (essentially) never coincides with a single member's score."""
    cat_key = CAT_PREFIX + cat
    candidates: set[str] | None = None
    for pt in points:
        if cat_key not in pt:
            continue
        cv = pt[cat_key]
        here = {
            k[len(TASK_PREFIX):]
            for k in pt
            if k.startswith(TASK_PREFIX) and pt[k] == cv
        }
        candidates = here if candidates is None else (candidates & here)
    if candidates and len(candidates) == 1:
        return next(iter(candidates))
    return None


def resolve_layout(
    series: list[tuple[str, list[dict]]],
    cat_tasks_per_run: list[dict[str, set[str]]],
) -> list[tuple[str, str]]:
    """Build the ordered list of (point-key, panel-label) to plot.

    Single-task categories are relabeled to their task name and the
    duplicate per-task panel is suppressed; multi-task categories keep
    their name (marked as an average) and their tasks each get a panel.
    """
    merged: dict[str, set[str]] = {}
    for ct in cat_tasks_per_run:
        for cat, names in ct.items():
            merged.setdefault(cat, set()).update(names)

    # Categories and tasks present anywhere, in first-seen order.
    cat_order: list[str] = []
    task_order: list[str] = []
    for _, points in series:
        for pt in points:
            for k in pt:
                if k.startswith(CAT_PREFIX) and k[len(CAT_PREFIX):] not in cat_order:
                    cat_order.append(k[len(CAT_PREFIX):])
                elif k.startswith(TASK_PREFIX) and k[len(TASK_PREFIX):] not in task_order:
                    task_order.append(k[len(TASK_PREFIX):])

    rename: dict[str, str] = {}
    suppressed: set[str] = set()
    for cat in cat_order:
        if cat in merged:
            task = next(iter(merged[cat])) if len(merged[cat]) == 1 else None
        else:
            inferred = {
                t
                for (_, points) in series
                if (t := _infer_single_task(points, cat)) is not None
            }
            task = next(iter(inferred)) if len(inferred) == 1 else None
        if task is not None:
            rename[cat] = task
            suppressed.add(task)

    metrics: list[tuple[str, str]] = [("macro", "macro (overall)")]
    for cat in cat_order:
        if cat in rename:
            metrics.append((CAT_PREFIX + cat, rename[cat]))
        else:
            metrics.append((CAT_PREFIX + cat, f"{cat} (category avg)"))
    for task in task_order:
        if task not in suppressed:
            metrics.append((TASK_PREFIX + task, task))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(prog="plot_evals")
    p.add_argument("runs", nargs="+", help="paths to runs/<RUN_ID>/")
    p.add_argument("--labels", nargs="*", help="display label per run")
    p.add_argument("--output", "-o", help="save figure instead of showing")
    p.add_argument(
        "--stitch",
        action="store_true",
        help="treat the runs as chronological segments of one resumed run "
        "and merge them into a single series (later runs win on "
        "overlapping steps)",
    )
    args = p.parse_args()

    if args.stitch:
        if args.labels and len(args.labels) != 1:
            raise SystemExit("--stitch takes at most one --labels entry")
        label = args.labels[0] if args.labels else Path(args.runs[0]).name
        loaded = [load_run(Path(r)) for r in args.runs]
        series = [(label, stitch_points([points for points, _ in loaded]))]
        cat_tasks_per_run = [ct for _, ct in loaded]
    else:
        labels = args.labels or [Path(r).name for r in args.runs]
        if len(labels) != len(args.runs):
            raise SystemExit("--labels count must match number of runs")

        loaded = [(label, *load_run(Path(r))) for label, r in zip(labels, args.runs)]
        series = [(label, points) for label, points, _ in loaded]
        cat_tasks_per_run = [ct for _, _, ct in loaded]

    metrics = resolve_layout(series, cat_tasks_per_run)

    n_cols = min(len(metrics), 3)
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    for i, (key, label) in enumerate(metrics):
        ax = axes[i // n_cols][i % n_cols]
        for run_label, points in series:
            xs = [pt["step"] for pt in points if pt.get(key) is not None]
            ys = [pt[key] for pt in points if pt.get(key) is not None]
            if xs:
                ax.plot(xs, ys, marker="o", label=run_label)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.3)
        if len(series) > 1:
            ax.legend()

    for j in range(len(metrics), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=120)
        print(f"saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
