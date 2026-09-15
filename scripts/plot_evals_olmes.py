"""Plot OLMES eval results from one or more run directories.

Companion to ``plot_evals.py`` (same CLI), reading the output of
``pretrain.cli.eval_olmes`` instead:

    python scripts/plot_evals_olmes.py runs/<RUN_ID>
    python scripts/plot_evals_olmes.py runs/<a> runs/<b> --labels torch repop
    python scripts/plot_evals_olmes.py runs/<a> -o olmes.png
    python scripts/plot_evals_olmes.py runs/<a> runs/<a-resume1> --stitch

Reads each ``evals_olmes/step_<N>/summary.json`` (aggregates: per-task
best-of-MC/RC, suite macros) plus ``metrics.json`` (per-leaf scores) and
draws, vs training step:

  - the 10-task OLMES macro and the core_9mcqa macro,
  - one panel per task: the OLMES score (best of the two formulations,
    solid) with the underlying MCF (dashed) and CF/RC (dotted) traces —
    the MCF-vs-CF crossover is itself a capability signal (Gu et al.
    2024, Fig. 1), so the panels keep both visible.

If a run dir also holds ``evals_olmes_ext/step_<N>/summary.json`` (the
extension suite: AGIEval, MMLU-Pro,
NaturalQs, DROP, TriviaQA, GSM8K), one single-trace panel per extension
task is appended. Ext-only checkpoints (evaluated on the extension but
not the MC standard) contribute points to just those panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Canonical OLMES panel order (Gu et al. 2024, Table 2).
OLMES_TASK_ORDER = [
    "arc_easy", "arc_challenge", "boolq", "csqa", "hellaswag",
    "openbookqa", "piqa", "socialiqa", "winogrande", "mmlu",
]
FORM_STYLES = {"best": "-", "mc": "--", "rc": ":"}

# Extension-suite panels, in display order:
# panel key -> the alias its headline score carries in summary.json.
# agi_eval/mmlu_pro are suite aggregates; the rest are single tasks that
# summarize_metrics files under "extended_tasks" (num_leaf_tasks counts
# only the OLMES-standard leaves).
EXT_TASK_ALIASES = {
    "agi_eval_english": "agi_eval_english:1shot::olmes",
    "mmlu_pro": "mmlu_pro:mc::none",
    "naturalqs": "naturalqs::olmes",
    "drop": "drop::olmes",
    "triviaqa": "triviaqa::olmes",
    "gsm8k": "gsm8k::olmes",
}


def load_run(run_dir: Path) -> list[dict]:
    """Return one point per evaluated checkpoint.

    Each point holds ``step``, ``macro``, ``core9``, and per task
    ``<task>:best`` / ``<task>:mc`` / ``<task>:rc``. Best-of and suite
    scores come from summary.json; the mc/rc traces come from the leaf
    entries in metrics.json (mmlu's from its per-formulation aggregates,
    since its leaves are the 57 subjects).
    """
    evals_dir = run_dir / "evals_olmes"
    ext_dir = run_dir / "evals_olmes_ext"
    if not evals_dir.exists() and not ext_dir.exists():
        raise SystemExit(f"no evals_olmes[_ext] dir under {run_dir}")

    points: list[dict] = []
    if not evals_dir.exists():
        evals_dir = Path("/nonexistent")  # ext-only run dir
    for d in sorted(evals_dir.glob("step_*")):
        summary_file = d / "summary.json"
        if not summary_file.exists():
            continue  # checkpoint not (fully) evaluated
        aggs = json.loads(summary_file.read_text()).get("aggregates") or {}

        pt: dict = {
            "step": int(d.name.removeprefix("step_")),
            "macro": _olmes_macro(aggs),
            "core9": aggs.get("core_9mcqa::olmes"),
        }
        for alias, score in aggs.items():
            task = alias.removesuffix("::olmes")
            if ":" not in task and task in OLMES_TASK_ORDER:
                pt[f"{task}:best"] = score
            elif task.endswith((":mc", ":rc")):  # e.g. mmlu:mc::olmes
                name, form = task.rsplit(":", 1)
                if name in OLMES_TASK_ORDER:
                    pt[f"{name}:{form}"] = score

        metrics_file = d / "metrics.json"
        if metrics_file.exists():
            for entry in json.loads(metrics_file.read_text()).get("tasks", []):
                alias = entry.get("alias", "")
                score = (entry.get("metrics") or {}).get("primary_score")
                task = alias.removesuffix("::olmes")
                if score is None or not task.endswith((":mc", ":rc")):
                    continue
                name, form = task.rsplit(":", 1)
                if name in OLMES_TASK_ORDER:
                    pt.setdefault(f"{name}:{form}", score)
        points.append(pt)

    # Extension-suite results live beside the standard ones; a checkpoint
    # may have either or both, so merge by step.
    by_step: dict[int, dict] = {pt["step"]: pt for pt in points}
    for d in sorted(ext_dir.glob("step_*")) if ext_dir.exists() else []:
        summary_file = d / "summary.json"
        if not summary_file.exists():
            continue
        summary = json.loads(summary_file.read_text())
        scores = {**(summary.get("extended_tasks") or {}),
                  **(summary.get("aggregates") or {})}
        pt = by_step.setdefault(int(d.name.removeprefix("step_")), {})
        pt["step"] = int(d.name.removeprefix("step_"))
        for key, alias in EXT_TASK_ALIASES.items():
            if scores.get(alias) is not None:
                pt[f"ext:{key}"] = scores[alias]

    if not by_step:
        raise SystemExit(f"no summary.json results under {run_dir}")
    return [by_step[s] for s in sorted(by_step)]


def _olmes_macro(aggs: dict) -> float | None:
    """Equal-weight macro over the 10 OLMES tasks (core9 counts as 9)."""
    core = aggs.get("core_9mcqa::olmes")
    mmlu = aggs.get("mmlu::olmes")
    if core is None or mmlu is None:
        return None
    return (core * 9 + mmlu) / 10


def stitch_points(segments: list[list[dict]]) -> list[dict]:
    """Merge resumed-run segments into one series ordered by step.

    Segments are given in chronological order; where two segments both
    evaluated the same step, the later segment's point wins.
    """
    by_step: dict[int, dict] = {}
    for points in segments:
        for pt in points:
            by_step[pt["step"]] = pt
    return [by_step[s] for s in sorted(by_step)]


def resolve_layout(series: list[tuple[str, list[dict]]]) -> list[tuple[str, str]]:
    """Ordered (panel-key, panel-label) list: macros, then tasks seen."""
    seen = {k.split(":")[0] for _, points in series for pt in points for k in pt}
    seen_ext = {
        k.removeprefix("ext:")
        for _, points in series for pt in points for k in pt
        if k.startswith("ext:")
    }
    metrics = [("macro", "OLMES macro (10-task)"), ("core9", "core_9mcqa (macro)")]
    metrics += [(t, t) for t in OLMES_TASK_ORDER if t in seen]
    metrics += [(f"ext:{t}", f"{t} [ext]") for t in EXT_TASK_ALIASES if t in seen_ext]
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(prog="plot_evals_olmes")
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
    p.add_argument(
        "--best-only",
        action="store_true",
        help="plot only the best-of-MC/RC score per task (drop the "
        "per-formulation traces)",
    )
    args = p.parse_args()

    if args.stitch:
        if args.labels and len(args.labels) != 1:
            raise SystemExit("--stitch takes at most one --labels entry")
        label = args.labels[0] if args.labels else Path(args.runs[0]).name
        series = [(label, stitch_points([load_run(Path(r)) for r in args.runs]))]
    else:
        labels = args.labels or [Path(r).name for r in args.runs]
        if len(labels) != len(args.runs):
            raise SystemExit("--labels count must match number of runs")
        series = [(label, load_run(Path(r))) for label, r in zip(labels, args.runs)]

    metrics = resolve_layout(series)
    forms = ["best"] if args.best_only else ["best", "mc", "rc"]

    n_cols = min(len(metrics), 3)
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    for i, (key, label) in enumerate(metrics):
        ax = axes[i // n_cols][i % n_cols]
        # Ext tasks have a single formulation — plot them like the macros.
        is_task = key not in ("macro", "core9") and not key.startswith("ext:")
        for run_idx, (run_label, points) in enumerate(series):
            color = f"C{run_idx}"
            for form in forms if is_task else ["best"]:
                pk = f"{key}:{form}" if is_task else key
                xs = [pt["step"] for pt in points if pt.get(pk) is not None]
                ys = [pt[pk] for pt in points if pt.get(pk) is not None]
                if not xs:
                    continue
                best = form == "best"
                ax.plot(
                    xs, ys,
                    linestyle=FORM_STYLES[form],
                    marker="o" if best else None,
                    color=color,
                    alpha=1.0 if best else 0.6,
                    linewidth=1.8 if best else 1.0,
                    label=run_label if best else (form if run_idx == 0 else None),
                    # Earlier runs draw on top: where series share a step
                    # (e.g. a resume/midtrain series anchored on the base
                    # run's final checkpoint), the base run's point must
                    # stay visible. Best-of stays above the mc/rc traces.
                    zorder=2 + (0.2 if best else 0) + 0.1 * (len(series) - 1 - run_idx),
                )
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.3)
        if len(series) > 1 or (is_task and not args.best_only):
            ax.legend(fontsize=8)

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
