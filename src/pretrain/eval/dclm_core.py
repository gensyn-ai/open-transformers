"""DCLM-CORE runner — the loglikelihood subset we actually score.

We track a small, MMLU-centric slice of DCLM CORE rather than the full
~53-task suite. The generative tasks (humaneval, trivia_qa) are omitted:
our lm-eval adapter has no KV cache, so they would dominate per-checkpoint
cost. What remains is the multiple-choice / loglikelihood set, grouped
into categories in ``DCLM_CORE_TASKS``.

The runner:

  - registers tasks via a small dict (``DCLM_CORE_TASKS``);
  - dispatches each through ``lm_eval`` (with per-task few-shot from
    ``DCLM_CORE_FEWSHOT``);
  - writes ``runs/<run_id>/evals/<step>/step_<step>.json`` with per-task
    primaries, per-category macros, the category->task mapping, and the
    overall macro-average. ``scripts/plot_evals.py`` renders these.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

LOG = logging.getLogger(__name__)


# Categories matching the DCLM CORE breakdown. Filled in lazily; the
# runner happily evaluates whichever subset is registered.
DCLM_CORE_TASKS: dict[str, list[str]] = {
    "commonsense": ["hellaswag", "piqa", "winogrande"],
    "world_knowledge": ["mmlu"],
    "reading": ["boolq", "openbookqa"],
    # Generative tasks (trivia_qa, humaneval) omitted: our adapter has
    # no KV cache, so per-checkpoint cost is dominated by them.
}


# Per-task few-shot counts. Tasks not listed default to 0-shot.
DCLM_CORE_FEWSHOT: dict[str, int] = {
    "mmlu": 5,
}


def run_dclm_core(
    model: torch.nn.Module,
    tokenizer,
    out_dir: str | Path,
    step: int,
    task_subset: dict[str, list[str]] | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Execute the registered tasks via lm-eval and persist results.

    Returns the same dict that's written to disk. Macro-averaging is
    weighted equally per category (DCLM convention).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from lm_eval import evaluator

        from pretrain.eval.lm_eval_adapter import PretrainLM

        lm = PretrainLM(model, tokenizer, device=device)
    except Exception as e:
        LOG.warning("lm-eval unavailable (%s); writing an empty result", e)
        result = {"step": step, "error": str(e), "categories": {}}
        (out_dir / f"step_{step:09d}.json").write_text(json.dumps(result, indent=2))
        return result

    tasks = task_subset or DCLM_CORE_TASKS
    per_task: dict[str, dict[str, float]] = {}
    cat_macros: dict[str, float] = {}

    for cat, names in tasks.items():
        cat_scores: list[float] = []
        for task_name in names:
            try:
                r = evaluator.simple_evaluate(
                    model=lm,
                    tasks=[task_name],
                    num_fewshot=DCLM_CORE_FEWSHOT.get(task_name, 0),
                )
                metrics = r.get("results", {}).get(task_name, {})
                primary = _primary_metric(metrics)
                per_task[task_name] = {"primary": primary, "all": metrics}
                cat_scores.append(primary)
            except Exception as e:
                LOG.warning("task %s failed: %s", task_name, e)
                per_task[task_name] = {"error": str(e)}
        if cat_scores:
            cat_macros[cat] = sum(cat_scores) / len(cat_scores)

    macro = (
        sum(cat_macros.values()) / max(len(cat_macros), 1) if cat_macros else 0.0
    )
    result = {
        "step": step,
        "tasks": per_task,
        "categories": cat_macros,
        # Category -> task membership so downstream plotting can relabel
        # single-task categories (e.g. world_knowledge -> mmlu) without
        # re-deriving the mapping from this module.
        "category_tasks": {cat: list(names) for cat, names in tasks.items()},
        "macro_average": macro,
    }
    (out_dir / f"step_{step:09d}.json").write_text(json.dumps(result, indent=2))
    LOG.info(
        "DCLM CORE @ step %d: macro=%.4f categories=%s",
        step, macro, {k: round(v, 4) for k, v in cat_macros.items()},
    )
    return result


def _primary_metric(metrics: dict[str, float]) -> float:
    """Pick a sensible primary score from an lm-eval result dict."""
    for k in ("acc_norm,none", "acc,none", "exact_match,none", "pass_at_1,none"):
        if k in metrics:
            return float(metrics[k])
    # Fallback: first numeric value.
    for v in metrics.values():
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0
