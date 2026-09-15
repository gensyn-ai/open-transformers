"""OLMES runner — the full standard from Gu et al., 2024 (arXiv:2406.08446).

Runs the reference implementation (github.com/allenai/olmes, ``oe_eval``)
against our model: 10 MCQA tasks, curated 5-shot prompts, both MCF and CF
formulations with per-task CF normalization, and the best-of-both scoring
rule. This is the externally comparable eval; ``dclm_core`` remains the
cheap in-loop tracker.

Integration shape: ``oe_eval`` has no plugin registry for model backends
(``load_model`` hardcodes hf/vllm/litellm/olmo_core), so ``run_olmes``
swaps ``oe_eval.run_eval.load_model`` for one that returns our LM and then
drives ``oe_eval.run_eval.run_eval`` as a library. The LM subclasses
olmes' ``HFLM_Verbose`` with a duck-typed "HF model": lm-eval's HFLM only
touches ``.device``/``.config``/``.tie_weights()``/``forward().logits``
on it. All of this is exercised against the *pinned* olmes commit
``OLMES_PIN`` below — bump the pin and this module must be re-smoked
end to end.

The OLMES suites are all loglikelihood-scored (MCF ranks the answer-label
tokens), so the missing-KV-cache limitation of our stack never bites.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

LOG = logging.getLogger(__name__)

# The olmes harness this module is written against. It is not a declared
# dependency (heavy, and only needed on eval boxes) — the eval job installs it
# with:
#
#   pip install --no-deps \
#       'ai2-olmes @ git+https://github.com/allenai/olmes.git@5a51f502d463b8cdc4a2dcad7d7096c41ff1197e'
#
# alongside ``lm-eval==0.4.3`` (NOT 0.4.4 — olmes reaches into HFLM internals
# that 0.4.4 moved) and ``ai2-olmo-core==2.4.0``. Bumping the pin means
# re-smoking this module end to end: the duck-typed LM below depends on
# HFLM internals that are not part of any stable API.
OLMES_PIN = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"


def _shim_fsdp2_names() -> None:
    """Alias FSDP2 names into torch.distributed.fsdp when missing.

    NGC 25.01 ships a torch 2.6.0a0 cut that predates the public re-export
    of the FSDP2 API (``FSDPModule``, ``fully_shard``, ...) out of
    ``torch.distributed._composable.fsdp`` — and ai2-olmo-core (a hard
    import of oe_eval's model registry) imports them from the public path
    at module load. The aliases are the same objects, additive only, and a
    no-op on torch >= 2.6 GA. Validated on the NGC image itself.
    """
    import torch.distributed.fsdp as fsdp

    if hasattr(fsdp, "FSDPModule"):
        return
    import torch.distributed._composable.fsdp as cfsdp

    for name in dir(cfsdp):
        if not name.startswith("_") and not hasattr(fsdp, name):
            setattr(fsdp, name, getattr(cfsdp, name))


# Must run before any oe_eval import (they all happen lazily below, so
# module-import time is early enough for every path through this file).
_shim_fsdp2_names()

# The full OLMES standard: 9 MCQA tasks + MMLU (57 subjects), each in both
# MCF (":mc") and CF (":rc") formulations — 132 leaf tasks. The suite
# definitions live in oe_eval.configs.task_suites; aggregation back into
# per-task best-of and the overall macro happens inside oe_eval.run_eval.
OLMES_DEFAULT_SUITES = ["core_9mcqa::olmes", "mmlu::olmes"]

# OLMES restricts all inputs (context + continuation) to 2048 tokens for
# consistency across models (paper §3.5), regardless of the model's own
# context length.
OLMES_MAX_LENGTH = 2048

# Extension beyond the 10-task standard: the OLMo 2 paper's remaining dev
# and held-out tasks (its Tables 6/9 columns), so a checkpoint can be lined
# up against that paper in full. AGIEval/MMLU-Pro are MCF loglikelihood;
# the other four are generative and need the greedy `_model_generate` path
# below — budget several extra hours per checkpoint at 1B (no KV cache;
# TriviaQA's full 7,993-instance validation set dominates). At early base
# checkpoints the generative scores carry little signal; this list is meant
# for anchor/final checkpoints, hence opt-in (--extended / explicit --task).
OLMES_EXTENDED_TASKS = [
    "agi_eval_english:1shot::olmes",  # 8 MC subtasks, 1-shot, MCF-only
    "mmlu_pro:mc::none",              # 14 domains, 5-shot, MCF-only, micro
    "naturalqs::olmes",               # generative, 5-shot, F1, 1000 val
    "drop::olmes",                    # generative, 5-shot, F1, 1000 val
    "triviaqa::olmes",                # generative, 5-shot, F1, full 7993 val
    "gsm8k::olmes",                   # generative, 8-shot CoT, EM, 1319 test
]

# Special-token strings as trained into data/tokenizer.json (see
# pretrain.data.tokenizer.train_tokenizer). <|endoftext|> doubles as the
# document separator, so it is the right "empty context" prefix token.
_EOS_TOKEN = "<|endoftext|>"
_BOS_TOKEN = "<|begin_of_text|>"
_PAD_TOKEN = "<|pad|>"


class _CausalWrapper(torch.nn.Module):
    """Duck-typed ``transformers.PreTrainedModel`` around our Llama3.

    lm-eval's HFLM, handed a pre-built model object, reads ``.device`` and
    ``.config``, calls ``.eval()``/``.tie_weights()``, and expects
    ``forward(input_ids).logits``. Nothing else.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device | str) -> None:
        super().__init__()
        self.wrapped = model
        self._device = torch.device(device)
        # getattr-probed only (gemma/qwen special cases, seq-len fallbacks);
        # max_length is always passed explicitly so no seq-len attr is needed.
        self.config = SimpleNamespace(model_type="pretrain_llama3")

    @property
    def device(self) -> torch.device:
        return self._device

    def tie_weights(self) -> None:  # HFLM calls this unconditionally
        pass

    # repop's flash kernels want seq_len % 32 == 0 (training always feeds
    # fixed multiples; lm-eval feeds ragged prompts). Right-padding is
    # causally inert — trailing tokens cannot influence earlier logits —
    # and the pad columns are sliced off before returning.
    _SEQ_MULTIPLE = 32

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        T = input_ids.shape[1]
        pad = -T % self._SEQ_MULTIPLE
        if pad:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=0)
        out = self.wrapped(input_ids)
        logits = out.logits if hasattr(out, "logits") else out
        if pad:
            logits = logits[:, :T]
        # HF causal LMs upcast logits to fp32 before returning; match that
        # so log_softmax runs in fp32 like it would for an HF checkpoint.
        return SimpleNamespace(logits=logits.float())


def build_hf_tokenizer(tokenizer_path: str | Path):
    """Wrap our frozen tokenizer.json as a transformers fast tokenizer.

    The file is already HF ``tokenizers`` format; only the special-token
    registration is missing (PreTrainedTokenizerFast does not read it from
    the file). pad is registered so lm-eval's configure_pad_token doesn't
    fall back to mutating eos.
    """
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        eos_token=_EOS_TOKEN,
        bos_token=_BOS_TOKEN,
        pad_token=_PAD_TOKEN,
    )


def _build_lm(
    model: torch.nn.Module,
    hf_tokenizer,
    device: torch.device | str,
    batch_size: int,
    max_length: int,
):
    """Construct the olmes LM adapter for a ready-made model."""
    from lm_eval.models.huggingface import HFLM

    from oe_eval.models.eleuther_huggingface import HFLM_Verbose

    class PretrainOlmesLM(HFLM_Verbose):
        def __init__(self) -> None:
            # Skip HFLM_Verbose.__init__: its multi-GPU heuristics
            # (parallelize=True when device_count > 1) assert against
            # pre-built model objects. HFLM itself handles them.
            HFLM.__init__(
                self,
                pretrained=_CausalWrapper(model, device),
                tokenizer=hf_tokenizer,
                backend="causal",  # skip config.model_type sniffing
                batch_size=int(batch_size),
                max_length=int(max_length),
            )
            # The one piece of HFLM_Verbose.__init__ its methods rely on.
            self.tokenizer_size = len(self.tokenizer)

        def _model_generate(self, context, attention_mask, stop, **kwargs):
            """Greedy decoding shaped like HF ``generate`` output.

            HFLM's version calls ``self.model.generate`` (GenerationMixin),
            which our wrapper doesn't have — and can't batch-decode
            faithfully anyway: the model takes no attention mask (repop
            causal-flash only), so HFLM's left-padded batches would attend
            to pad prefixes. Decode each row independently instead (no KV
            cache: one full-prefix forward per generated token — the cost
            note on OLMES_EXTENDED_TASKS).

            The caller consumes ``output["sequences"]`` ([B, ctx+gen], rows
            padded with pad_token after their end) and ``output["scores"]``
            (per-step [B, vocab] logits). Rows that stop early are padded
            with pad_token — the caller trims at the first pad/eos, so the
            zero-filled scores past a row's end are never read (pad != eos
            in our tokenizer, which that trimming relies on).
            """
            if kwargs.get("do_sample") or (kwargs.get("temperature") or 0.0) > 0.0:
                raise NotImplementedError("PretrainOlmesLM only supports greedy decoding")
            max_new = max(int(kwargs.get("max_length", context.shape[1] + self.max_gen_toks)) - context.shape[1], 1)
            B, ctx_len = context.shape
            pad_id = self.tokenizer.pad_token_id
            produced_rows: list[list[int]] = []
            score_rows: list[list[torch.Tensor]] = []
            for b in range(B):
                prefix = context[b][attention_mask[b].bool()].tolist()  # strip left pads
                produced: list[int] = []
                scores: list[torch.Tensor] = []
                for _ in range(max_new):
                    inp = torch.tensor([prefix + produced], dtype=torch.long, device=self.device)
                    # .clone(): the slice is a view into the full [1, T, V]
                    # fp32 logits (~1 GB at T=2048) — retaining it per step
                    # pins the whole parent tensor and OOMs within ~50 steps.
                    logits = self._model_call(inp)[0, -1].clone()
                    scores.append(logits)
                    nxt = int(logits.argmax())
                    produced.append(nxt)
                    if nxt == self.eot_token_id:
                        break
                    if stop and any(s in self.tok_decode(produced) for s in stop):
                        break
                produced_rows.append(produced)
                score_rows.append(scores)
            gen_len = max(len(p) for p in produced_rows)
            sequences = torch.full(
                (B, ctx_len + gen_len), pad_id, dtype=torch.long, device=self.device
            )
            sequences[:, :ctx_len] = context
            vocab = score_rows[0][0].shape[-1]
            scores_t = torch.zeros((B, gen_len, vocab), device=self.device)
            for b, (produced, scores) in enumerate(zip(produced_rows, score_rows)):
                sequences[b, ctx_len : ctx_len + len(produced)] = torch.tensor(
                    produced, dtype=torch.long, device=self.device
                )
                scores_t[b, : len(scores)] = torch.stack(scores)
            return {
                "sequences": sequences,
                "scores": tuple(scores_t[:, t] for t in range(gen_len)),
            }

    return PretrainOlmesLM()


def expand_olmes_tasks(specs: list[str]) -> list[dict[str, Any]]:
    """Expand suite/task names into leaf task configs, launch.py-style.

    ``oe_eval.run_eval`` takes leaf tasks only; suite expansion (and the
    ``metadata.alias`` that suite aggregation later matches on) is done by
    the olmes launcher, which we bypass — so replicate it here.
    """
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    from oe_eval.configs.tasks import TASK_CONFIGS

    leaves: list[str] = []

    def _resolve(name: str) -> None:
        if name in TASK_SUITE_CONFIGS:
            for sub in TASK_SUITE_CONFIGS[name]["tasks"]:
                _resolve(sub)
        else:
            leaves.append(name)

    for spec in specs:
        _resolve(spec)

    configs = []
    for name in leaves:
        if name not in TASK_CONFIGS:
            raise ValueError(f"unknown olmes task {name!r} (not in TASK_CONFIGS)")
        cfg = copy.deepcopy(TASK_CONFIGS[name])
        cfg.setdefault("metadata", {})["alias"] = name
        configs.append(cfg)
    return configs


def run_olmes(
    model: torch.nn.Module,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    tasks: list[str] | None = None,
    extended: bool = False,
    batch_size: int = 4,
    max_length: int = OLMES_MAX_LENGTH,
    limit: int | None = None,
    device: torch.device | str = "cuda",
    model_label: str = "pretrain",
) -> dict[str, Any]:
    """Run OLMES suites on a ready-made model; return a small summary.

    Full per-task output (metrics-all.jsonl, per-task predictions/requests)
    lands in ``out_dir``; the returned dict holds the aggregate primary
    scores plus a top-level macro, and is also written to summary.json.
    ``limit`` overrides each task's instance cap — smoke tests only, it
    breaks OLMES comparability.
    """
    import oe_eval.run_eval as oe_run

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = list(tasks or OLMES_DEFAULT_SUITES)
    if extended:
        specs += [t for t in OLMES_EXTENDED_TASKS if t not in specs]
    task_configs = expand_olmes_tasks(specs)
    if limit is not None:
        for cfg in task_configs:
            cfg["limit"] = limit

    argv = ["--task"]
    argv += [json.dumps(cfg) for cfg in task_configs]
    argv += [
        "--model", model_label,
        "--model-type", "hf",  # metadata only; load_model below is ours
        "--output-dir", str(out_dir),
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
    ]
    args_dict = vars(oe_run._parser.parse_args(argv))

    lm = _build_lm(model, build_hf_tokenizer(tokenizer_path), device, batch_size, max_length)

    # No backend registry in oe_eval — swap the loader for the duration.
    orig_load_model = oe_run.load_model
    oe_run.load_model = lambda _cfg: lm
    try:
        oe_run.run_eval(args_dict)
    finally:
        oe_run.load_model = orig_load_model

    return summarize_metrics(out_dir)


def summarize_metrics(out_dir: str | Path) -> dict[str, Any]:
    """Distill metrics-all.jsonl into summary.json (idempotency sentinel).

    Aggregate entries (the per-task best-of-MC/RC and the suite macros) are
    listed first in metrics-all.jsonl; per-leaf metrics stay in the full
    file. ``olmes_macro`` is the core_9mcqa macro and MMLU averaged with
    equal weight — the 10-task OLMES headline number.
    """
    out_dir = Path(out_dir)
    aggregates: dict[str, float] = {}
    per_task: dict[str, float] = {}
    extended_tasks: dict[str, float] = {}
    with (out_dir / "metrics-all.jsonl").open() as f:
        for line in f:
            m = json.loads(line)
            alias = m.get("task_config", {}).get("metadata", {}).get("alias", m["task_name"])
            score = m.get("metrics", {}).get("primary_score")
            if score is None:
                continue
            # Suite aggregates (per-task best-of-MC/RC, suite macros) are
            # synthesized by oe_eval's add_aggregate_tasks with task_idx=None;
            # real leaf tasks carry their run index.
            if m.get("task_idx") is None:
                aggregates[alias] = score
            elif alias.endswith("::olmes") and (":mc::" in alias or ":rc::" in alias):
                per_task[alias] = score  # core-suite formulation leaves
            else:
                # Extension leaves: generative tasks (their own primary) and
                # the AGIEval/MMLU-Pro sub-tasks (suite macros land in
                # `aggregates` via oe_eval's aggregation).
                extended_tasks[alias] = score

    core = aggregates.get("core_9mcqa::olmes")
    mmlu = aggregates.get("mmlu::olmes")
    macro = None
    if core is not None and mmlu is not None:
        macro = (core * 9 + mmlu) / 10  # per-task macro over the 10 OLMES tasks
    summary = {
        "olmes_macro": macro,
        "aggregates": aggregates,
        "num_leaf_tasks": len(per_task),
    }
    if extended_tasks:
        summary["extended_tasks"] = extended_tasks
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
