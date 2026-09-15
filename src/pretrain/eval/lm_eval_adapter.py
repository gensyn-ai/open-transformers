"""Bridge our model to ``lm-eval-harness``.

We don't optimise this path — anyone wanting deep eval can run lm-eval
post-hoc on a checkpoint (plan/07 §2 last paragraph).

The adapter wraps our model in the LM interface that lm-eval expects:
``loglikelihood``, ``loglikelihood_rolling``, ``generate_until``.
``generate_until`` uses naive greedy decoding without a KV cache. No
task in our registered DCLM-CORE set currently needs it (the generative
tasks are omitted — see ``dclm_core``); it's kept only so direct,
post-hoc lm-eval runs against generation tasks still work.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:
    from lm_eval.api.model import LM
except Exception:  # pragma: no cover - optional dependency
    LM = object  # type: ignore[assignment]


class PretrainLM(LM):    # type: ignore[misc]
    """Minimal lm-eval LM adapter.

    Construct with a ready-made ``model`` (already on device, eval-mode)
    and a ``tokenizer`` exposing ``encode`` / ``decode`` / ``vocab_size``.
    """

    def __init__(self, model: torch.nn.Module, tokenizer, device: torch.device | str = "cuda") -> None:
        super().__init__()
        self._model = model
        self._tok = tokenizer
        self._device = torch.device(device)

    @property
    def eot_token_id(self) -> int:
        return self._tok.eos_id

    @property
    def max_length(self) -> int:
        return getattr(self._model.cfg, "max_seq_len_pretrain", 4096)

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self._tok.encode(context)
            cont_ids = self._tok.encode(continuation)
            ll, is_greedy = self._loglikelihood_pair(ctx_ids, cont_ids)
            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        results: list[float] = []
        for req in requests:
            (text,) = req.args
            ids = self._tok.encode(text)
            ll, _ = self._loglikelihood_pair([self.eot_token_id], ids)
            results.append(ll)
        return results

    def generate_until(self, requests: list[Any]) -> list[str]:
        results: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", 256))
            results.append(self._greedy_generate(context, until, max_gen_toks))
        return results

    @torch.no_grad()
    def _greedy_generate(
        self, context: str, until: list[str], max_gen_toks: int
    ) -> str:
        ctx_ids = self._tok.encode(context)
        # Reserve room for the generation so we don't have to slide the window mid-decode.
        max_ctx = max(1, self.max_length - max_gen_toks)
        if len(ctx_ids) > max_ctx:
            ctx_ids = ctx_ids[-max_ctx:]
        x = torch.tensor(ctx_ids, dtype=torch.long, device=self._device).unsqueeze(0)

        produced: list[int] = []
        for _ in range(max_gen_toks):
            out = self._model(x)
            logits = out.logits if hasattr(out, "logits") else out
            next_id = int(logits[0, -1].argmax().item())
            if next_id == self.eot_token_id:
                break
            produced.append(next_id)
            next_t = torch.tensor([[next_id]], dtype=torch.long, device=self._device)
            x = torch.cat([x, next_t], dim=1)
            if until and any(s in self._tok.decode(produced) for s in until):
                break

        text = self._tok.decode(produced)
        cut = len(text)
        for s in until:
            i = text.find(s)
            if 0 <= i < cut:
                cut = i
        return text[:cut]

    @torch.no_grad()
    def _loglikelihood_pair(
        self, ctx_ids: list[int], cont_ids: list[int]
    ) -> tuple[float, bool]:
        if not cont_ids:
            return 0.0, True
        full = ctx_ids + cont_ids
        # truncate from the left if needed
        full = full[-self.max_length :]
        x = torch.tensor(full, dtype=torch.long, device=self._device).unsqueeze(0)
        out = self._model(x[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out
        # The continuation occupies the last len(cont_ids) positions of the
        # input; the corresponding logits are at the same positions.
        cont_len = min(len(cont_ids), logits.size(1))
        target = x[:, -cont_len:]
        cont_logits = logits[:, -cont_len:]
        log_probs = F.log_softmax(cont_logits.float(), dim=-1)
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        ll = float(gathered.sum().item())
        greedy_tokens = cont_logits.argmax(dim=-1)
        is_greedy = bool(torch.equal(greedy_tokens, target))
        return ll, is_greedy
