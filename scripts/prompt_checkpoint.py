#!/usr/bin/env python3
"""Interactive prompt REPL against a checkpoint.

Loads a checkpoint produced by the training loop and lets you type
prompts, streaming a completion token-by-token. This is a base model
(no instruction tuning), so think continuation rather than chat: feed
it the opening of a paragraph and watch what it writes next.

Usage:
    python scripts/prompt_checkpoint.py \\
        --config-name 1b_proxy_repop \\
        --checkpoint runs/<RUN_ID>/checkpoints/step_<N> \\
        --tokenizer data/tokenizer.json

Meta-commands at the prompt:
    :temp <float>    sampling temperature (0 = greedy)
    :topk <int>      top-k filter (0 = off)
    :topp <float>    nucleus filter (1.0 = off)
    :tokens <int>    max new tokens per generation
    :seed <int>      reseed the RNG
    :show            print current settings
    :help            this list
    :quit / Ctrl-D   exit
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from pretrain.config import load_config
from pretrain.data.tokenizer import Tokenizer
from pretrain.model import build_model
from pretrain.parallel import ParallelDims, parallelize_llama3_repop


@dataclass
class SampleParams:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 200

    def describe(self) -> str:
        return (
            f"temp={self.temperature} top_k={self.top_k} "
            f"top_p={self.top_p} max_new_tokens={self.max_new_tokens}"
        )


def _load_model(args: argparse.Namespace, device: torch.device):
    cfg = load_config(args.config_name, overrides=args.override)
    ckpt_dir = Path(args.checkpoint)
    # Untrusted-input gate up front (this tool is pointed at downloaded/shared
    # checkpoints): reject a doctored dcp/.metadata before the model build.
    from pretrain.train.checkpoint import _validate_dcp_metadata, load_dcp_validated

    _validate_dcp_metadata(ckpt_dir / "dcp")
    model = build_model(cfg.model, device=device)
    # Mirror the training-time wrap so DCP keys match. ws=1 means FSDP is
    # a no-op (TP no longer exists as a dim — removed with torchtitan's
    # extra dims); activation checkpointing is still applied and inserts
    # the ``_checkpoint_wrapped_module.`` prefix the checkpoint carries.
    pdims = ParallelDims(dp_replicate=1, dp_shard=1, world_size=1)
    model = parallelize_llama3_repop(model, cfg, pdims)
    state = {"model": model.state_dict()}
    load_dcp_validated(state, ckpt_dir / "dcp")
    model.load_state_dict(state["model"])
    model.eval()
    return cfg, model


def _filter_logits(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Apply temperature + top-k + top-p to a [vocab] logit row."""
    if temperature <= 0:
        # Greedy: collapse to a one-hot at argmax via a huge logit.
        idx = int(logits.argmax().item())
        out = torch.full_like(logits, float("-inf"))
        out[idx] = 0.0
        return out

    logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Mask tokens past the nucleus; always keep the top-1.
        cutoff = cum > top_p
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    params: SampleParams,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    max_seq_len: int,
) -> None:
    """Stream a continuation to stdout. Does not return text."""
    ids = tokenizer.encode(prompt)
    # Leave room for new tokens.
    ctx_budget = max(1, max_seq_len - params.max_new_tokens)
    if len(ids) > ctx_budget:
        ids = ids[-ctx_budget:]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    eos = tokenizer.eos_id
    produced: list[int] = []
    last_decoded_len = 0

    def _autocast():
        if autocast_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    # Echo the prompt so the continuation flows naturally on screen.
    sys.stdout.write(prompt)
    sys.stdout.flush()

    # repop's flash-attention kernels require seq_len % 32 == 0 (the _32_256
    # block family). Generation feeds arbitrary lengths, so right-pad each
    # forward up to the next multiple of 32 — the pad tokens sit at future
    # positions that causal masking never lets a real token attend — and read
    # the logits at the TRUE last token, not the padded tail.
    ATTN_BLOCK = 32
    for _ in range(params.max_new_tokens):
        real_len = x.size(1)
        pad = (-real_len) % ATTN_BLOCK
        x_in = F.pad(x, (0, pad), value=eos) if pad else x
        with _autocast():
            out = model(x_in)
        logits = out.logits if hasattr(out, "logits") else out
        next_logits = logits[0, real_len - 1].float()
        filtered = _filter_logits(
            next_logits, params.temperature, params.top_k, params.top_p
        )
        probs = F.softmax(filtered, dim=-1)
        if params.temperature <= 0:
            next_id = int(torch.argmax(probs).item())
        else:
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_id == eos:
            break

        produced.append(next_id)
        # Decode incrementally: BPE tokens don't map 1:1 to characters,
        # so decode the whole produced list each step and write the delta.
        text = tokenizer.decode(produced)
        delta = text[last_decoded_len:]
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
            last_decoded_len = len(text)

        next_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
        x = torch.cat([x, next_t], dim=1)
        if x.size(1) >= max_seq_len:
            break

    sys.stdout.write("\n")
    sys.stdout.flush()


def _handle_command(line: str, params: SampleParams) -> bool:
    """Returns True if the line was a meta-command (and was handled)."""
    if not line.startswith(":"):
        return False
    parts = line[1:].split()
    if not parts:
        return True
    cmd, *rest = parts
    try:
        if cmd in ("q", "quit", "exit"):
            raise SystemExit(0)
        if cmd in ("h", "help", "?"):
            print(__doc__)
        elif cmd == "show":
            print(params.describe())
        elif cmd == "temp" and rest:
            params.temperature = float(rest[0])
            print(f"temperature = {params.temperature}")
        elif cmd == "topk" and rest:
            params.top_k = int(rest[0])
            print(f"top_k = {params.top_k}")
        elif cmd == "topp" and rest:
            params.top_p = float(rest[0])
            print(f"top_p = {params.top_p}")
        elif cmd == "tokens" and rest:
            params.max_new_tokens = int(rest[0])
            print(f"max_new_tokens = {params.max_new_tokens}")
        elif cmd == "seed" and rest:
            torch.manual_seed(int(rest[0]))
            print(f"seeded RNG with {int(rest[0])}")
        else:
            print(f"unknown command: :{cmd}  (try :help)")
    except (ValueError, IndexError) as e:
        print(f"bad argument: {e}")
    return True


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="prompt_checkpoint")
    p.add_argument("--config-name", required=True,
                   help="train config (e.g. 1b_proxy_repop)")
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint dir containing dcp/")
    p.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    p.add_argument("--override", nargs="*", default=[],
                   help="hydra-style config overrides")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="autocast dtype on CUDA (ignored on CPU)")
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps; default: cuda if available")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading checkpoint on {device} ...", file=sys.stderr)
    cfg, model = _load_model(args, device)
    tokenizer = Tokenizer(args.tokenizer)
    print(
        f"loaded {model.num_parameters() / 1e9:.2f}B params, "
        f"vocab={tokenizer.vocab_size}, max_seq_len={cfg.model.max_seq_len_pretrain}",
        file=sys.stderr,
    )

    autocast_dtype: torch.dtype | None
    if device.type == "cuda":
        autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    else:
        autocast_dtype = None

    params = SampleParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    print(
        "Enter a prompt to continue. :help for commands, :quit or Ctrl-D to exit.",
        file=sys.stderr,
    )
    print(f"settings: {params.describe()}", file=sys.stderr)

    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        stripped = line.strip()
        if not stripped:
            continue
        if _handle_command(stripped, params):
            continue
        try:
            generate(
                model=model,
                tokenizer=tokenizer,
                prompt=line,
                params=params,
                device=device,
                autocast_dtype=autocast_dtype,
                max_seq_len=cfg.model.max_seq_len_pretrain,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue


if __name__ == "__main__":
    main()
