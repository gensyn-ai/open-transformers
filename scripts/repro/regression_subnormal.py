#!/usr/bin/env python3
"""Subnormal-domain cross-device reproducibility regression (algorithmic side).

The standard end-to-end regression trains from
NORMAL-magnitude init, so fp32 subnormals never appear and any backend that
flushes-to-zero (Apple MPS is DAZ+FTZ in hardware) vs keeps subnormals
(CPU/CUDA) stays latent — exactly how the cluster-run step~1000 MPS audit
divergence in the AdamW moment tail slipped past the passing regression.

This test deliberately seeds the ENTIRE hashed state — params, grads, and
AdamW moments — with deterministic values that straddle the fp32
subnormal boundary (exponents 2^-100 … 2^-149, i.e. small-normal down through
the deepest subnormal), then runs the REAL post-backward algorithmic ops the
state hash covers (global clip + repop AdamW step) and digests the resulting
state. Run it on each backend and compare:

    python regression_subnormal.py --device cpu  --out cpu.json
    python regression_subnormal.py --device mps  --out mps.json
    python regression_subnormal.py --device cuda --out cuda.json   # on a CUDA box
    python regression_subnormal.py --compare cpu.json mps.json cuda.json

A per-group digest that differs across backends is a FAIL: some op keeps a
subnormal on one backend and flushes it on another. After repop's consistent
FTZ (CUDA/CPU adopt Metal's flush) all three agree → PASS. Bytes are built on
CPU then moved to the device, so the seed is identical across backends.

The legs map to the documented cross-device contracts: repop kernel ops
(AdamW step + LSQ scale refresh) and the global clip on reachable-domain
grads must match on ALL backends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _seed_env_from_meta(meta_path: str | None) -> dict:
    if not meta_path:
        return {}
    meta = json.loads(Path(meta_path).read_text())
    for k, v in (meta.get("repop_env") or {}).items():
        os.environ[k] = str(v)
    return meta


def subnormal_fill(numel: int, off: int):
    """Deterministic fp32 values spanning small-normal → deepest subnormal.

    Element k gets ±2^(-(100 + (k+off) % 50)): exponents -100..-126 are normal
    fp32, -127..-149 are subnormal, so every tensor mixes normals that MUST be
    byte-preserved with subnormals that MUST flush identically. Built in fp64 on
    CPU then cast to fp32 (identical bytes on every backend when moved).
    """
    import torch

    idx = torch.arange(numel, dtype=torch.float64)
    off_l = float(off)
    exps = -(100.0 + ((idx + off_l) % 50.0))
    signs = torch.where(((idx.long() + off) % 2) == 0, 1.0, -1.0).double()
    return (signs * torch.pow(2.0, exps)).to(torch.float32)


def _hash_group(named_tensors) -> str:
    import torch

    h = hashlib.blake2b(digest_size=16)
    for name, t in sorted(named_tensors, key=lambda x: x[0]):
        if t is None:
            continue
        h.update(name.encode())
        h.update(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def run_once(device: str, config_name: str, meta_path: str | None, out_path: str | None) -> dict:
    import torch

    meta = _seed_env_from_meta(meta_path)
    os.environ.setdefault("REPOP_EXECUTION_MODE", "cross_device_reproducible")

    from pretrain.config import load_config, parse_config_resolved
    from pretrain.model import build_model
    from pretrain.model.init import init_weights
    from pretrain.optim.registry import build_optimizer
    from pretrain.parallel.parallel_dims import ParallelDims
    from pretrain.parallel.parallelize_llama3_repop import parallelize_llama3_repop

    cfg = (parse_config_resolved(meta["config_resolved"])
           if meta.get("config_resolved") else load_config(config_name))
    dev = torch.device(device)
    dp_shard = int(getattr(cfg.run, "dp_shard_size", 1) or 1)

    model = build_model(cfg.model, device=dev)
    model = parallelize_llama3_repop(model, cfg, ParallelDims(dp_replicate=1, dp_shard=1, world_size=1))
    init_weights(model, seed=cfg.run.seed)
    optimizer = build_optimizer(model, cfg.optim)

    # One real step to let the optimizer allocate its state (exp_avg/exp_avg_sq/step),
    # then overwrite the WHOLE hashed state with the subnormal-straddling seed.
    with torch.no_grad():
        for i, (n, p) in enumerate(model.named_parameters()):
            p.grad = torch.zeros_like(p) + 1e-3
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)

    # Seed integrity: the whole test is void if the seed stops exercising
    # subnormals (fp32 subnormal range: 0 < |v| < 1.1754944e-38).
    probe = subnormal_fill(1000, off=0)
    n_sub = int(((probe.abs() > 0) & (probe.abs() < 1.1754944e-38)).sum())
    assert n_sub >= 400, f"seed integrity: only {n_sub}/1000 subnormals — seed rotted"

    step_no = 1000  # late-step regime: bias correction + a decayed subnormal tail
    with torch.no_grad():
        for i, (n, p) in enumerate(model.named_parameters()):
            p.data.copy_(subnormal_fill(p.numel(), off=i).reshape(p.shape).to(dev))
            p.grad = subnormal_fill(p.numel(), off=i + 7).reshape(p.shape).to(dev)
            st = optimizer.state[p]
            st["exp_avg"].copy_(subnormal_fill(p.numel(), off=i + 13).reshape(p.shape).to(dev))
            st["exp_avg_sq"].copy_(subnormal_fill(p.numel(), off=i + 29).reshape(p.shape).to(dev).abs())
            if torch.is_tensor(st.get("step")):
                st["step"].fill_(step_no)

    # The real post-backward algorithmic ops the state hash covers, mapped to
    # the DOCUMENTED cross-device contracts (docs/audit-replay-usage.md):
    #
    # Leg 1 (ALL backends must match): repop kernel ops on fully-subnormal
    # state — the AdamW step (consistent-FTZ kernel) and the LSQ weight-scale
    # refresh (repop sum_dim reduction over subnormal-seeded weights).
    # Leg 2 runs FIRST (before state mutates): the clip path
    # (global_clip) on REACHABLE-domain grads — normals + exact zeros; grads
    # with subnormal ELEMENTS are outside the MPS contract by the documented
    # fold floor (|g| < ~1.08e-19), so the clip leg uses the domain real
    # training inhabits.
    from pretrain.train import global_clip
    from repop.qat.lsq import refresh_lsq_weight_scales

    # Leg 2: global clip on reachable-domain grads (save/restore around it so
    # leg 1 still sees the subnormal grad seed).
    _saved = [(p, p.grad.detach().clone()) for _n, p in model.named_parameters()]
    with torch.no_grad():
        for i, (_n, p) in enumerate(model.named_parameters()):
            # Reachable-domain seed from PURE INDEX MATH — no arithmetic on
            # subnormal values while constructing it (sign()/abs() on a
            # subnormal already diverge under MPS DAZ; the first version of
            # this leg found that out the hard way).
            # Index math in EXACT integer space (int64), magnitudes in fp64,
            # ONE fp32 rounding at the end. fp32 arange is integer-exact only
            # to 2^24; params above that (the 98M-element embeddings) turned
            # the %2/%3/%7 masks into torch-BUILD-dependent garbage — laptop
            # torch 2.10 and the pod's 2.11 nightly rounded differently, so
            # identical code hashed differently per environment and
            # masqueraded as a CUDA BFR break (2026-07-22). Same lesson as
            # subnormal_fill's fp64 construction above.
            idx = torch.arange(p.numel(), dtype=torch.int64).reshape(p.shape)
            sign = torch.where(((idx + i) % 2) == 0, 1.0, -1.0).double()
            mag = 1e-3 * (1.0 + 0.5 * ((idx + i) % 7).double() / 7.0)
            reach = torch.where(idx % 3 == 0, torch.zeros_like(mag), sign * mag)
            p.grad.copy_(reach.float().to(dev))
    # Input digest BEFORE the op under test: if this group differs across
    # backends/machines, the harness seed itself is not build-independent and
    # every downstream verdict is void (2026-07-22: fp32-arange seed made
    # torch 2.10 vs 2.11 hash differently and masqueraded as a CUDA BFR
    # break). Inputs-match + outputs-differ is the ONLY signature of a real
    # backend break.
    groups2 = {"globalclip_inputs": _hash_group([(n, p.grad) for n, p in model.named_parameters()])}
    global_clip.clip_audit(model, 1.0, dp_shard, dev)
    groups2["globalclip_grads"] = _hash_group([(n, p.grad) for n, p in model.named_parameters()])
    with torch.no_grad():
        for p, g in _saved:
            p.grad.copy_(g)

    # Leg 1: repop kernel ops.
    optimizer.step()
    refresh_lsq_weight_scales(model)

    groups = {
        "params": _hash_group([(n, p.data) for n, p in model.named_parameters()]),
        "exp_avg": _hash_group([(n, optimizer.state[p].get("exp_avg")) for n, p in model.named_parameters()]),
        "exp_avg_sq": _hash_group([(n, optimizer.state[p].get("exp_avg_sq")) for n, p in model.named_parameters()]),
        "weight_scales_refreshed": _hash_group(
            [(n, p.data) for n, p in model.named_parameters() if "weight_scale" in n]),
        **groups2,
    }

    res = {"device": device, "config": config_name, "groups": groups}
    print(json.dumps(res, indent=2))
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=2))
        print("wrote", out_path)
    return res


# Input-digest groups: a mismatch here means the HARNESS seed is not
# build/device-independent — the comparison is invalid and must be fixed in
# the test, not attributed to a backend.
INPUT_GROUPS = {"globalclip_inputs"}


def compare(paths: list[str]) -> int:
    runs = [json.loads(Path(p).read_text()) for p in paths]
    keys = sorted(runs[0]["groups"].keys())
    ok = True
    print(f"comparing {len(runs)} backends: {[r['device'] for r in runs]}")
    for k in keys:
        vals = {r["device"]: r["groups"].get(k) for r in runs}
        same = len(set(vals.values())) == 1
        if k in INPUT_GROUPS:
            if not same:
                ok = False
                print(f"  {k:24s} HARNESS-FAIL  {vals}")
                print("     ^ the TEST SEED differs across runs — the harness input is "
                      "not build/device-independent (fix the seed construction; every "
                      "downstream verdict from this comparison is VOID, do NOT attribute "
                      "to a backend).")
            else:
                print(f"  {k:24s} MATCH  [input digest — comparison valid]")
            continue
        ok &= same
        print(f"  {k:24s} {'MATCH' if same else 'DIFFER'}  {vals}")
    print()
    print("global clip + repop kernels: "
          + ("✅ PASS — BFR holds on subnormal-domain state" if ok
             else "❌ FAIL — BFR broken; a current-path op diverges"))
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--config-name", default="100m_smoke_repop")
    ap.add_argument("--meta", default=None, help="checkpoint meta.json to source config+repop_env (overrides --config-name)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="+", default=None, help="two+ result JSONs to diff")
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(args.compare))
    if not args.device:
        ap.error("--device required (or use --compare)")
    run_once(args.device, args.config_name, args.meta, args.out)


if __name__ == "__main__":
    main()
