"""LSQ int8 QAT under FSDP2 — correctness on CPU + gloo.

Covers ``method="lsq"`` QAT under pure FSDP2 / HSDP
(``pretrain.parallel.parallelize_llama3_repop``):

  * ``test_lsq_weight_scale_init`` (no spawn): the learnable per-channel
    ``weight_scale`` survives ``init_weights`` and equals the LSQ-recommended
    ``2·mean(|w_row|)/√qmax`` of the *final* weights — the 1-D zero-pass would
    otherwise clobber it.
  * ``test_lsq_fsdp2_one_step`` (spawn world=2, dp_shard=2): one
    forward/backward/step with LSQ on; assert the
    ``weight_scale`` params are dim-0-sharded DTensors that receive non-zero
    gradients, and the optimizer step fires. Parametrised over fp32 / bf16 and
    with/without activation checkpointing.

CPU+gloo is a correctness check, not a perf one. SKIPs when ``repop`` is
not importable.
"""

from __future__ import annotations

import math
import os
import tempfile
import traceback
from pathlib import Path

import pytest

pytest.importorskip("repop")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch.distributed.tensor import DTensor, Shard  # noqa: E402

from pretrain.config import load_config  # noqa: E402
from pretrain.model import build_model  # noqa: E402
from pretrain.model.init import init_weights_seeded  # noqa: E402
from pretrain.optim.adamw_repop import FSDPAwareRepopAdamW  # noqa: E402
from pretrain.model.fused_loss import fused_ce_z_loss  # noqa: E402
from pretrain.parallel.parallel_dims import ParallelDims  # noqa: E402
from pretrain.parallel.parallelize_llama3_repop import (  # noqa: E402
    parallelize_llama3_repop,
)

# ``++`` force-adds: the smoke model config has no ``qat`` block (it relies on
# the schema default), so a plain override can't reach ``model.qat.*``.
_LSQ_OVERRIDES = [
    "++model.qat.enabled=true",
    "++model.qat.method=lsq",
    "++model.qat.weight_bits=8",
    "++model.qat.act_bits=8",
]


# ---------------------------------------------------------------------------
# Unit: weight_scale init survives + matches the LSQ formula (no distribution).
# ---------------------------------------------------------------------------
def test_lsq_weight_scale_init():
    from repop.qat.functional import _qmax_for_bits
    from repop.qat.lsq import LSQQuantizedLinear

    cfg = load_config("100m_smoke_repop", overrides=_LSQ_OVERRIDES)
    model = build_model(cfg.model, device=torch.device("cpu"))
    init_weights_seeded(model, seed=cfg.run.seed)

    lsq_layers = [m for m in model.modules() if isinstance(m, LSQQuantizedLinear)]
    assert lsq_layers, "expected LSQQuantizedLinear layers with qat.method=lsq"

    for m in lsq_layers:
        ws = m.weight_scale.detach()
        # The 1-D zero-pass in init_weights would leave this all-zero.
        assert torch.all(ws > 0), "weight_scale must be strictly positive after init"
        w_qmax = _qmax_for_bits(m.weight_bits)
        expected = (
            2.0 * m.weight.detach().abs().mean(dim=1) / math.sqrt(w_qmax)
        ).clamp_min(1e-6)
        torch.testing.assert_close(ws, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Distributed workers.
# ---------------------------------------------------------------------------
def _worker(rank, world_size, port, status_path, scenario):  # noqa: ANN001
    """One spawned process. Writes ``OK`` (or a traceback) to ``status_path``."""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["REPOP_USE_CPU_BACKEND"] = "1"

        dist.init_process_group(backend="gloo")
        _run_fsdp_step(rank, world_size, scenario)
        dist.destroy_process_group()

        Path(status_path).write_text(f"rank{rank}:OK\n")
    except Exception:
        Path(status_path).write_text(f"rank{rank}:FAIL\n{traceback.format_exc()}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise


def _build_cfg(scenario):  # noqa: ANN001
    overrides = list(_LSQ_OVERRIDES) + [
        f"++run.mixed_precision={'true' if scenario['mixed_precision'] else 'false'}",
        f"++run.activation_checkpoint={'true' if scenario['ac'] else 'false'}",
    ]
    return load_config("100m_smoke_repop", overrides=overrides)


def _run_fsdp_step(rank, world_size, scenario):  # noqa: ANN001
    """Pure FSDP: one fwd/bwd/step with LSQ on."""
    cfg = _build_cfg(scenario)
    device = torch.device("cpu")

    model = build_model(cfg.model, device=device)
    init_weights_seeded(model, seed=cfg.run.seed)

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=world_size, world_size=world_size,
    )
    parallel_dims.build_mesh(device_type="cpu")

    # Must not raise now that LSQ is FSDP-safe.
    model = parallelize_llama3_repop(model, cfg, parallel_dims)

    # weight_scale must be a dim-0-sharded DTensor (same axis as the weight).
    scale_params = [
        (n, p) for n, p in model.named_parameters() if n.endswith("weight_scale")
    ]
    assert scale_params, f"rank{rank}: no weight_scale params found under FSDP"
    for n, p in scale_params:
        assert isinstance(p.data, DTensor), f"rank{rank}: {n} not a DTensor"
        assert p.data.placements == (Shard(0),), (
            f"rank{rank}: {n} placements {p.data.placements} != (Shard(0),)"
        )

    opt = FSDPAwareRepopAdamW(
        model.parameters(), lr=1e-3, betas=cfg.optim.betas, eps=cfg.optim.eps,
    )

    B, T, V = 2, 32, cfg.model.vocab_size
    torch.manual_seed(rank)  # every rank is its own dp rank
    input_ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))

    out = model(input_ids)
    z_coeff = cfg.model.z_loss.coeff if cfg.model.z_loss.enabled else 0.0
    ce, zloss = fused_ce_z_loss(out.logits, labels, z_coeff)
    loss = ce + zloss
    assert torch.isfinite(loss), f"rank{rank}: loss not finite ({loss.item()})"

    loss.backward()

    # The learnable scale must actually receive a gradient — otherwise LSQ is
    # silently not learning its step size under FSDP.
    scale_grad_nonzero = False
    for n, p in scale_params:
        assert p.grad is not None, f"rank{rank}: {n} got no grad"
        g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
        assert torch.isfinite(g).all(), f"rank{rank}: {n} grad NaN/Inf"
        if g.abs().sum().item() > 0:
            scale_grad_nonzero = True
    assert scale_grad_nonzero, f"rank{rank}: all weight_scale grads zero"

    sample = next(
        p for p in model.parameters() if p.grad is not None and hasattr(p, "to_local")
    )
    before = sample.to_local().clone().detach()
    opt.step()
    delta = (sample.to_local().detach() - before).abs().sum().item()
    assert delta > 0, f"rank{rank}: optimizer step produced no change"


# ---------------------------------------------------------------------------
# Spawn harness.
# ---------------------------------------------------------------------------
def _spawn_and_collect(world_size, scenario):  # noqa: ANN001
    with tempfile.TemporaryDirectory() as tmpdir:
        status_paths = [
            os.path.join(tmpdir, f"status_{r}.txt") for r in range(world_size)
        ]
        port = 29500 + (os.getpid() % 1000)
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_worker,
                args=(rank, world_size, port, status_paths[rank], scenario),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=300)
            if p.is_alive():
                p.terminate()
                raise RuntimeError("worker timed out after 300s")
        return [
            Path(p).read_text() if Path(p).exists() else "rank?:NO_OUTPUT"
            for p in status_paths
        ]


def _assert_all_ok(statuses):
    failures = [s for s in statuses if "OK" not in s.splitlines()[0]]
    if failures:
        pytest.fail("one or more ranks failed:\n" + "\n---\n".join(failures))


@pytest.mark.parametrize(
    "mixed_precision,ac",
    [(False, False), (True, True)],
    ids=["fp32", "bf16+ac"],
)
def test_lsq_fsdp2_one_step(mixed_precision, ac):
    """world=2 pure FSDP, LSQ on; one fwd/bwd/step succeeds with a sharded,
    learning weight_scale."""
    scenario = {
        "mixed_precision": mixed_precision,
        "ac": ac,
    }
    _assert_all_ok(_spawn_and_collect(world_size=2, scenario=scenario))
