"""Distributed bootstrap + seeding + NCCL env.

NCCL env values live in ``scripts/env/nccl.env`` and are sourced by the
launcher; this module merely consumes whatever the env gave us. The
asserts here protect against silent misconfig (e.g. missing
``MASTER_ADDR`` because someone ran ``python`` instead of ``torchrun``).
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

LOG = logging.getLogger(__name__)


def init_distributed(backend: Optional[str] = None) -> int:
    """Initialise ``torch.distributed`` if torchrun set the env.

    Returns the local rank. If launched without torchrun (single GPU dev
    or CPU smoke test), runs in single-process mode and returns 0.
    """
    if "RANK" not in os.environ:
        # Not under torchrun. Single-process dev path.
        torch.distributed.init_process_group  # type: ignore[unused-ignore]
        return 0

    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if not torch.distributed.is_initialized():
        fqdn = os.environ.get("DIST_MASTER_FQDN")
        store_file = os.environ.get("TORCH_DIST_STORE_FILE")
        if fqdn:
            # Multi-pod, preferred: a TCPStore on pod-0's resolvable headless-
            # Service FQDN. The PVC FileStore (below) doesn't scale past ~4 pods
            # on this cluster's RWX Filestore — the default-PG NCCL unique-id
            # write (key '0') silently fails to propagate across pods, so the
            # first collective times out. A TCPStore has no PVC dependency.
            #
            # We pass the FQDN in a dedicated env var rather than reuse
            # MASTER_ADDR: torchrun rewrites the workers' MASTER_ADDR to the
            # rdzv master's gethostname() (the short, non-resolvable pod name)
            # after rendezvous — which is exactly why the FileStore workaround
            # existed. The c10d rendezvous already stands up a TCPStore at this
            # same FQDN successfully, so this is proven to work cross-pod. Use a
            # port distinct from the rdzv port (DIST_MASTER_STORE_PORT). rank 0
            # (always on pod 0) binds the server; the others connect.
            ws = int(os.environ["WORLD_SIZE"])
            r = int(os.environ["RANK"])
            port = int(os.environ.get("DIST_MASTER_STORE_PORT", "29501"))
            store = torch.distributed.TCPStore(
                fqdn, port, world_size=ws, is_master=(r == 0),
            )
            torch.distributed.init_process_group(
                backend=backend, store=store, rank=r, world_size=ws,
            )
        elif store_file:
            # Fallback: FileStore on the shared PVC. Avoids hostname-based
            # bootstrap when no resolvable FQDN is provided, but see the scaling
            # caveat above — prefer DIST_MASTER_FQDN for multi-pod runs.
            ws = int(os.environ["WORLD_SIZE"])
            r = int(os.environ["RANK"])
            store = torch.distributed.FileStore(store_file, ws)
            torch.distributed.init_process_group(
                backend=backend, store=store, rank=r, world_size=ws,
            )
        else:
            torch.distributed.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    LOG.info(
        "rank %s/%s (local %s) backend=%s",
        rank(), world_size(), local_rank, backend,
    )
    return local_rank


def world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    return rank() == 0


def set_seed(seed: int) -> None:
    """Seed the global RNG with a per-rank offset so any future runtime
    stochasticity (e.g. dropout) draws unique noise per rank. Model init
    does NOT use this — ``init_weights`` takes its own seed (no offset)
    so all ranks construct the same full model before FSDP sharding; see
    ``pretrain.model.init.init_weights``.

    Under either repop execution mode (``deterministic`` or
    ``cross_device_reproducible``) we additionally call
    ``repop.utils.set_determinism`` to flip PyTorch's global determinism
    toggles (cudnn.deterministic, no TF32, use_deterministic_algorithms).
    Repop's env var only steers repop's own kernel dispatch — it does not
    touch the torch flags, and ``nn.Embedding``'s grad scatter (still
    pytorch in our wiring) uses CUDA atomic adds otherwise. The reproducible
    mode needs these flags just as much as deterministic mode does:
    cross-device bitwise equality is meaningless if torch's own ops drift
    via TF32 / non-deterministic cuDNN / atomic-add reductions.
    Repop's seeding is then overridden by the per-rank seeding below.
    """
    if os.environ.get("REPOP_EXECUTION_MODE") in (
        "deterministic",
        "cross_device_reproducible",
    ):
        from repop.utils import set_determinism
        set_determinism(seed)
    r = rank()
    torch.manual_seed(seed + r)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + r)
    np.random.seed((seed + r) % (2**31))
    random.seed(seed + r)


def shutdown_distributed() -> None:
    """Graceful counterpart of ``init_distributed``.

    Called from the train loop's ``finally`` so the process group exits
    cleanly even on exception / Halt / Ctrl-C. Without this, NCCL's
    ProcessGroup destructor warns at interpreter exit ("WARNING:
    destroy_process_group() was not called before program exit") and on
    rare occasions leaks GPU memory.
    """
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    try:
        # Best-effort barrier so ranks agree on shutdown order; ignore
        # failures (a halted rank will not reach the barrier).
        torch.distributed.barrier()
    except Exception:
        pass
    torch.distributed.destroy_process_group()


def assert_flash_sdp_enabled() -> None:
    """Plan/06 §3: error out at startup rather than silently fall back."""
    if not torch.cuda.is_available():
        LOG.warning("CUDA unavailable — skipping FlashAttention assertion")
        return
    if not torch.backends.cuda.flash_sdp_enabled():
        raise RuntimeError(
            "FlashAttention SDPA backend not enabled. "
            "Check NGC container + torch.backends.cuda.flash_sdp_enabled()."
        )
