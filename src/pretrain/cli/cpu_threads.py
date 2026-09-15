"""Startup policy for CPU audit workers, outside the hashed model state."""

from __future__ import annotations

import os
import platform
import subprocess

import torch


def parse_cpu_threads(value: str) -> str:
    if value == "auto":
        return value
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError("CPU threads must be 'auto' or a positive integer") from exc
    if count < 1:
        raise ValueError("CPU threads must be 'auto' or a positive integer")
    return str(count)


def _performance_cores() -> int | None:
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        return None
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.perflevel0.physicalcpu"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )
        count = int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return min(count, os.cpu_count() or count) if count > 0 else None


def configure_cpu_threads(request: str = "auto") -> dict[str, str | int | None]:
    """Respect environment overrides; otherwise prefer Apple performance cores.

    Other hosts retain their Torch default. This selects a starting count,
    not an affinity mask or a benchmark-derived optimum.
    """
    request = parse_cpu_threads(request)
    count = None
    if request != "auto":
        count = int(request)
        source = "explicit"
    elif any(os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        source = "environment"
    else:
        count = _performance_cores()
        source = "apple-performance-cores" if count is not None else "runtime-default"
    if count is not None:
        # Set both startup hints and Torch's already-initialized intra-op pool.
        os.environ["OMP_NUM_THREADS"] = str(count)
        os.environ["MKL_NUM_THREADS"] = str(count)
        torch.set_num_threads(count)
    return {
        "requested": request,
        "source": source,
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
