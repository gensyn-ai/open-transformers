"""The audit-kit provenance gate: a replay must name a repop build that can
actually serve the requested device.

``_require_repop_backend`` is the pure gate underneath ``audit_replay``'s
device resolution: on mps it demands the Metal backend be compiled into the
installed repop, on cuda the CUDA backend — a build missing them would fall
back to CPU kernels op by op and "pass" without exercising the hardware the
verification claims to cover. CPU is always allowed: it is the reference.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("repop")

from pretrain.cli.audit_replay import _require_repop_backend  # noqa: E402

_CPU_ONLY = {"commit": "abc123def456", "backends": ["cpu"], "cuda_arch_list": ""}
_MACOS_WHEEL = {"commit": "abc123def456", "backends": ["cpu", "metal"], "cuda_arch_list": ""}
_LINUX_WHEEL = {"commit": "abc123def456", "backends": ["cpu", "cuda"], "cuda_arch_list": "8.0;9.0"}
_UNSTAMPED = {"commit": "unknown", "backends": [], "cuda_arch_list": "unknown"}


def test_mps_rejects_metal_less_build():
    with pytest.raises(RuntimeError, match="no Metal backend"):
        _require_repop_backend(torch.device("mps"), _CPU_ONLY)


def test_mps_rejects_unstamped_build():
    # An unstamped source tree cannot prove which backends it carries, and an
    # audit result must be traceable — refuse rather than assume.
    with pytest.raises(RuntimeError, match="no Metal backend"):
        _require_repop_backend(torch.device("mps"), _UNSTAMPED)


def test_mps_accepts_macos_wheel():
    _require_repop_backend(torch.device("mps"), _MACOS_WHEEL)


def test_cuda_rejects_cuda_less_build():
    with pytest.raises(RuntimeError, match="no CUDA backend"):
        _require_repop_backend(torch.device("cuda"), _MACOS_WHEEL)


def test_cuda_accepts_linux_wheel():
    _require_repop_backend(torch.device("cuda"), _LINUX_WHEEL)


def test_cpu_always_allowed():
    for info in (_CPU_ONLY, _MACOS_WHEEL, _LINUX_WHEEL, _UNSTAMPED):
        _require_repop_backend(torch.device("cpu"), info)


def test_error_names_the_build():
    # The refusal must carry the provenance it refused on, so the failure is
    # actionable without re-running under a debugger.
    with pytest.raises(RuntimeError, match="abc123def456"):
        _require_repop_backend(torch.device("mps"), _CPU_ONLY)
