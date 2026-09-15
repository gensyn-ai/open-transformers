"""Thin wrapper around ``wandb.init`` so the rest of the code only depends
on a tiny interface (``log`` / ``finish``).

If ``wandb`` is unavailable or ``WANDB_MODE=disabled``, the wrapper turns
into a no-op.
"""

from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger(__name__)


class _NullRun:
    def log(self, *a, **kw) -> None:
        pass

    def finish(self) -> None:
        pass


class WandBRun:
    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict[str, Any] | None = None,
        run_dir: str | None = None,
    ) -> None:
        self._impl: Any = _NullRun()
        try:
            import os

            if os.environ.get("WANDB_MODE", "").lower() == "disabled":
                LOG.info("wandb disabled by WANDB_MODE")
                return
            import wandb

            self._impl = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                dir=run_dir,
                reinit=True,
            )
        except Exception as e:
            LOG.warning("wandb init failed (%s); proceeding without it", e)

    def log(self, *a, **kw) -> None:
        self._impl.log(*a, **kw)

    def finish(self) -> None:
        try:
            self._impl.finish()
        except Exception:
            pass
