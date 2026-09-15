"""Load Hydra YAML → `RootConfig` (Pydantic-validated).

Hydra handles composition + CLI overrides; Pydantic validates the result.
We deliberately do not use Hydra's structured-configs binding because
maintaining a parallel ``@dataclass`` hierarchy doubled the surface area
without adding type safety beyond what Pydantic gives us.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from pretrain.config.schema import RootConfig


def _default_configs_dir() -> Path:
    here = Path(__file__).resolve()
    # walk up until we find a sibling configs/ directory
    for ancestor in [here] + list(here.parents):
        candidate = ancestor.parent / "configs"
        if candidate.is_dir() and (candidate / "train").is_dir():
            return candidate
    # fall back to repo root if PRETRAIN_CONFIGS not set
    env = os.environ.get("PRETRAIN_CONFIGS")
    if env:
        return Path(env)
    # Installed-wheel layout: the wheel ships the repo's configs/ tree as
    # pretrain/_configs (pyproject force-include), so an audit-kit install can
    # resolve --config-name with no source checkout. Last in the order on
    # purpose: a repo checkout and an explicit PRETRAIN_CONFIGS both name a
    # configs tree that can be newer than the one baked into the wheel.
    bundled = here.parents[1] / "_configs"
    if bundled.is_dir() and (bundled / "train").is_dir():
        return bundled
    raise FileNotFoundError(
        "Could not locate configs/ directory. Set PRETRAIN_CONFIGS env var."
    )


def load_config(
    config_name: str,
    overrides: list[str] | None = None,
    config_dir: str | os.PathLike[str] | None = None,
) -> RootConfig:
    """Compose a Hydra config and validate against the schema.

    `overrides` are Hydra-style ``key=value`` strings, e.g.
    ``["optim.peak_lr=3e-4", "train.total_tokens=300_000_000_000"]``.
    """

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    cdir = Path(config_dir) if config_dir else _default_configs_dir()
    if not cdir.is_dir():
        raise FileNotFoundError(f"configs dir not found: {cdir}")

    # initialize_config_dir is reentrant only if we clear first.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(cdir.resolve()), version_base=None):
        cfg = compose(
            config_name=f"train/{config_name}",
            overrides=overrides or [],
        )

    return resolve_to_typed(cfg)


def resolve_to_typed(cfg: Any) -> RootConfig:
    """Turn an OmegaConf DictConfig (or plain dict) into a `RootConfig`.

    Pydantic does the real validation. OmegaConf's interpolations are
    resolved before handoff so the model only sees concrete values.
    """
    if hasattr(cfg, "_content") or hasattr(cfg, "to_container"):
        plain = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    else:
        plain = cfg
    return RootConfig.model_validate(plain)


# Config keys that were removed from the schema but appear in the
# ``config_resolved`` of checkpoints written by pre-removal code. Each entry
# is a path into the resolved-config dict; the key is dropped before
# validation (the schema forbids unknown keys). Only keys the recorded run
# never acted on belong here — anything that shaped the compute must instead
# fail the parse.
_RETIRED_CONFIG_KEYS = (
    # AdaGC hyperparameters. Inert since the clipper was retired (2026-07-21,
    # clip_algo="global" from then on); the block kept being serialized until
    # the schema field was deleted (2026-08-04). Runs that actually CLIPPED
    # with AdaGC are refused by audit_replay's clip_algo gate, not here.
    ("train", "adagc"),
)


def parse_config_resolved(config_resolved: str) -> RootConfig:
    """Parse a checkpoint meta's stored ``config_resolved`` back into a
    :class:`RootConfig`, dropping retired keys (``_RETIRED_CONFIG_KEYS``) that
    older code serialized but the schema no longer knows."""
    import json

    obj = json.loads(config_resolved)
    for *path, key in _RETIRED_CONFIG_KEYS:
        node = obj
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(key, None)
    return RootConfig.model_validate(obj)
