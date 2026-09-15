"""``parse_config_resolved`` — re-loading a checkpoint's stored config.

Checkpoints written before a schema field is retired keep serializing the old
block in ``config_resolved``; the schema forbids unknown keys, so a plain
``model_validate_json`` refuses them. ``parse_config_resolved`` drops the
retired keys (``_RETIRED_CONFIG_KEYS``) so those checkpoints — e.g. the whole
20260722 v2 run, which clipped with ``clip_algo="global"`` but serialized the
inert ``train.adagc`` block until the field was deleted — stay auditable.
"""

from __future__ import annotations

import json

import pydantic
import pytest

from pretrain.config import load_config, parse_config_resolved
from pretrain.config.schema import RootConfig


def _resolved_with_stale_adagc() -> str:
    """A config_resolved as pre-removal code serialized it: current schema
    plus the retired ``train.adagc`` block."""
    obj = json.loads(load_config("100m_smoke_repop").model_dump_json())
    obj["train"]["adagc"] = {
        "lambda_rel": 1.04,
        "lambda_abs": 1.0,
        "beta": 0.99,
        "t_start": 100,
        "eps": 1e-6,
        "gamma_min": 1e-12,
    }
    return json.dumps(obj)


def test_stale_adagc_block_is_dropped():
    blob = _resolved_with_stale_adagc()
    # The plain path refuses it (extra="forbid") — the compat helper is load-bearing.
    with pytest.raises(pydantic.ValidationError, match="train.adagc"):
        RootConfig.model_validate_json(blob)
    cfg = parse_config_resolved(blob)
    assert not hasattr(cfg.train, "adagc")


def test_current_config_roundtrips_unchanged():
    blob = json.loads(_resolved_with_stale_adagc())
    del blob["train"]["adagc"]
    cfg = parse_config_resolved(json.dumps(blob))
    assert cfg == RootConfig.model_validate(blob)


def test_unknown_nonretired_key_still_fails():
    """Only listed retired keys are tolerated — a genuine typo must still fail."""
    blob = json.loads(_resolved_with_stale_adagc())
    blob["train"]["grad_cilp"] = 2.0
    with pytest.raises(pydantic.ValidationError, match="grad_cilp"):
        parse_config_resolved(json.dumps(blob))


def test_checkpoint_meta_from_dict_drops_retired_keys():
    """A live-run meta.json written by pre-removal code carries the retired
    adagc_* fields; ``from_dict`` must load it, while a genuinely unknown key
    still fails loudly."""
    from pretrain.train.checkpoint import CheckpointMeta

    base = dict(
        consumed_tokens=1,
        step=1,
        git_sha="x",
        config_resolved="{}",
        tokenizer_hash="t",
        container_digest="c",
    )
    stale = dict(
        base,
        adagc_lambda_rel=1.04,
        adagc_lambda_abs=1.0,
        adagc_beta=0.99,
        adagc_t_start=100,
        adagc_eps=1e-6,
        adagc_gamma_min=1e-12,
    )
    meta = CheckpointMeta.from_dict(stale)
    assert meta.step == 1 and not hasattr(meta, "adagc_lambda_rel")
    with pytest.raises(TypeError, match="not_a_field"):
        CheckpointMeta.from_dict(dict(base, not_a_field=1))
