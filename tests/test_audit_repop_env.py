"""``audit_replay`` must not apply arbitrary env vars from a checkpoint meta.

A checkpoint's ``meta.json`` is untrusted input (the audit is designed to load
checkpoints produced by someone else), and its ``repop_env`` used to be applied
to ``os.environ`` unrestricted — letting a doctored meta set ``LD_*``,
``*_PROXY``, ``GOOGLE_APPLICATION_CREDENTIALS`` (before the ``--gcs-root``
fetch runs), allocator knobs, etc. ``_apply_repop_env`` allowlists exactly
what ``loop._capture_repop_env`` can record: ``REPOP*``-prefixed vars plus the
cuBLAS/arch contract keys.

Deliberately repop-free (unit tests on the helper) so this runs on any dev box.
"""

from __future__ import annotations

import os

import pytest

from pretrain.cli.audit_replay import _apply_repop_env


def test_allowed_vars_are_applied(monkeypatch):
    for k in ("REPOP_EXECUTION_MODE", "CUBLAS_WORKSPACE_CONFIG", "REPOPX"):
        monkeypatch.delenv(k, raising=False)
    _apply_repop_env(
        {
            "REPOP_EXECUTION_MODE": "cross_device_reproducible",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "REPOPX": 1,  # REPOP* prefix (no underscore), scalar non-str value
        }
    )
    assert os.environ["REPOP_EXECUTION_MODE"] == "cross_device_reproducible"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["REPOPX"] == "1"
    monkeypatch.delenv("REPOPX", raising=False)


def test_none_is_a_noop():
    _apply_repop_env(None)


@pytest.mark.parametrize(
    "key",
    ["LD_PRELOAD", "HTTPS_PROXY", "GOOGLE_APPLICATION_CREDENTIALS", "PATH", ""],
)
def test_disallowed_var_is_rejected(monkeypatch, key):
    monkeypatch.delenv(key, raising=False) if key else None
    with pytest.raises(ValueError, match="disallowed variable"):
        _apply_repop_env({key: "evil"})
    if key:
        assert os.environ.get(key) != "evil"


def test_rejection_applies_nothing(monkeypatch):
    """Validation is all-or-nothing: a bad key later in the dict must not
    leave earlier (benign) keys applied."""
    monkeypatch.delenv("REPOP_BENIGN_FIRST", raising=False)
    with pytest.raises(ValueError):
        _apply_repop_env({"REPOP_BENIGN_FIRST": "1", "LD_PRELOAD": "evil"})
    assert "REPOP_BENIGN_FIRST" not in os.environ


def test_non_dict_is_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        _apply_repop_env(["REPOP_EXECUTION_MODE=fast"])


def test_non_scalar_value_is_rejected():
    with pytest.raises(ValueError, match="must be a scalar"):
        _apply_repop_env({"REPOP_EXECUTION_MODE": {"nested": "dict"}})


def test_non_string_key_is_rejected():
    with pytest.raises(ValueError, match="disallowed variable"):
        _apply_repop_env({1: "x"})


@pytest.mark.parametrize(
    "key",
    [
        "REPOP_METAL_SHADER_DIR",   # the concrete path-redirect repop honors
        "REPOP_KERNEL_PATH",
        "REPOP_CACHE_ROOT",
        "REPOP_PLUGIN_SO",
        "REPOP_EXTRA_LIB",
        "REPOP_WEIGHTS_FILE",
    ],
)
def test_repop_redirect_keys_are_rejected(monkeypatch, key):
    """A REPOP*-prefixed key that denotes a filesystem/loader redirect must be
    refused even though it passes the prefix + scalar rules — it would point
    repop at an attacker-controlled path."""
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="redirect"):
        _apply_repop_env({key: "/tmp/evil"})
    assert os.environ.get(key) != "/tmp/evil"


def test_scalar_knobs_near_redirect_words_still_pass(monkeypatch):
    """The redirect filter keys on path SHAPE, not any substring — genuine
    scalar knobs must not be caught. (Guards against an over-broad filter.)"""
    for k in ("REPOP_LSQ_STE_BWD_INT8", "REPOP_USE_HFMA2_MMACC"):
        monkeypatch.delenv(k, raising=False)
    _apply_repop_env({"REPOP_LSQ_STE_BWD_INT8": "1", "REPOP_USE_HFMA2_MMACC": "0"})
    assert os.environ["REPOP_LSQ_STE_BWD_INT8"] == "1"
    monkeypatch.delenv("REPOP_LSQ_STE_BWD_INT8", raising=False)
    monkeypatch.delenv("REPOP_USE_HFMA2_MMACC", raising=False)


def test_contract_keys_match_the_writer():
    """The hand-copied contract keys must equal loop._CONTRACT_ENV_KEYS, so the
    validator (which cannot import repop-backed loop at apply time without
    breaking the env-ordering contract) never drifts from the writer. Imported
    here in the test, where the ordering constraint does not apply."""
    pytest.importorskip("repop")
    from pretrain.cli.audit_replay import _REPOP_ENV_CONTRACT_KEYS
    from pretrain.train.loop import _CONTRACT_ENV_KEYS

    assert tuple(_REPOP_ENV_CONTRACT_KEYS) == tuple(_CONTRACT_ENV_KEYS)
