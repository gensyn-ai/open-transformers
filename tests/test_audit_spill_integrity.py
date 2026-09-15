"""A spilled tensor that changes on disk must stop the replay, not skew its hash.

The offload paths stream the fold partials, the AdamW moments and the fp32
master through ``torch.save``/``torch.load``. ``torch.load`` does not checksum,
so a flipped bit comes back as a valid float and the replay carries on to
produce a wrong state hash -- reported as NO MATCH against a step that is
actually correct, which is indistinguishable from a real reproducibility
failure and is the worse of the two outcomes for an audit.

The 2026-09-13 OPEN-1B step-103 report is the case this guards: a 24 GB Mac
against a ~43 GB working set returned two different wrong hashes on two runs,
while the same step matched on a 48 GB machine. Losses were bit-identical to an
H100 replay at every step, so the forward agreed and the divergence sat
downstream of it -- where these spills are.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from pretrain.cli.audit_replay import (
    SpillCorruption,
    _digest_payload,
    _load_verified,
    _save_verified,
    _unlink_verified,
)
from pretrain.train.state_hash import feed_tensor, tensor_bytes


def _flip_one_bit_in_payload(path, sample: torch.Tensor) -> None:
    """Flip a single bit inside the raw tensor payload of a saved container.

    Locates the payload by its leading bytes rather than assuming an offset, so
    the test does not encode the zip layout of any one torch version.
    """
    raw = path.read_bytes()
    needle = sample.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()[:64]
    off = raw.find(needle)
    assert off > 0, "could not locate the tensor payload in the saved container"
    buf = bytearray(raw)
    buf[off + 17] ^= 0x01
    path.write_bytes(bytes(buf))


@pytest.fixture
def payload():
    return {
        "exp_avg": torch.arange(512, dtype=torch.float32),
        "exp_avg_sq": torch.arange(512, dtype=torch.float32) * 2.0,
    }


def test_clean_round_trip_returns_identical_bytes(tmp_path, payload):
    path = tmp_path / "opt_0.pt"
    _save_verified(payload, path)
    got = _load_verified(path, "cpu")
    for k in payload:
        assert torch.equal(
            got[k].view(torch.uint8), payload[k].view(torch.uint8)
        ), f"{k} did not survive a clean round-trip"


def test_flipped_bit_raises_instead_of_returning_it(tmp_path, payload):
    """The control this whole file exists for.

    Without verification this same flip loads clean and returns a value one ULP
    off, which is exactly how a correct step gets reported as NO MATCH.
    """
    path = tmp_path / "opt_0.pt"
    _save_verified(payload, path)
    _flip_one_bit_in_payload(path, payload["exp_avg"])

    # Unverified, torch hands the corruption back without complaint -- this is
    # the behaviour being fixed, asserted so the premise cannot rot.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert not torch.equal(raw["exp_avg"], payload["exp_avg"]), (
        "torch.load began rejecting corrupt payloads; the sidecar may be redundant"
    )

    with pytest.raises(SpillCorruption) as excinfo:
        _load_verified(path, "cpu")
    assert path.name in str(excinfo.value)


def test_master_tensor_round_trip_and_corruption(tmp_path):
    """The master spill stores a bare tensor, not a mapping."""
    master = torch.arange(256, dtype=torch.float32)
    path = tmp_path / "master_0.pt"
    _save_verified(master, path)
    assert torch.equal(_load_verified(path, "cpu"), master)

    _flip_one_bit_in_payload(path, master)
    with pytest.raises(SpillCorruption):
        _load_verified(path, "cpu")


def test_digest_covers_dtype_and_shape_not_only_bytes():
    """Same bits under different metadata must not collide.

    A payload that survives with the right bytes but the wrong dtype or shape
    is still a corrupted payload.
    """
    base = torch.arange(8, dtype=torch.int32)
    assert _digest_payload(base) != _digest_payload(base.view(torch.float32))
    assert _digest_payload(base) != _digest_payload(base.reshape(2, 4))


def test_digest_handles_dtypes_neither_spelling_covers_alone():
    """bfloat16 has no ``numpy()``; a 0-dim tensor has no ``view(uint8)``."""
    for t in (
        torch.randn(4, dtype=torch.bfloat16),
        torch.zeros((), dtype=torch.int64),
        torch.randn(0),
    ):
        assert len(_digest_payload(t)) == 64


def test_absent_optional_moment_is_distinguished_from_present(tmp_path):
    """``max_exp_avg_sq`` is absent unless amsgrad; that must change the digest."""
    without = {"exp_avg": torch.ones(4)}
    with_none = {"exp_avg": torch.ones(4), "max_exp_avg_sq": None}
    assert _digest_payload(without) != _digest_payload(with_none)

    path = tmp_path / "opt_1.pt"
    _save_verified(with_none, path)
    got = _load_verified(path, "cpu")
    assert got["max_exp_avg_sq"] is None


def test_missing_sidecar_loads_unchecked(tmp_path, payload):
    """A spill written before verification existed must not break a live run."""
    path = tmp_path / "fold_0.pt"
    _save_verified(payload, path)
    path.with_suffix(path.suffix + ".blake2b").unlink()
    got = _load_verified(path, "cpu")
    assert torch.equal(got["exp_avg"], payload["exp_avg"])


def test_disable_switch_skips_both_halves(tmp_path, payload, monkeypatch):
    monkeypatch.setenv("PRETRAIN_AUDIT_SPILL_VERIFY", "0")
    path = tmp_path / "fold_1.pt"
    _save_verified(payload, path)
    assert not path.with_suffix(path.suffix + ".blake2b").exists()
    _flip_one_bit_in_payload(path, payload["exp_avg"])
    _load_verified(path, "cpu")  # must not raise


def test_unlink_removes_the_sidecar_too(tmp_path, payload):
    """The fold unlinks each partial as it is consumed; sidecars must not pile up."""
    path = tmp_path / "fold_2.pt"
    _save_verified(payload, path)
    sidecar = path.with_suffix(path.suffix + ".blake2b")
    assert sidecar.exists()
    _unlink_verified(path)
    assert not path.exists() and not sidecar.exists()


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.int8, torch.bool],
)
@pytest.mark.parametrize("layout", ["contiguous", "scalar", "empty", "transpose"])
def test_payload_bytes_are_a_view_not_a_copy(dtype, layout):
    """The digest must not transiently duplicate the payload.

    A ``bytes`` copy here would cost a second full allocation of the largest
    tensor on every save and every load, on exactly the machines that are
    already swapping. This is the shared helper, so the property is asserted
    against the spellings the spill payloads actually take.
    """
    t = torch.arange(12).to(dtype).reshape(3, 4)
    if layout == "scalar":
        t = t[0, 0]
    elif layout == "empty":
        t = t[:0]
    elif layout == "transpose":
        t = t.t()

    c = t.detach().to("cpu").contiguous()
    buf = tensor_bytes(c)
    assert isinstance(buf, memoryview)
    assert bytes(buf) == c.reshape(-1).view(torch.uint8).numpy().tobytes()
    if layout == "contiguous":
        assert buf.obj.__array_interface__["data"][0] == c.data_ptr()


def test_digest_is_blake2b_32_over_the_shared_tensor_walk():
    """Pins the algorithm, the digest width, and the byte stream.

    Width first: the sidecar is 64 hex characters either way, so a switch of
    algorithm cannot be caught by looking at one. The pinned value is what
    blake2b-32 over ``state_hash.feed_tensor`` produces, so a future edit that
    re-forks the tensor walk, or quietly reverts to sha256, fails here rather
    than at some auditor's replay.
    """
    payload = {
        "exp_avg": torch.arange(8, dtype=torch.float32),
        "exp_avg_sq": torch.arange(8, dtype=torch.bfloat16),
        "step": torch.tensor(3, dtype=torch.int64),
    }
    digest = _digest_payload(payload)
    assert len(digest) == 64
    assert digest == (
        "e5c4e063474d164d811d27dbbb6d549faa2c833e035465a9c041e99a2e12025e"
    )

    expected = hashlib.blake2b(digest_size=32)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        feed_tensor(expected, key.encode() + b"\0", payload[key])
    assert digest == expected.hexdigest()
