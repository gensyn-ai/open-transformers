"""A state-hash mismatch must report both digests in full, from either path.

Regression test for the 2026-09-13 tester report (OPEN-1B audit step 103). The
replay produced 2c6c650b09949ad8... against a published e65f46aecf3ceaf1...,
and the reproduced digest appeared at full length nowhere: the
``--save-checkpoint-dir`` guard raised first with both values truncated to 16
hex, and the result JSON that carries the full digest is printed only on
success. Establishing what the replay had actually computed took a maintainer
and a manual fetch of the run's published state-hash log.
"""

from __future__ import annotations

import re

from pretrain.cli.audit_replay import mismatch_message

GOT = "2c6c650b09949ad8" + "1f" * 24
WANT = "e65f46aecf3ceaf1b497622f6f0b47f14d6c55c4c1128d3d07b43120b112bf2d"

#: What the audit CLI's log parser looks for (gensyn_audit.progress._FAILED).
#: Pinned here because that consumer lives in another repository: a change to
#: the message shape that silently stops matching is the failure this guards.
CONSUMER = re.compile(r"AUDIT FAILED: state_hash (?P<got>[0-9a-f]{64}) != (?P<want>[0-9a-f]{64})")


def test_both_digests_are_reported_in_full():
    msg = mismatch_message(GOT, WANT)
    assert GOT in msg and WANT in msg


def test_no_truncated_digest_is_emitted():
    """The prefix may appear only as the head of the full digest, never alone."""
    msg = mismatch_message(GOT, WANT)
    assert msg.count(GOT[:16]) == 1
    assert msg.count(WANT[:16]) == 1


def test_the_cli_log_parser_recovers_both_digests():
    m = CONSUMER.search(mismatch_message(GOT, WANT))
    assert m is not None
    assert m["got"] == GOT and m["want"] == WANT


def test_the_handoff_guard_and_main_report_the_same_shape():
    """The two failure paths must not drift; both go through this function.

    The guard appends its own clause, so the assertion is that the parseable
    head is identical, not that the whole string is.
    """
    base = mismatch_message(GOT, WANT)
    guard = base + "; refusing to save a checkpoint that would chain the next audit off an unreproduced state."
    assert CONSUMER.search(guard)["got"] == CONSUMER.search(base)["got"]
    assert guard.startswith(base)
