"""Guard against cold-starting on top of a prior run's checkpoints.

Regression coverage for the run_id-collision overwrite: a launch that reused a
run_id (pinned RUN_ID, or a pod recreation reading a stale value from the RWX
PVC) and cold-started at step 0, letting DCP overwrite step_*/dcp in place.
``Checkpointer.has_checkpoints`` is what the train loop consults to refuse that
cold start when ``--resume-from`` is absent.
"""

from __future__ import annotations

from pretrain.train.checkpoint import Checkpointer


def test_has_checkpoints_empty(tmp_path):
    ckpt = Checkpointer(tmp_path / "checkpoints")
    assert ckpt.has_checkpoints() is False


def test_has_checkpoints_detects_complete_and_partial(tmp_path):
    ckpt = Checkpointer(tmp_path / "checkpoints")
    # A partial save (no _COMPLETE sentinel) must still block a cold start —
    # it's a prior run's directory and DCP would overwrite its dcp shards.
    (ckpt.root / "step_000000010").mkdir()
    assert ckpt.has_checkpoints() is True
    # latest() ignores it (no sentinel); has_checkpoints() does not.
    assert ckpt.latest() is None
    (ckpt.root / "step_000000010" / "_COMPLETE").write_text("")
    assert ckpt.has_checkpoints() is True


def test_has_checkpoints_ignores_unrelated_files(tmp_path):
    ckpt = Checkpointer(tmp_path / "checkpoints")
    (ckpt.root / "logs").mkdir()
    (ckpt.root / "notes.txt").write_text("hi")
    assert ckpt.has_checkpoints() is False
