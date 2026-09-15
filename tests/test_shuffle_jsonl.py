"""Raw-level global shuffle for multi-component sources (scripts/build_corpus.py).

Guards the fix for the component-blocking artifact: a block-ordered raw
JSONL sharded in order + the sampler's shard-granular permutation put
shard-sized homogeneous stretches into the training stream (the loss_ce
square wave on the first midtrain anneal). ``shuffle_jsonl`` must produce a
deterministic uniform permutation of the raw lines without ever corrupting
or losing a document.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_corpus.py"


@pytest.fixture(scope="module")
def bc():
    spec = importlib.util.spec_from_file_location("build_corpus", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Must be registered before exec: the module's @dataclass resolution
    # looks itself up in sys.modules.
    sys.modules["build_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_blocked_jsonl(path: Path, components: dict[str, int]) -> list[str]:
    """Component-block-ordered file, like a real multi-component pull."""
    lines = []
    with path.open("w", encoding="utf-8") as f:
        for comp, n in components.items():
            for i in range(n):
                line = json.dumps({"text": f"{comp} doc {i}"}) + "\n"
                f.write(line)
                lines.append(line)
    return lines


COMPONENTS = {"tinygsm": 800, "mathcoder": 150, "gsm8k": 50}


def test_preserves_multiset_and_is_deterministic(bc, tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    original = _write_blocked_jsonl(a, COMPONENTS)
    _write_blocked_jsonl(b, COMPONENTS)

    bc.shuffle_jsonl(a, seed=0, num_buckets=8)
    bc.shuffle_jsonl(b, seed=0, num_buckets=8)

    out_a = a.read_text().splitlines(keepends=True)
    assert collections.Counter(out_a) == collections.Counter(original)
    assert out_a != original  # actually permuted
    assert out_a == b.read_text().splitlines(keepends=True)  # same seed, same order

    c = tmp_path / "c.jsonl"
    _write_blocked_jsonl(c, COMPONENTS)
    bc.shuffle_jsonl(c, seed=1, num_buckets=8)
    assert c.read_text().splitlines(keepends=True) != out_a  # seed changes order


def test_deblocks_component_runs(bc, tmp_path):
    # The property the training stream needs: no long single-component runs.
    # With 800/150/50 blocked input, the max run of the dominant component
    # in a uniform shuffle is ~tens at worst; the unshuffled input has 800.
    path = tmp_path / "x.jsonl"
    _write_blocked_jsonl(path, COMPONENTS)
    bc.shuffle_jsonl(path, seed=0, num_buckets=8)

    comps = [json.loads(l)["text"].split()[0] for l in path.open()]
    longest = cur = 1
    for prev, nxt in zip(comps, comps[1:]):
        cur = cur + 1 if nxt == prev else 1
        longest = max(longest, cur)
    assert longest < 100, f"still block-ordered: longest run {longest}"


def test_more_buckets_than_lines(bc, tmp_path):
    path = tmp_path / "tiny.jsonl"
    original = _write_blocked_jsonl(path, {"a": 3, "b": 2})
    bc.shuffle_jsonl(path, seed=0, num_buckets=64)
    assert collections.Counter(path.read_text().splitlines(keepends=True)) == (
        collections.Counter(original)
    )


def test_line_count_mismatch_refuses_replace(bc, tmp_path, monkeypatch):
    # If pass 2 ever emits a different line count, the input must survive.
    path = tmp_path / "x.jsonl"
    _write_blocked_jsonl(path, {"a": 10})
    before = path.read_text()

    class _LossyRandom:
        def __init__(self, *a): ...
        def shuffle(self, lines):
            if lines:
                lines.pop()

        def randrange(self, n):
            return 0

    monkeypatch.setattr(bc.random, "Random", _LossyRandom)
    with pytest.raises(RuntimeError, match="line count changed"):
        bc.shuffle_jsonl(path, seed=0, num_buckets=2)
    assert path.read_text() == before
    assert not (tmp_path / "x.jsonl.shuffled.part").exists()


def test_shuffled_sources_and_marker_contract(bc, tmp_path):
    # dolma3_dolmino_mix is registered for the raw-level shuffle, and the
    # marker sidecar the orchestration gates on lives next to the raw file.
    assert "dolma3_dolmino_mix" in bc.SHUFFLED_SOURCES
    marker = bc._shuffled_path(tmp_path / "dolma3_dolmino_mix.jsonl")
    assert marker == tmp_path / "dolma3_dolmino_mix.jsonl.shuffled"
