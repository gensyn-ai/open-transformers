"""Per-source data manifest.

A manifest records, for one source: the list of shard prefixes, the
per-shard token counts, the dtype, and the hash of the tokenizer that
produced them. The manifest is the single source of truth for mix
weights — never use the dataset card's claim, never recount on the fly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import yaml


def blake2b_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """blake2b digest (32 bytes / 64 hex chars) of a file, streamed in 1 MiB chunks.

    Matches the digest function used by training state hashing (see
    ``pretrain.train.state_hash``) so every "bytes I committed to"
    artifact in the project speaks the same hash algorithm — manifests,
    checkpoints, and the per-step state digest are all comparable
    without per-call algorithm awareness.
    """
    h = hashlib.blake2b(digest_size=32)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


@dataclasses.dataclass
class ShardInfo:
    prefix: str         # path stem (no .bin/.idx suffix)
    num_documents: int
    token_count: int
    bin_blake2b: str = ""    # blake2b-32 of <prefix>.bin (empty if not recorded)
    idx_blake2b: str = ""    # blake2b-32 of <prefix>.idx


@dataclasses.dataclass
class SourceManifest:
    name: str
    tokenizer_hash: str
    dtype: str          # numpy dtype name, e.g. "uint32"
    shards: list[ShardInfo]
    filter_version: str = ""    # e.g. for stack-v2 license filter
    # blake2b-32 of each raw input JSONL that fed the sharder, keyed by
    # filename. Single-worker: one entry. Fanout: one per worker pod.
    # Empty if not recorded (older manifests).
    raw_jsonl_blake2b: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self.shards)

    @property
    def total_documents(self) -> int:
        return sum(s.num_documents for s in self.shards)

    def to_yaml(self) -> str:
        return yaml.safe_dump(dataclasses.asdict(self), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "SourceManifest":
        d = yaml.safe_load(text)
        d["shards"] = [ShardInfo(**s) for s in d["shards"]]
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "SourceManifest":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    def hash(self) -> str:
        return hashlib.sha256(self.to_yaml().encode("utf-8")).hexdigest()
