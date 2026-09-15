"""Tokenizer wrapper.

We train a fresh byte-level BPE (vocab 128 256) using the HF ``tokenizers``
library, then freeze it. The trained ``tokenizer.json`` is stored in
artifact storage with a content-hash filename and recorded in every
checkpoint.

This module exposes:
- :func:`train_tokenizer` — one-shot training from a text iterator.
- :class:`Tokenizer` — load + encode/decode a frozen tokenizer.
- :func:`compute_hash` — content hash used for the manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers


def train_tokenizer(
    text_iterator: Iterable[str],
    output_path: str | Path,
    vocab_size: int = 128_256,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> Path:
    """Train a fresh BPE tokenizer matching Llama 3's pre-tokenizer settings.

    The trained file is written atomically (``output_path`` is the final
    location; we write to ``<output_path>.tmp`` then rename).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if special_tokens is None:
        # Match Llama 3's special-token reservation: <|endoftext|> + a few slots.
        special_tokens = [
            "<|endoftext|>",
            "<|begin_of_text|>",
            "<|start_header|>",
            "<|end_header|>",
            "<|eot|>",
            "<|pad|>",
        ]

    tok = HFTokenizer(models.BPE(byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # Llama 3 / GPT-4 style pre-tokenizer: byte-level, split digits,
            # add no prefix space.
            pre_tokenizers.Split(
                pattern=r"\d",  # explicit digit splitting
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iterator, trainer=trainer)

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tok.save(str(tmp))
    tmp.replace(output_path)
    return output_path


def compute_hash(tokenizer_path: str | Path) -> str:
    """SHA-256 of the tokenizer.json contents (canonicalised)."""
    p = Path(tokenizer_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


class Tokenizer:
    """Frozen tokenizer for use in data prep, train, and eval.

    Wraps HF ``tokenizers.Tokenizer`` so consumers don't have to import
    HF directly. Stateless beyond the underlying tokenizer instance.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._tok = HFTokenizer.from_file(str(self._path))
        self._hash = compute_hash(self._path)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def eos_id(self) -> int:
        # Default special token used as a document separator in the
        # indexed-binary stream.
        eos = self._tok.token_to_id("<|endoftext|>")
        if eos is None:
            raise ValueError(
                "tokenizer is missing <|endoftext|>; required as document separator"
            )
        return eos

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, ids: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return self._tok.decode_batch(ids, skip_special_tokens=skip_special_tokens)

    def encode_iter(self, texts: Iterable[str], batch_size: int = 1024) -> Iterator[list[int]]:
        """Memory-friendly encoding for prepare-time pipelines."""
        buffer: list[str] = []
        for t in texts:
            buffer.append(t)
            if len(buffer) >= batch_size:
                yield from self.encode_batch(buffer)
                buffer.clear()
        if buffer:
            yield from self.encode_batch(buffer)


def bytes_per_token(tokenizer: Tokenizer, sample_text: str) -> float:
    """A diagnostic for the tokenizer-quality unit test (plan/03 §2)."""
    if not sample_text:
        return 0.0
    n_bytes = len(sample_text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(sample_text))
    return n_bytes / max(n_tokens, 1)
