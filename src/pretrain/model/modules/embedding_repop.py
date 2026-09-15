"""Untied embedding using repop ops for both halves.

Token lookup uses ``repop.nn.embedding.Embedding`` — a drop-in for
``torch.nn.Embedding`` with a **bitwise-reproducible backward**. Stock
``nn.Embedding`` scatters its gradient via ``index_add``, whose repeated-token
accumulation is a nondeterministic atomic-add reduction on CUDA (non-associative
fp → not byte-reproducible, differs CPU↔GPU); repop's version makes the grad a
deterministic fixed-order scatter-add, so it is byte-identical across devices.
This is the op that closes the last cross-device BFR gap in the training step
(needed for the single-device audit to match a cluster run bit-for-bit).

The output projection is a [d_model → vocab_size] matmul — the largest single
GEMM in the forward — covered by the repop ``Linear`` kernel.

Attribute names (``tok_embeddings`` and ``output``) — and the parameter name
``tok_embeddings.weight`` (repop's ``Embedding`` exposes the same ``.weight``) —
are kept identical to ``UntiedEmbedding`` so ``parallelize_llama3_repop`` (which
``fully_shard``-wraps the inner sub-modules), the optimizer param-group split,
and checkpoint FQNs continue to work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pretrain.model.registry import EMBEDDING
from repop.nn.embedding import Embedding as RepopEmbedding
from repop.nn.linear import Linear as RepopLinear


@EMBEDDING.register("untied_repop")
class UntiedEmbeddingRepop(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.tok_embeddings = RepopEmbedding(vocab_size, d_model)
        self.output = RepopLinear(d_model, vocab_size, bias=False)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embeddings(input_ids)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden)
