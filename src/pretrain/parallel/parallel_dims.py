"""Parallelism dimensions + device mesh construction.

Adapted from torchtitan's ``ParallelDims``
(``torchtitan/distributed/parallel_dims.py``), trimmed to the dims we
actually use: ``dp_replicate`` and ``dp_shard`` (FSDP). Tensor, pipeline,
context, and expert parallel are out of scope (TP was removed — we run
``tp=1`` only).

Mesh layout: ``("dp_replicate", "fsdp")``. ``dp_replicate`` at degree 1 is
created with a real (but comm-free) group; ``fsdp`` is *always* created so
``fully_shard`` can apply its MixedPrecisionPolicy even at degree 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

LOG = logging.getLogger(__name__)

__all__ = ["ParallelDims"]


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    world_size: int

    _meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None
    _dense_mesh: DeviceMesh | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        assert self.dp_replicate >= 1, f"dp_replicate must be >= 1, got {self.dp_replicate}"
        assert self.dp_shard == -1 or self.dp_shard >= 1, "dp_shard must be -1 or >= 1"
        if self.dp_shard < 0:
            self.dp_shard = self.world_size // self.dp_replicate
        assert self.dp_shard >= 1
        assert self.dp_replicate * self.dp_shard == self.world_size, (
            f"Invalid parallel dims: dp_replicate({self.dp_replicate}) * "
            f"dp_shard({self.dp_shard}) != world_size({self.world_size})"
        )

    def _mesh_exists(self, name: str, degree: int) -> bool:
        # fsdp is always created with a real backend so fully_shard()'s
        # MixedPrecisionPolicy can apply even at degree 1.
        if name == "fsdp":
            return True
        return degree > 1

    def build_mesh(self, device_type: str | None = None) -> DeviceMesh:
        """Build the device mesh and cache sub-meshes by name.

        ``device_type`` defaults to ``"cuda"`` when CUDA is available and
        ``"cpu"`` otherwise. The CPU multi-rank verification harness
        passes ``"cpu"`` explicitly even though CUDA is available, so the
        process group + mesh tensors all live on CPU together with the
        test's gloo backend.
        """
        if self._world_mesh is not None:
            return self._world_mesh

        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info(
            "building device mesh: dp_replicate=%d dp_shard=%d (world=%d)",
            self.dp_replicate, self.dp_shard, self.world_size,
        )

        self._world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",)
        )

        dim_names = ("dp_replicate", "fsdp")
        dim_degrees = (self.dp_replicate, self.dp_shard)
        # We used to request the ``fake`` backend for degree-1 axes to
        # skip trivial PG creation, but that backend is only registered
        # in some PyTorch builds — the NGC pytorch:26.03 container raises
        # ``Unknown c10d backend type FAKE``. The cost of creating real
        # but degree-1 groups is negligible (one extra group per skippable
        # axis, no comms), so we always pass an empty override.
        if hasattr(self._world_mesh, "_unflatten"):
            self._dense_mesh = self._world_mesh._unflatten(
                0, dim_degrees, dim_names, backend_override={}
            )
        else:
            # Torch < 2.7 (e.g. NGC pytorch:25.01) lacks DeviceMesh._unflatten.
            # Build the dense mesh directly; this creates a separate PG
            # hierarchy from _world_mesh, but the extra groups are inert.
            # Drop this branch once we're back on a container with torch ≥ 2.7.
            self._dense_mesh = init_device_mesh(
                device_type, dim_degrees, mesh_dim_names=dim_names
            )

        self._meshes = {
            "dp_replicate": self._dense_mesh["dp_replicate"],
            "fsdp": self._dense_mesh["fsdp"],
        }
        return self._world_mesh

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Return a 1D or composed mesh, or None if the dim is degree-1 (and
        not ``fsdp``, which is always exposed)."""
        if not self._meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        for name in dims:
            if name not in self._meshes:
                raise ValueError(
                    f"Invalid mesh dim '{name}'. Valid: {list(self._meshes.keys())}"
                )

        if any(not self._mesh_exists(d, self._meshes[d].size()) for d in dims):
            return None

        if len(dims) == 1:
            return self._meshes[dims[0]]

        assert self._dense_mesh is not None
        return self._dense_mesh[tuple(dims)]

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            raise ValueError(
                f"Mesh '{dims}' not available — corresponding parallelism dim "
                f"is degree 1 (and is not 'fsdp')."
            )
        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self.build_mesh()
        assert self._world_mesh is not None
        return self._world_mesh

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def fsdp_enabled(self) -> bool:
        return self.dp_shard_enabled

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate_enabled or self.dp_shard_enabled

    @property
    def dp_world_size(self) -> int:
        # Every rank consumes unique tokens (no TP groups sharing a minibatch).
        # If CP/PP are added later, divide by those.
        return self.dp_replicate * self.dp_shard

    def dp_rank(self, global_rank: int | None = None) -> int:
        """Unique-token rank used to stride the data loader. With the
        ``("dp_replicate", "fsdp")`` mesh and no TP, this is just the global
        rank."""
        if global_rank is None:
            global_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
        return global_rank

    def make_dp_gloo_group(self):
        """Build a gloo ProcessGroup spanning all ranks (all are DP-equivalent
        with no TP). Returns ``None`` when not under distributed init or when
        there's only one rank. gloo because the intended traffic is tiny
        CPU-side metadata (e.g. 32-byte digests), matching
        :class:`pretrain.train.checkpoint.Checkpointer`."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None
        if self.dp_world_size <= 1:
            return None
        world_size = torch.distributed.get_world_size()
        return torch.distributed.new_group(ranks=list(range(world_size)), backend="gloo")
