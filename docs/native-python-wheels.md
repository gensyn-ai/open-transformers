# Native Python audit distribution

`PRETRAIN_PYTHON_NATIVE=1 uv build --python 3.11 --wheel` builds a pretraining
wheel with compiled CPython modules and no Python source or bytecode. Build
once on each target platform. The ordinary development wheel remains Python.
On macOS set `MACOSX_DEPLOYMENT_TARGET=14.0` for the audit's deployment floor.

The build preserves the configuration files, package metadata, licence and
console entry points. Use `pretrain-audit-replay`; `python -m` cannot execute
a compiled module. The kit installer must select the platform's pretraining
wheel as well as its repop wheel. Passing both platform wheels to pip is an
error. No replacement kit is published by this change.

Supply each prebuilt platform wheel to the publisher with `--pretrain-wheel`.
The publisher requires matching native pretraining/repop platform sets and
checks the pretraining build stamp against the publishing commit. An archived
build must set `PRETRAIN_BUILD_COMMIT` to the verified archive's full commit;
an unstamped or dirty wheel cannot pass publication. The exact digest policy
still applies after these structural checks.

The Cython build disables docstrings, inferred annotation types and floating
point contraction. Pydantic treats compiled methods as methods. Hash tags use
explicit NUL bytes: Cython 3.1.2 miscompiled the NUL inside the formatted
optimizer tags, which changed state hashes despite identical tensor bytes.
`test_native_hash_wire_format.py` covers both full and sharded hash formats.

Compiled distributions still expose machine code, names and constants. This
removes directly readable implementation files; it is not a secrecy guarantee.
