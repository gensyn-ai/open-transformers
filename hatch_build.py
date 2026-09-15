"""Optional native audit distribution; ordinary development wheels stay Python."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if os.environ.get("PRETRAIN_PYTHON_NATIVE") == "1":
            if version == "editable":
                raise ValueError("native audit builds cannot be editable")
            build_data["pure_python"] = False
            build_data["infer_tag"] = True

    def finalize(self, version, build_data, artifact_path):
        if os.environ.get("PRETRAIN_PYTHON_NATIVE") != "1":
            return
        from wheel.wheelfile import WheelFile

        artifact = Path(artifact_path).resolve()
        with TemporaryDirectory(
            prefix="pretrain-native-", dir=artifact.parent
        ) as temporary:
            root = Path(temporary)
            sources, payload = root / "sources", root / "payload"
            sources.mkdir()
            payload.mkdir()
            with ZipFile(artifact) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("duplicate wheel members")
                for name in names:
                    if name.endswith(("/", "/RECORD", "/RECORD.jws", "/RECORD.p7s")):
                        continue
                    member = Path(name)
                    if member.is_absolute() or ".." in member.parts or "\\" in name:
                        raise ValueError(f"unsafe wheel member: {name}")
                    target = (sources if name.endswith(".py") else payload) / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
            if not list(sources.rglob("*.py")):
                raise ValueError("native wheel has no Python inputs")
            commit = os.environ.get("PRETRAIN_BUILD_COMMIT", "")
            if not commit:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.root,
                    text=True,
                    capture_output=True,
                )
                commit = result.stdout.strip() if result.returncode == 0 else "unknown"
                if (
                    commit != "unknown"
                    and subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=self.root
                    ).strip()
                ):
                    commit += "-dirty"
            stamp_file = payload / "pretrain/_native_build.json"
            stamp_file.parent.mkdir(parents=True, exist_ok=True)
            stamp_file.write_text(
                json.dumps({"commit": commit, "python_native": True}, sort_keys=True)
                + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(sources),
                    str(payload),
                ],
                check=True,
                cwd=root,
            )
            for source in sources.rglob("*.py"):
                relative = source.relative_to(sources).with_suffix("")
                if not list((payload / relative.parent).glob(relative.name + ".*.so")):
                    raise ValueError(f"module did not compile: {relative}")
            output = root / artifact.name
            with WheelFile(output, "w") as wheel:
                for member in sorted(payload.rglob("*")):
                    if member.is_file():
                        if member.suffix.lower() in {
                            ".py",
                            ".pyc",
                            ".pyo",
                            ".pyx",
                            ".pxd",
                            ".c",
                            ".cpp",
                            ".cu",
                            ".h",
                            ".hpp",
                            ".metal",
                            ".mm",
                            ".air",
                        }:
                            raise ValueError(
                                f"source survived native build: {member.name}"
                            )
                        wheel.write(member, str(member.relative_to(payload)))
            os.replace(output, artifact)


def compile_modules(sources: Path, payload: Path) -> None:
    from Cython.Build import cythonize
    from Cython.Compiler import Options
    from setuptools import Extension, setup
    from setuptools.command.build_ext import build_ext

    class NativeBuildExtension(build_ext):
        def build_extensions(self):
            self.compiler.linker_so = [
                arg
                for arg in self.compiler.linker_so
                if not arg.startswith("-Wl,-rpath,")
            ]
            super().build_extensions()

    Options.docstrings = False
    flags = ["-O2", "-g0", "-ffp-contract=off", "-fvisibility=hidden"]
    for directory, replacement in (
        (str(sources.parent), "/native-build"),
        (str(Path.home()), "/build"),
        (sys.prefix, "/python-env"),
    ):
        flags.append(f"-ffile-prefix-map={directory}={replacement}")
    link_flags = ["-Wl,-x"] if sys.platform == "darwin" else ["-Wl,--strip-all"]
    extensions = [
        Extension(
            ".".join(source.relative_to(sources).with_suffix("").parts),
            [str(source.relative_to(sources))],
            extra_compile_args=flags,
            extra_link_args=link_flags,
        )
        for source in sorted(sources.rglob("*.py"))
    ]
    os.chdir(sources)
    setup(
        name="pretrain-native-python",
        cmdclass={"build_ext": NativeBuildExtension},
        ext_modules=cythonize(
            extensions,
            build_dir=str(sources.parent / "generated"),
            compiler_directives={
                "language_level": 3,
                "annotation_typing": False,
                "infer_types": False,
                "binding": True,
                "embedsignature": False,
                "emit_code_comments": False,
            },
        ),
        script_args=[
            "build_ext",
            "--build-lib",
            str(payload),
            "--build-temp",
            str(sources.parent / "objects"),
            "-j",
            os.environ.get("MAX_JOBS", "4"),
        ],
    )


if __name__ == "__main__":
    compile_modules(Path(sys.argv[1]), Path(sys.argv[2]))
