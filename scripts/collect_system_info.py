#!/usr/bin/env python3
"""Per-pod system + GPU snapshot for reproducibility records.

Captures OS, kernel, drivers, CUDA, NCCL, Python/Torch versions, GPU
inventory, NVLink topology, git SHAs, and reproducibility-relevant env
vars. Designed to be invoked once at the start of every k8s pod in a
distributed training job so we have a durable record of what hardware
actually ran each rank.

Output is JSON (stdlib, no extra deps). Convert to YAML with
`yq -P` if a human wants to skim it.

CLI:
    python scripts/collect_system_info.py \\
        --output runs/<run_id>/system_info/pod-<index>.json

The script never raises on missing tools or files — every probe is
isolated; failures land in the output as {"error": ...}.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD")

# Env var prefixes worth recording. Anything that affects determinism,
# NCCL transport, rendezvous, or repop dispatch.
ENV_PREFIXES = (
    "NCCL_", "TORCH_", "TORCHELASTIC_", "CUDA", "CUDNN_", "CUBLAS",
    "REPOP_", "HF_", "OMP_", "MKL_", "PYTHON",
    "JOB_", "POD_", "KUBERNETES_", "RUN_", "WANDB_",
    "MASTER_", "RDZV_", "GROUP_",
    "LD_LIBRARY", "LD_PRELOAD",
)

ENV_EXTRA = (
    "HOSTNAME", "USER", "PATH", "PWD",
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "PIP_BREAK_SYSTEM_PACKAGES", "TF32_OVERRIDE",
)


def _run(cmd, timeout=15):
    """Run a shell command; return {cmd, returncode, stdout, stderr} or {error}."""
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=isinstance(cmd, str),
        )
        return {
            "cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
            "returncode": out.returncode,
            "stdout": out.stdout.strip(),
            "stderr": out.stderr.strip(),
        }
    except FileNotFoundError as e:
        return {"cmd": str(cmd), "error": f"not found: {e}"}
    except subprocess.TimeoutExpired:
        return {"cmd": str(cmd), "error": f"timeout after {timeout}s"}
    except Exception as e:  # noqa: BLE001 — best-effort probe
        return {"cmd": str(cmd), "error": repr(e)}


def _which(name):
    return shutil.which(name)


def _read_text(path, max_bytes=65536):
    try:
        with open(path, "r", errors="replace") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            data = data[:max_bytes] + "\n... [truncated]"
        return data
    except OSError as e:
        return f"<unreadable: {e}>"


def _exists(path):
    return Path(path).exists()


def collect_host():
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uname": " ".join(platform.uname()),
    }
    if _which("hostname"):
        info["fqdn"] = _run("hostname -f").get("stdout")
    for path in (
        "/etc/os-release",
        "/proc/version",
        "/etc/nv_tags",
        "/etc/image-release",
        "/etc/shinit_v2",
    ):
        if _exists(path):
            info[path] = _read_text(path, 4096)
    return info


def collect_kubernetes():
    keys = (
        "JOB_COMPLETION_INDEX", "JOB_ID", "POD_NAME", "POD_NAMESPACE",
        "POD_IP", "NODE_NAME", "HOSTNAME",
        "KUBERNETES_SERVICE_HOST", "KUBERNETES_PORT",
    )
    return {k: os.environ[k] for k in keys if k in os.environ}


def collect_cpu():
    info = {"logical_count": os.cpu_count()}
    if _which("lscpu"):
        r = _run("lscpu")
        if r.get("returncode") == 0:
            info["lscpu"] = r["stdout"]
    if _exists("/proc/cpuinfo"):
        # First processor block: model name, flags, microcode.
        lines = _read_text("/proc/cpuinfo", 16384).splitlines()
        block = []
        for ln in lines:
            if not ln.strip():
                break
            block.append(ln)
        info["proc_cpuinfo_first"] = "\n".join(block)
    return info


def collect_memory():
    if _exists("/proc/meminfo"):
        text = _read_text("/proc/meminfo", 4096)
        head = "\n".join(text.splitlines()[:10])
        return {"proc_meminfo_head": head}
    return {}


def collect_disk():
    info = {}
    if _which("df"):
        info["df_hT"] = _run("df -hT").get("stdout")
    return info


def collect_network():
    info = {}
    if _which("ip"):
        info["ip_addr"] = _run("ip -o addr").get("stdout")
        info["ip_route"] = _run("ip route").get("stdout")
    iface = os.environ.get("NCCL_SOCKET_IFNAME")
    if iface and _which("ethtool"):
        info[f"ethtool_{iface}"] = _run(f"ethtool {iface}").get("stdout")
    return info


def collect_gpu():
    if not _which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not on PATH"}

    fields = (
        "index", "uuid", "name", "serial", "vbios_version",
        "pci.bus_id", "compute_cap", "driver_version",
        "memory.total", "ecc.mode.current",
        "ecc.errors.uncorrected.aggregate.total",
        "mig.mode.current", "persistence_mode",
        "power.max_limit", "power.default_limit",
        "clocks.max.graphics", "clocks.max.sm", "clocks.max.memory",
        "compute_mode",
    )
    csv = _run([
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader",
    ])
    devices = []
    if csv.get("returncode") == 0 and csv.get("stdout"):
        for line in csv["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            devices.append(dict(zip(fields, parts)))

    return {
        "available": True,
        "count": len(devices),
        "list": _run("nvidia-smi -L").get("stdout"),
        "topology": _run("nvidia-smi topo -m").get("stdout"),
        "devices": devices,
        "query_csv_raw": csv,
    }


def collect_nvidia_driver():
    info = {}
    if _exists("/proc/driver/nvidia/version"):
        info["proc_driver_nvidia_version"] = _read_text(
            "/proc/driver/nvidia/version", 2048,
        )
    if _which("nvidia-smi"):
        info["nvidia_smi_version"] = _run("nvidia-smi --version").get("stdout")
    if _exists("/usr/lib/x86_64-linux-gnu"):
        # NCCL .so version is informative when torch ships its own.
        r = _run("ls -1 /usr/lib/x86_64-linux-gnu/libnccl* 2>/dev/null || true")
        if r.get("stdout"):
            info["libnccl_files"] = r["stdout"]
    return info


def collect_cuda():
    info = {}
    if _which("nvcc"):
        info["nvcc_version"] = _run("nvcc --version").get("stdout")
    for path in (
        "/usr/local/cuda/version.json",
        "/usr/local/cuda/version.txt",
    ):
        if _exists(path):
            info[path] = _read_text(path, 4096)
    # Resolve the symlink target if /usr/local/cuda is a symlink — the
    # NGC containers point /usr/local/cuda -> /usr/local/cuda-<ver>.
    cuda_root = Path("/usr/local/cuda")
    if cuda_root.is_symlink():
        info["cuda_root_symlink_target"] = os.readlink(cuda_root)
    return info


def collect_python():
    return {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "pythonpath_env": os.environ.get("PYTHONPATH", ""),
        "sys_path_head": sys.path[:8],
    }


def collect_torch():
    try:
        import torch
    except ImportError as e:
        return {"available": False, "import_error": repr(e)}

    info = {
        "available": True,
        "version": torch.__version__,
        "git_version": getattr(torch.version, "git_version", None),
        "cuda_built": torch.version.cuda,
        "hip_built": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as e:  # noqa: BLE001
        info["cudnn_version_error"] = repr(e)
    try:
        info["cuda_runtime_compiled_version"] = torch._C._cuda_getCompiledVersion()
    except Exception as e:  # noqa: BLE001
        info["cuda_runtime_compiled_version_error"] = repr(e)
    try:
        info["nccl_version"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as e:  # noqa: BLE001
        info["nccl_version_error"] = repr(e)
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "major": props.major,
                    "minor": props.minor,
                    "total_memory_bytes": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                })
    except Exception as e:  # noqa: BLE001
        info["device_query_error"] = repr(e)
    return info


def collect_repop():
    try:
        import repop  # type: ignore
    except ImportError as e:
        return {
            "available": False,
            "import_error": repr(e),
            "execution_mode_env": os.environ.get("REPOP_EXECUTION_MODE"),
        }
    return {
        "available": True,
        "version": getattr(repop, "__version__", None),
        "file": getattr(repop, "__file__", None),
        "execution_mode_env": os.environ.get("REPOP_EXECUTION_MODE"),
    }


def collect_git(paths):
    out = {}
    for p in paths:
        if not _exists(p):
            continue
        entry = {}
        for label, args in (
            ("head_sha", ["rev-parse", "HEAD"]),
            ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
            ("describe", ["describe", "--always", "--tags", "--dirty"]),
            ("status_short", ["status", "--short"]),
            ("origin", ["config", "--get", "remote.origin.url"]),
        ):
            r = _run(["git", "-C", str(p), *args], timeout=10)
            if r.get("returncode") == 0:
                entry[label] = r["stdout"]
            else:
                entry[label] = r.get("error") or r.get("stderr") or r.get("stdout")
        out[str(p)] = entry
    return out


def _looks_like_secret(name):
    upper = name.upper()
    return any(hint in upper for hint in SECRET_HINTS)


def collect_env():
    matched = {}
    for k, v in sorted(os.environ.items()):
        if any(k.startswith(p) for p in ENV_PREFIXES) or k in ENV_EXTRA:
            matched[k] = "<redacted>" if _looks_like_secret(k) else v
    return matched


def collect_pip_freeze():
    if not _which("pip"):
        return None
    r = _run("pip list --format=freeze", timeout=30)
    if r.get("returncode") == 0:
        return r["stdout"].splitlines()
    return r


def collect(git_paths=None):
    git_paths = git_paths or (".",)
    return {
        "schema_version": SCHEMA_VERSION,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": collect_host(),
        "kubernetes": collect_kubernetes(),
        "cpu": collect_cpu(),
        "memory": collect_memory(),
        "disk": collect_disk(),
        "network": collect_network(),
        "gpu": collect_gpu(),
        "nvidia_driver": collect_nvidia_driver(),
        "cuda": collect_cuda(),
        "python": collect_python(),
        "torch": collect_torch(),
        "repop": collect_repop(),
        "git": collect_git(git_paths),
        "env": collect_env(),
        "pip_freeze": collect_pip_freeze(),
    }


def _summary_line(info):
    host = info["host"].get("hostname")
    k8s_idx = info["kubernetes"].get("JOB_COMPLETION_INDEX")
    gpu = info["gpu"]
    if gpu.get("available") and gpu.get("devices"):
        first = gpu["devices"][0]
        driver = first.get("driver_version", "n/a")
        gpu_name = first.get("name", "n/a")
        gpu_part = f"{gpu['count']}×{gpu_name} driver={driver}"
    else:
        gpu_part = "no GPUs"
    torch_v = info["torch"].get("version", "n/a")
    torch_cuda = info["torch"].get("cuda_built", "n/a")
    return (
        f"[system-info] host={host} k8s_index={k8s_idx} "
        f"torch={torch_v} torch.cuda={torch_cuda} {gpu_part}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect per-pod system + GPU info for reproducibility.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON path. Parent dirs are created. Stdout if omitted.",
    )
    parser.add_argument(
        "--git-path", action="append", default=None,
        help="Repo path to record HEAD/branch/dirty for. Repeatable.",
    )
    args = parser.parse_args(argv)

    info = collect(git_paths=args.git_path)
    payload = json.dumps(info, indent=2, default=str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(_summary_line(info), file=sys.stderr)
        print(f"[system-info] wrote {args.output}", file=sys.stderr)
    else:
        print(payload)
        print(_summary_line(info), file=sys.stderr)


if __name__ == "__main__":
    main()
