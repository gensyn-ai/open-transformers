"""Thread defaults must preserve overrides and the replay's hash gate."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pretrain.cli import audit_replay, cpu_threads


@pytest.fixture
def runtime(monkeypatch):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)
    state = {"threads": 8, "sets": []}
    monkeypatch.setattr(cpu_threads.torch, "get_num_threads", lambda: state["threads"])

    def set_threads(count):
        state["threads"] = count
        state["sets"].append(count)

    monkeypatch.setattr(cpu_threads.torch, "set_num_threads", set_threads)
    return state


def test_auto_uses_performance_cores(runtime, monkeypatch):
    monkeypatch.setattr(cpu_threads, "_performance_cores", lambda: 10)
    result = cpu_threads.configure_cpu_threads()
    assert result == {
        "requested": "auto",
        "source": "apple-performance-cores",
        "torch_num_threads": 10,
        "omp_num_threads": "10",
        "mkl_num_threads": "10",
    }
    assert runtime["sets"] == [10]


@pytest.mark.parametrize(
    "key,value",
    [("OMP_NUM_THREADS", "6"), ("OMP_NUM_THREADS", "6,2"), ("MKL_NUM_THREADS", "4")],
)
def test_auto_preserves_environment(runtime, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        cpu_threads, "_performance_cores", lambda: pytest.fail("must not probe")
    )
    result = cpu_threads.configure_cpu_threads()
    assert result["source"] == "environment"
    assert result["torch_num_threads"] == 8
    assert cpu_threads.os.environ[key] == value
    assert runtime["sets"] == []


def test_explicit_count_overrides_environment(runtime, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "6")
    result = cpu_threads.configure_cpu_threads("10")
    assert result["source"] == "explicit"
    assert result["omp_num_threads"] == result["mkl_num_threads"] == "10"
    assert runtime["sets"] == [10]


def test_unavailable_topology_preserves_runtime(runtime, monkeypatch):
    monkeypatch.setattr(cpu_threads, "_performance_cores", lambda: None)
    result = cpu_threads.configure_cpu_threads()
    assert result["source"] == "runtime-default"
    assert result["torch_num_threads"] == 8
    assert runtime["sets"] == []


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "all", ""])
def test_invalid_count_is_rejected(runtime, value):
    with pytest.raises(ValueError, match="positive integer"):
        cpu_threads.configure_cpu_threads(value)
    assert runtime["sets"] == []


@pytest.mark.parametrize("answer", ["10\n", "0\n", "nonsense\n"])
def test_topology_output(monkeypatch, answer):
    monkeypatch.setattr(cpu_threads.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cpu_threads.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cpu_threads.os, "cpu_count", lambda: 14)
    monkeypatch.setattr(
        cpu_threads.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=answer)
    )
    assert cpu_threads._performance_cores() == (10 if answer == "10\n" else None)


def test_topology_timeout_falls_back(monkeypatch):
    monkeypatch.setattr(cpu_threads.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cpu_threads.platform, "machine", lambda: "arm64")

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("sysctl", 1)

    monkeypatch.setattr(cpu_threads.subprocess, "run", timeout)
    assert cpu_threads._performance_cores() is None


def test_non_apple_host_does_not_probe(monkeypatch):
    monkeypatch.setattr(cpu_threads.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        cpu_threads.subprocess, "run", lambda *a, **k: pytest.fail("must not probe")
    )
    assert cpu_threads._performance_cores() is None


def test_cli_applies_before_replay_and_reports(runtime, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["audit", "--device", "cpu", "--cpu-threads", "6"])

    def replay(*a, **k):
        assert runtime["threads"] == 6
        return {"match": True, "state_hash": "a" * 64}

    monkeypatch.setattr(audit_replay, "audit_replay", replay)
    audit_replay.main()
    result = json.loads(capsys.readouterr().out)
    assert result["cpu_threads"]["torch_num_threads"] == 6
    assert result["state_hash"] == "a" * 64


def test_cli_keeps_hash_failure(runtime, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["audit", "--device", "cpu", "--cpu-threads", "6"])
    monkeypatch.setattr(
        audit_replay,
        "audit_replay",
        lambda *a, **k: {"match": False, "state_hash": "a" * 64, "expected": "b" * 64},
    )
    with pytest.raises(SystemExit, match="AUDIT FAILED"):
        audit_replay.main()


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_accelerator_default_is_unchanged(runtime, monkeypatch, device):
    monkeypatch.setattr(sys, "argv", ["audit", "--device", device])
    monkeypatch.setattr(audit_replay, "audit_replay", lambda *a, **k: {"match": True})
    audit_replay.main()
    assert runtime["sets"] == []


def test_accelerator_explicit_threads_rejected(runtime, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["audit", "--device", "mps", "--cpu-threads", "6"])
    with pytest.raises(SystemExit, match="requires --device cpu"):
        audit_replay.main()
    assert runtime["sets"] == []
