"""Health check tests (ADR 0003 §Validation). Run: python -m pytest -q health/test_health.py"""
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "..", "..", "submission")
sys.path.insert(0, SUB)
from health import health_check, xid_scan  # noqa: E402

TH = yaml.safe_load(open(os.path.join(SUB, "health", "thresholds.yaml")))
FIX = os.path.join(HERE, "fixtures", "health")


def test_xid_classes_and_no_second_drain(tmp_path):
    lines = open(os.path.join(FIX, "kernel_xid.log")).read().splitlines()
    state = str(tmp_path / "seen.json")
    first = xid_scan.scan(lines, TH["xid"], state)
    assert first["status"] == "fail"
    assert {(x["xid"], x["class"]) for x in first["new"]} == {(13, "warn"), (79, "fatal"), (94, "fatal"),
                                                              (157, "unknown")}
    again = xid_scan.scan(lines, TH["xid"], state)   # same log after a repair/reboot: no new drain
    assert again["status"] == "pass" and again["already_seen"] == 4


def test_gpu_fixture_findings():
    res = health_check.gpu_checks(TH, 8, os.path.join(FIX, "nvidia_smi_bad.csv"))
    by = {(c["name"], c["gpu"]): c["status"] for c in res}
    assert by[("ecc_uncorrectable", 3)] == "fail"
    assert by[("gpu_temperature", 2)] == "warn" and by[("gpu_throttle", 2)] == "warn"
    assert by[("pcie_link", 4)] == "warn"
    assert health_check.verdict(res) == "fail"


def test_missing_tools_are_skipped_never_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("V5_HEALTH_STATE", str(tmp_path))
    monkeypatch.setattr(health_check, "STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", "/nonexistent")
    res = health_check.gpu_checks(TH, 8)
    assert res[0]["status"] == "skipped" and res[0]["required"]
    assert health_check.verdict(res) == "warn"          # a required check we could not run is not a pass
    assert health_check.verdict(health_check.gpu_checks(TH, 0)) == "pass"  # CPU node: not required


def test_prolog_warn_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("V5_EXPECT_GPUS", "8")      # GPU node with no nvidia-smi -> warn
    monkeypatch.setattr(health_check, "STATE_DIR", str(tmp_path))
    assert health_check.main(["--mode", "prolog", "--report-only", "--kernel-log", os.devnull]) == 0
    assert health_check.main(["--mode", "periodic", "--report-only", "--kernel-log", os.devnull]) == 3


def test_forced_fail_and_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("V5_EXPECT_GPUS", "0")
    monkeypatch.setattr(health_check, "STATE_DIR", str(tmp_path))
    (tmp_path / "force_fail").write_text("drill: GPU 3 Xid 79")
    out = tmp_path / "r.json"
    code = health_check.main(["--mode", "prolog", "--report-only", "--kernel-log", os.devnull, "--out", str(out)])
    r = json.load(open(out))
    assert code == 4 and r["verdict"] == "fail" and r["drain_reason"] == "drill_force_fail"
    assert set(r) == {"schema", "time", "host", "mode", "checks", "verdict", "drain_reason", "runtime_seconds"}
    assert all(c["status"] in ("pass", "warn", "fail", "skipped") for c in r["checks"])


def test_allreduce_exact_on_gloo(tmp_path):
    out = tmp_path / "ar.json"
    p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                        os.path.join(SUB, "health", "allreduce_test.py"), "--max-mb", "4", "--iters", "3", "--out", str(out)],
                       capture_output=True, text=True, timeout=120)
    r = json.load(open(out))
    assert p.returncode == 0 and r["status"] == "pass" and r["backend"] == "gloo" and r["ranks"] == 2
    assert all(s["busbw_gbps"] > 0 for s in r["metrics"]["sizes"])


def test_periodic_mode_never_runs_dcgm_diag(tmp_path, monkeypatch):
    """diag loads the GPUs; on a training node a false fail would drain it and requeue the main run."""
    bin_dir, log = tmp_path / "bin", tmp_path / "dcgmi.log"
    bin_dir.mkdir()
    (bin_dir / "dcgmi").write_text(f'#!/bin/sh\necho "$@" >> {log}\necho \'{{"Overall Health": "Healthy"}}\'\n')
    (bin_dir / "dcgmi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("V5_EXPECT_GPUS", "8")
    monkeypatch.setattr(health_check, "STATE_DIR", str(tmp_path))
    fx = os.path.join(FIX, "nvidia_smi_bad.csv")
    for mode in ("periodic", "prolog"):
        health_check.main(["--mode", mode, "--report-only", "--nvidia-smi-fixture", fx, "--out", str(tmp_path / mode)])
    calls = log.read_text().splitlines()
    assert calls[:2] == ["health -s a", "health -c -j"]           # periodic: watches only
    assert calls[2:] == ["diag -r 1 -j"]                          # prolog: the GPUs are idle
    rep = json.load(open(tmp_path / "periodic"))
    assert [c["status"] for c in rep["checks"] if c["name"] == "dcgm_health"] == ["pass"]


def test_renamed_throttle_field_is_not_a_failure(monkeypatch):
    """Newer drivers reject clocks_throttle_reasons.*; the retry with clocks_event_reasons.* must be used, not
    reported as a broken GPU (that would drain healthy nodes)."""
    good = "".join(f"{i}, 40, 300.0, 0, 0, 0, Not Active, 16, 16\n" for i in range(8))

    def fake_run(cmd, timeout=60):
        old = any("clocks_throttle_reasons" in c for c in cmd)
        return subprocess.CompletedProcess(cmd, 2 if old else 0, "" if old else good, "unknown field" if old else "")

    monkeypatch.setattr(health_check.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(health_check, "run", fake_run)
    res = health_check.gpu_checks(TH, 8)
    assert health_check.verdict(res) == "pass", res
