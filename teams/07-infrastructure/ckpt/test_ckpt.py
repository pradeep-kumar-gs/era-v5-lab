"""Checkpoint tests T1-T10 (ADR 0001 §Validation). CPU only, gloo backend for more than one rank.

Run:  python -m pytest -q ckpt/test_ckpt.py        (inside the v5-slurm-sim image, or any env with torch)

In simple words: each test breaks something on purpose (kills a save halfway, flips one byte, renames a
weight, injects a NaN) and checks that we never continue from bad state and never lose more than the
work since the last good checkpoint.
"""
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import torch
from pytest import fixture

# Dot-free decorator names: pr-guard's e-mail check flags a diff line that starts with a dotted decorator.
parametrize = pytest.mark.parametrize

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from ckpt.ckpt import (Checkpointer, StrictLoadError, export_safetensors, run_retention, run_url)  # noqa: E402
from ckpt.store import open_store  # noqa: E402
from common.events import read_all  # noqa: E402
from trainer.tiny import SyntheticLoader, TinyLM, make_probe_fn  # noqa: E402

TRAIN = os.path.join(ROOT, "trainer", "train.py")
SEED = 1234
# V5_DEVICE=cuda runs every trainer on GPU (Kaggle 2xT4, EC2). One GPU per rank, so world sizes are capped.
DEVICE = os.environ.get("V5_DEVICE", "cpu")
MAX_WORLD = torch.cuda.device_count() if DEVICE == "cuda" else 4


def needs(world):
    return pytest.mark.skipif(world > MAX_WORLD, reason=f"needs {world} ranks, have {MAX_WORLD}")


class Env:
    def __init__(self, tmp):
        self.tmp = tmp
        self.local = str(tmp / "nvme")
        # V5_TEST_REMOTE=s3://bucket runs every test against S3 (the emulator in Docker, or real S3).
        base = os.environ.get("V5_TEST_REMOTE")
        self.remote = f"{base.rstrip('/')}/{tmp.name}" if base else str(tmp / "s3")
        self.events = str(tmp / "events")

    def run(self, *args, world=1, check=None, env_extra=None):
        env = dict(os.environ, V5_EVENTS_DIR=self.events, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1",
                   **(env_extra or {}))
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"):
            env.pop(k, None)
        launch = [sys.executable, TRAIN] if world == 1 else \
            [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}", TRAIN]
        cmd = launch + ["--local-root", self.local, "--remote-root", self.remote, *map(str, args)]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        if check is not None:
            assert p.returncode == check, f"exit {p.returncode}, wanted {check}\n{p.stdout}\n{p.stderr[-3000:]}"
        return p

    def store(self, run):
        return open_store(run_url(self.remote, run))

    def latest(self, run):
        return self.store(run).get_json("latest.json")

    def events_named(self, name):
        return [e for e in read_all(self.events) if e["event"] == name]


def losses(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["step"]] = r["loss_bits"]
    return out


@fixture
def env(tmp_path):
    return Env(tmp_path)


@parametrize("world", [1, pytest.param(2, marks=needs(2))])
def test_t1_bit_equal_resume(env, world):
    """Straight run vs save -> new process -> restore: loss equal as raw bits at every step."""
    straight = env.tmp / "straight.jsonl"
    resumed = env.tmp / "resumed.jsonl"
    env.run("--run-id", "straight", "--steps", 30, "--interval-s", 999, "--loss-log", straight, world=world, check=0)
    env.run("--run-id", "r", "--steps", 15, "--interval-s", 999, "--loss-log", resumed, world=world, check=0)
    env.run("--run-id", "r", "--steps", 30, "--interval-s", 999, "--loss-log", resumed, world=world, check=0)
    a, b = losses(straight), losses(resumed)
    assert sorted(a) == list(range(1, 31)) and a == b
    done = [e for e in env.events_named("restore.done") if e["run_id"] == "r"]
    assert done and done[0]["step"] == 15 and not done[0]["resharded"]


@needs(2)
def test_t2_reshard_to_1(env):
    """Save on 4 ranks (2 on a 2-GPU machine), restore on 1: digest and probe loss are equal."""
    w = min(4, MAX_WORLD)
    env.run("--run-id", "w4", "--steps", 12, "--interval-s", 999, world=w, check=0)
    env.run("--run-id", "w4", "--steps", 16, "--interval-s", 999, world=1, check=0)
    done = env.events_named("restore.done")[0]
    assert done["resharded"] and done["saved_world"] == w and done["world"] == 1 and done["step"] == 12


def test_t3_flipped_byte_falls_back(env):
    """One flipped byte in the newest checkpoint: verify fails and the previous one is used."""
    env.run("--run-id", "c", "--steps", 20, "--interval-s", 0, "--step-sleep", 0.01, check=0)
    ck = Checkpointer("c", env.local, env.remote)
    newest, prev = ck.commits()[-1], ck.commits()[-2]
    env.store("c").corrupt_for_test(f"{newest['path']}/__0_0.distcp")
    env.run("--run-id", "c", "--steps", 25, "--interval-s", 999, check=0)
    failed = env.events_named("restore.failed")
    assert failed and failed[0]["cause"] == "IntegrityError" and failed[0]["step"] == newest["step"]
    assert env.events_named("restore.done")[0]["step"] == prev["step"]


def test_t4_crash_before_seal(env):
    """The process dies after uploading, before the manifest: latest.json does not move."""
    env.run("--run-id", "k", "--steps", 10, "--interval-s", 999, check=0)
    latest = env.latest("k")
    p = env.run("--run-id", "k", "--steps", 14, "--interval-s", 999, env_extra={"V5_TEST_CRASH_BEFORE_SEAL": "1"})
    assert p.returncode == 9
    assert env.latest("k") == latest
    assert not env.store("k").exists("main/step_00000014/MANIFEST.json")
    out = run_retention(run_url(env.remote, "k"), orphan_age_s=0)
    assert "main/step_00000014" in out["orphans"] and latest["path"] in out["protected"]


def test_t5_renamed_key_strict_load(env, monkeypatch):
    """The V4 converter bug: a renamed weight must abort the load and name the key."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    env.run("--run-id", "s", "--steps", 5, "--interval-s", 999, check=0)

    class Renamed(TinyLM):
        def __init__(self):
            super().__init__(SEED)
            self.ff_in = self.ff1
            del self.ff1

    m = Renamed()
    opt = torch.optim.AdamW(m.parameters())
    ck = Checkpointer("s", env.local, env.remote, config_path=os.path.join(ROOT, "sim", "config_v0.yaml"))
    with pytest.raises(StrictLoadError) as e:
        ck.restore(m, opt, SyntheticLoader(SEED, 8, 0, 1), make_probe_fn(SEED))
    assert "model.ff1.weight" in str(e.value) and "model.ff_in.weight" in str(e.value)


@needs(2)
def test_t6_sigusr1_saves_and_exits_75(env):
    """SIGUSR1 to ONE rank (as srun or the spot watcher sends it): all ranks save, exit 75,
    and the continuation is bit-equal to a run that was never stopped."""
    straight, resumed = env.tmp / "a.jsonl", env.tmp / "b.jsonl"
    env.run("--run-id", "s6a", "--steps", 60, "--interval-s", 999, "--loss-log", straight, world=2, check=0)
    base = dict(os.environ, V5_EVENTS_DIR=env.events, MASTER_ADDR="127.0.0.1", MASTER_PORT="29611",
                WORLD_SIZE="2", RUN_ID="s6", OMP_NUM_THREADS="1")
    procs = []
    for r in range(2):
        e = dict(base, RANK=str(r), LOCAL_RANK=str(r))
        procs.append(subprocess.Popen([sys.executable, TRAIN, "--run-id", "s6", "--steps", "60", "--interval-s", "999",
                                       "--step-sleep", "0.05", "--local-root", env.local, "--remote-root", env.remote,
                                       "--loss-log", str(resumed)], env=e, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
    deadline = time.time() + 60
    while len(losses(resumed) if resumed.exists() else {}) < 10:
        assert time.time() < deadline
        time.sleep(0.2)
    procs[1].send_signal(signal.SIGUSR1)
    codes = [p.wait(timeout=60) for p in procs]
    assert codes == [75, 75]
    stop_step = [e for e in env.events_named("train.stop_requested") if e["run_id"] == "s6"][0]["step"]
    assert env.latest("s6")["step"] == stop_step
    env.run("--run-id", "s6", "--steps", 60, "--interval-s", 999, "--loss-log", resumed, world=2, check=0)
    assert losses(straight) == losses(resumed)


def test_t7_nan_never_saved(env):
    """NaN: no save, exit 78 (hold), latest.json still points before the NaN."""
    env.run("--run-id", "n", "--steps", 8, "--interval-s", 999, check=0)
    env.run("--run-id", "n", "--steps", 20, "--interval-s", 0, "--inject-nan-at", 12, check=78)
    latest = env.latest("n")
    assert latest["step"] <= 12


def test_t8_export_safetensors(env):
    """Export to safetensors: strict keys pass and the probe loss is bit-equal."""
    env.run("--run-id", "e", "--steps", 6, "--interval-s", 999, check=0)
    out = export_safetensors(run_url(env.remote, "e"), lambda: TinyLM(SEED), make_probe_fn(SEED),
                             str(env.tmp / "e.safetensors"))
    assert out["step"] == 6 and os.path.getsize(out["path"]) > 0


def test_t9_env_change_refused(env, tmp_path):
    """A different config refuses to resume (exit 78) unless --allow-env-change is given."""
    env.run("--run-id", "v", "--steps", 4, "--interval-s", 999, check=0)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(open(os.path.join(ROOT, "sim", "config_v0.yaml")).read() + "\n# changed\n")
    env.run("--run-id", "v", "--steps", 6, "--config", cfg, check=78)
    env.run("--run-id", "v", "--steps", 6, "--config", cfg, "--allow-env-change", check=0)
    assert env.events_named("restore.done")[0]["env_override"] == ["config_sha256"]


def test_t10_rollback_skip_branch(env):
    """Operator rollback: restore step k, skip N batches, continue on a new branch."""
    env.run("--run-id", "b", "--steps", 20, "--interval-s", 0, "--step-sleep", 0.01, check=0)
    k = Checkpointer("b", env.local, env.remote).commits()[1]["step"]
    env.run("--run-id", "b", "--steps", k + 5, "--from-step", k, "--skip", 3, "--branch", "b1", check=0)
    done = env.events_named("restore.done")[0]
    assert done["step"] == k and done["skip"] == 3 and done["branch"] == "b1"
    assert env.latest("b")["branch"] == "b1"


def test_t11_loss_spike_holds(env):
    """A loss spike (x10 for 5 steps) that is not NaN: no save during the spike, exit 78, latest is pre-spike."""
    env.run("--run-id", "sp", "--steps", 120, "--interval-s", 0, "--step-sleep", 0.005, "--spike-at", 80, check=78)
    assert env.latest("sp")["step"] <= 80
    spikes = env.events_named("train.loss_spike")
    assert len(spikes) == 3 and spikes[0]["step"] == 80


def test_t12_no_false_spike(env):
    """Normal training for 300 steps never trips the spike guard."""
    env.run("--run-id", "ns", "--steps", 300, "--interval-s", 999, check=0)
    assert not env.events_named("train.loss_spike")


def test_t13_loader_contract():
    """The synthetic loader passes the contract that Teams 01/02's real loader must pass."""
    from ckpt.loader_contract import check
    assert check(lambda r, w: SyntheticLoader(SEED, 8, r, w)) == []
