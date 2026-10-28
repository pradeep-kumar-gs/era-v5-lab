"""The queue file (submission/queue/job_queue.py) and the paths the cluster uses (ADR 0002)."""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(TEAM, "submission"), os.path.join(TEAM, "submission", "queue")]
import job_queue  # noqa: E402


def test_crash_loop_holds_on_the_third_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(job_queue, "CRASHLOOP_WINDOW_S", 60)
    got = [job_queue.crash_loop(str(tmp_path)) for _ in range(3)]
    assert got == [(False, 1), (False, 2), (True, 3)]


def test_spot_notice_from_drill_file(tmp_path, monkeypatch):
    f = tmp_path / "spot_notice"
    monkeypatch.setattr(job_queue, "IMDS", None)
    monkeypatch.setattr(job_queue, "NOTICE_FILE", str(f))
    assert job_queue.spot_notice() == ""
    f.write_text("")
    assert job_queue.spot_notice() == "spot notice (drill)"


def test_without_slurm_nothing_crashes(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("V5_EVENTS_DIR", str(tmp_path))
    monkeypatch.setattr(job_queue.shutil, "which", lambda name: None)
    assert job_queue.drain("drill_force_fail", "c1") is False      # reported as an event, not an exception
    assert job_queue.ask_jobs_to_save_and_stop("c1") == []
    assert job_queue.status() == 1 and "squeue is not available" in capsys.readouterr().out


def test_default_trainer_exists():
    assert os.path.exists(job_queue.TRAIN_SCRIPT), job_queue.TRAIN_SCRIPT


def test_every_cluster_path_exists():
    """/opt/v5 is the team folder in every container. A moved file must not leave a dangling path behind."""
    files = glob.glob(os.path.join(TEAM, "submission", "slurm", "*")) + \
        glob.glob(os.path.join(TEAM, "support", "simulation", "slurm-docker", "*")) + \
        [os.path.join(TEAM, "support", "simulation", "run_sim.py")]
    planned = {"submission/slurm/ec2_suspend.sh", "submission/slurm/ec2_resume.sh"}   # week 2, need AWS
    missing = set()
    for f in files:
        for p in re.findall(r"/opt/v5/([\w./-]+\.(?:py|sh|sbatch|conf))", open(f).read()):
            if p not in planned and not os.path.exists(os.path.join(TEAM, p)):
                missing.add(f"{os.path.basename(f)}: /opt/v5/{p}")
    assert not missing, sorted(missing)
