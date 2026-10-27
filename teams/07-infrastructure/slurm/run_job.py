"""Job wrapper used by every training job script (ADR 0002 §5).

In simple words: Slurm only understands "requeue" or "hold". This wrapper turns everything that can
happen to a training process into one of three answers:
    0  finished
    75 run me again (Slurm requeues: RequeueExit=75)
    78 stop and call a person (Slurm holds: RequeueExitHold=78)
A crash we do not understand (exit 1, killed, segfault) is retried, because most crashes at scale are
one-off hardware or network faults. But if it crashes 3 times in 30 minutes, it is not one-off: we hold
the job so it does not burn GPU hours in a loop.

Usage (inside sbatch):  python run_job.py -- <srun args> -- <train.py args>
"""
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.events import emit  # noqa: E402

EXIT_DONE, EXIT_RETRY, EXIT_FATAL = 0, 75, 78
CRASHLOOP_N = int(os.environ.get("CRASHLOOP_N", "3"))
CRASHLOOP_WINDOW_S = float(os.environ.get("CRASHLOOP_WINDOW_S", "1800"))


def crash_loop(run_dir: str) -> tuple:
    """Record this crash; return (is_loop, crashes_in_window)."""
    os.makedirs(run_dir, exist_ok=True)
    p = os.path.join(run_dir, "crashes.jsonl")
    now = time.time()
    with open(p, "a") as f:
        f.write(json.dumps({"time": now, "job_id": os.environ.get("SLURM_JOB_ID")}) + "\n")
    with open(p) as f:
        recent = [r for r in map(json.loads, f) if now - r["time"] <= CRASHLOOP_WINDOW_S]
    return len(recent) >= CRASHLOOP_N, len(recent)


def main(argv):
    sep = [i for i, a in enumerate(argv) if a == "--"]
    srun_args, train_args = argv[sep[0] + 1:sep[1]], argv[sep[1] + 1:]
    run_id = os.environ.get("RUN_ID", "run")
    run_dir = os.path.join(os.environ.get("V5_RUNS_DIR", "/s3/runs"), run_id)
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    emit("job.started", restart_count=restart, nodes=os.environ.get("SLURM_JOB_NODELIST", ""),
         account=os.environ.get("SLURM_JOB_ACCOUNT", ""), qos=os.environ.get("SLURM_JOB_QOS", ""))

    cmd = ["srun", "--kill-on-bad-exit=1", *srun_args, sys.executable, os.path.join(HERE, "..", "trainer", "train.py"),
           *train_args]
    child = subprocess.Popen(cmd)

    # Preemption sends SIGTERM to the batch shell too. Keep waiting: the trainers got the same signal,
    # they are saving, and we must report their exit code, not ours.
    def forward(signum, _frame):
        emit("job.signal", signal=signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGUSR1, forward)
    rc = child.wait()

    if rc in (EXIT_DONE, EXIT_RETRY, EXIT_FATAL):
        out, cause = rc, {0: "done", 75: "saved_and_stopped", 78: "fatal"}[rc]
    else:
        loop, n = crash_loop(run_dir)
        out = EXIT_FATAL if loop else EXIT_RETRY
        cause = f"crash_loop({n} in {int(CRASHLOOP_WINDOW_S)}s)" if loop else "crash"
    emit("job.ended", trainer_exit=rc, exit=out, cause=cause, restart_count=restart)
    if out == EXIT_RETRY:
        emit("job.requeued", cause=cause, restart_count=restart)
    elif out == EXIT_FATAL:
        emit("job.held", cause=cause, restart_count=restart)
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
