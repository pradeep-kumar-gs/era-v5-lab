"""Job wrapper used by every training job script (ADR 0002 §5).

In simple words: Slurm only understands "requeue" or "hold". This wrapper turns everything that can
happen to a training process into one of three answers:
    0  finished
    75 run me again (Slurm requeues: RequeueExit=75)
    78 stop and call a person (Slurm holds: RequeueExitHold=78)
A crash we do not understand (exit 1, killed, segfault) is retried, because most crashes at scale are
one-off hardware or network faults. But if it crashes 3 times in 30 minutes, it is not one-off: we hold
the job so it does not burn GPU hours in a loop.

It also says WHY, in plain words, where people look: the Slurm job comment (`squeue -o "%i %T %k"`,
`scontrol show job <id>`), a banner in the job log, and the job.* events (fields `title`, `you_do`). The words
come from common/reasons.py; the trainer leaves a short exit note (common.reasons.write_exit_note) saying why
it stopped.

Usage (inside sbatch):  python run_job.py -- <srun args> -- <train.py args>
The trainer is V5_TRAIN_SCRIPT (default: the tiny test trainer in simulation/trainer/train.py).
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
from common.reasons import EXIT_CODES, EXIT_DONE, EXIT_FATAL, EXIT_RETRY, RAW_EXIT, lookup, read_exit_notes  # noqa: E402

TRAIN_SCRIPT = os.environ.get("V5_TRAIN_SCRIPT", os.path.join(HERE, "..", "simulation", "trainer", "train.py"))
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


def set_comment(text: str) -> None:
    """Put the plain-words status on the Slurm job, so `squeue` and `scontrol show job` show it."""
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        try:
            subprocess.run(["scontrol", "update", f"JobId={jid}", f"Comment={text.replace(chr(34), chr(39))[:250]}"],
                           capture_output=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            pass  # the comment helps people; it must never change the exit code


def explain(out: int, cause: str, rc: int, note: dict, restart: int) -> dict:
    r = lookup(cause)
    step = note.get("step")
    where = f" at step {step}" if step is not None else ""
    raw = f" (trainer exit {rc}: {RAW_EXIT[rc]})" if rc in RAW_EXIT and cause in ("crash", "crash_loop") else ""
    head = f"V5 {EXIT_CODES[out]['title']} ({out}): {r['title']}{where}{raw}"
    todo = "nothing" if out == EXIT_RETRY and r["severity"] in ("none", "low") else r["you_do"]
    return {"title": head, "meaning": r["meaning"], "auto": r["auto"], "you_do": todo,
            "severity": r["severity"], "alert": r.get("alert"), "detail": note.get("detail", "")}


def banner(x: dict) -> None:
    lines = ["", "=" * 100, x["title"], f"What happened : {x['meaning']}",
             f"Already done  : {x['auto']}", f"What you do   : {x['you_do']}"]
    if x["detail"]:
        lines.append(f"Detail        : {x['detail']}")
    lines += ["More          : python metrics/explain.py <cause or code>  (metrics/README.md)", "=" * 100, ""]
    print("\n".join(lines), file=sys.stderr, flush=True)


def main(argv):
    sep = [i for i, a in enumerate(argv) if a == "--"]
    srun_args, train_args = argv[sep[0] + 1:sep[1]], argv[sep[1] + 1:]
    run_id = os.environ.get("RUN_ID", "run")
    run_dir = os.path.join(os.environ.get("V5_RUNS_DIR", "/s3/runs"), run_id)
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    emit("job.started", restart_count=restart, nodes=os.environ.get("SLURM_JOB_NODELIST", ""),
         account=os.environ.get("SLURM_JOB_ACCOUNT", ""), qos=os.environ.get("SLURM_JOB_QOS", ""))
    set_comment(f"V5 RUNNING (start #{restart + 1}): restoring from latest.json")

    cmd = ["srun", "--kill-on-bad-exit=1", *srun_args, sys.executable, TRAIN_SCRIPT, *train_args]
    child = subprocess.Popen(cmd)

    # Preemption sends SIGTERM to the batch shell too. Keep waiting: the trainers got the same signal,
    # they are saving, and we must report their exit code, not ours.
    def forward(signum, _frame):
        emit("job.signal", signal=signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGUSR1, forward)
    rc = child.wait()

    notes = read_exit_notes(run_id)
    note = notes[0] if notes and notes[0].get("code") == rc else {}
    if rc in (EXIT_DONE, EXIT_RETRY, EXIT_FATAL):
        out = rc
        cause = note.get("cause") or {0: "done", 75: "saved_and_stopped", 78: "unknown_fatal"}[rc]
    else:
        loop, n = crash_loop(run_dir)
        out = EXIT_FATAL if loop else EXIT_RETRY
        cause = "crash_loop" if loop else "crash"
        if loop:
            note = dict(note, detail=f"{n} crashes in {int(CRASHLOOP_WINDOW_S)} s; last trainer exit {rc}")
    x = explain(out, cause, rc, note, restart)
    banner(x)
    set_comment(f"{x['title']}. Explain: python metrics/explain.py {cause}. Do: {x['you_do']}")
    plain = {k: x[k] for k in ("title", "you_do", "severity")}
    emit("job.ended", trainer_exit=rc, exit=out, cause=cause, restart_count=restart, step=note.get("step"), **plain)
    if out == EXIT_RETRY:
        emit("job.requeued", cause=cause, restart_count=restart, **plain)
    elif out == EXIT_FATAL:
        emit("job.held", cause=cause, restart_count=restart, step=note.get("step"), **plain)
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
