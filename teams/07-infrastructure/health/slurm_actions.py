"""What a node does to itself when it is sick or about to disappear (ADR 0002 §6, ADR 0003).

In simple words: "drain" tells Slurm to give this node no new jobs. Then we ask the jobs already on
it to save and stop (SIGUSR1). They exit 75, Slurm puts them back in the queue, and they restart on a
healthy node (the hot spare) from the checkpoint they just wrote. Nothing paid for is lost.
"""
import os
import shutil
import socket
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.events import emit  # noqa: E402


def _run(cmd):
    if not shutil.which(cmd[0]):
        return None
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def node_name() -> str:
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname()


def drain(reason: str, node: str = None) -> bool:
    node = node or node_name()
    p = _run(["scontrol", "update", f"nodename={node}", "state=drain", f"reason=v5:{reason}"[:250]])
    ok = p is not None and p.returncode == 0
    emit("node.drained", node=node, reason=reason, ok=ok)
    return ok


def jobs_on(node: str = None) -> list:
    p = _run(["squeue", "-h", "-w", node or node_name(), "-t", "RUNNING", "-o", "%A"])
    if p is None or p.returncode != 0:
        return []
    return [j for j in p.stdout.split() if j.strip()]


def ask_jobs_to_save_and_stop(node: str = None, why: str = "") -> list:
    jobs = jobs_on(node)
    for j in jobs:
        _run(["scancel", "--signal=USR1", j])
        emit("job.signalled", node=node or node_name(), signalled_job=j, signal="USR1", why=why)
    return jobs
