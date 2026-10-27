"""Spot interruption watcher, one per node (ADR 0002 §5).

In simple words: AWS gives a spot instance a 2-minute warning before taking it back. We poll for that
warning every 5 s. When it comes, we drain the node and ask its jobs to save now (SIGUSR1). The job
saves, exits 75, and Slurm restarts it on another node: we lose seconds of work, not the whole
interval since the last checkpoint.

On EC2 it reads the instance metadata service (IMDSv2). In the Docker test it reads a file.
"""
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.events import emit  # noqa: E402
from health import slurm_actions  # noqa: E402

IMDS = os.environ.get("V5_IMDS_URL")          # e.g. http://169.254.169.254 on EC2
NOTICE_FILE = os.environ.get("V5_SPOT_NOTICE_FILE", "/var/lib/v5-health/spot_notice")
POLL_S = float(os.environ.get("V5_SPOT_POLL_S", "5"))


def notice() -> str:
    if IMDS:
        try:
            tok_req = urllib.request.Request(f"{IMDS}/latest/api/token", method="PUT",
                                             headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
            tok = urllib.request.urlopen(tok_req, timeout=2).read().decode()
            req = urllib.request.Request(f"{IMDS}/latest/meta-data/spot/instance-action",
                                         headers={"X-aws-ec2-metadata-token": tok})
            return urllib.request.urlopen(req, timeout=2).read().decode()   # 404 = no notice
        except Exception:  # noqa: BLE001 - 404 and timeouts both mean "no notice yet"
            return ""
    if os.path.exists(NOTICE_FILE):
        return open(NOTICE_FILE).read().strip() or "spot notice (drill)"
    return ""


def main():
    os.environ.setdefault("V5_EVENTS_DIR", "/s3/events")
    handled = False
    while True:
        n = notice()
        if n and not handled:
            emit("node.spot_notice", node=slurm_actions.node_name(), detail=n[:200])
            slurm_actions.drain("spot_notice")
            slurm_actions.ask_jobs_to_save_and_stop(why="spot_notice")
            handled = True
        elif not n:
            handled = False
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
