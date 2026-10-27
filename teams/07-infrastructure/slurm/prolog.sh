#!/bin/bash
# Slurm Prolog: quick health check before a job starts on this node (ADR 0003 §1).
# Exit non-zero only on "fail": Slurm then drains the node and puts the job back in the queue
# (SchedulerParameters=nohold_on_prolog_fail), and it starts on a healthy node instead.
source /etc/v5/node.env 2>/dev/null || true
export V5_EVENTS_DIR="${V5_EVENTS_DIR:-/s3/events}" SLURM_JOB_ID
/opt/venv/bin/python /opt/v5/health/health_check.py --mode prolog \
  > "/var/log/slurm/prolog-${SLURM_JOB_ID}.json" 2>&1
