#!/bin/bash
# Slurm HealthCheckProgram: runs every HealthCheckInterval on every node (ADR 0003).
# Slurm ignores this script's exit code, so health_check.py drains the node itself on "fail" and asks
# the jobs on it to save and stop (they exit 75 and requeue onto a healthy node).
source /etc/v5/node.env 2>/dev/null || true
export V5_EVENTS_DIR="${V5_EVENTS_DIR:-/var/log/v5/events}"
/opt/venv/bin/python /opt/v5/submission/health/health_check.py --mode periodic > /var/log/slurm/health-last.json 2>&1
exit 0
