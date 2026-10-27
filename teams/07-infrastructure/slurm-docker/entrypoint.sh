#!/bin/bash
# Starts one role of the test cluster: slurmdbd | slurmctld | slurmd.
set -euo pipefail
role="${1:?role: s3emu|slurmdbd|slurmctld|slurmd}"

# The S3 emulator (moto): stands in for the S3 bucket, so the real S3 code path (boto3) is tested.
if [[ "$role" == "s3emu" ]]; then
  moto_server -H 0.0.0.0 -p 9000 &
  until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000')" 2>/dev/null; do sleep 1; done
  AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test python3 -c "import boto3; \
boto3.client('s3', endpoint_url='http://127.0.0.1:9000', region_name='us-east-1').create_bucket(Bucket='v5-ckpt')"
  wait
fi

# The munge key is shared through a volume. The controller creates it once; the others wait for it.
if [[ "$role" == "slurmdbd" && ! -s /etc/munge/munge.key ]]; then
  dd if=/dev/urandom of=/etc/munge/munge.key bs=1 count=1024 status=none
fi
until [[ -s /etc/munge/munge.key ]]; do sleep 1; done
chown munge:munge /etc/munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key
mkdir -p /run/munge && chown munge:munge /run/munge
runuser -u munge -- /usr/sbin/munged --force

# Per-node settings read by the prolog, health check and job scripts (on EC2: written by the AMI build).
mkdir -p /etc/v5
printf '%s\n' "export V5_EVENTS_DIR=/s3/events" "export V5_EXPECT_GPUS=0" \
  "export V5_NVME_MIN_FREE_GB=1" "export V5_MEM_MIN_GB=0.2" "export V5_DEVICE=cpu" \
  "export V5_S3_ENDPOINT=http://s3emu:9000" "export AWS_ACCESS_KEY_ID=test" "export AWS_SECRET_ACCESS_KEY=test" \
  "export AWS_REGION=us-east-1" "export V5_REMOTE_ROOT=s3://v5-ckpt/ckpt" > /etc/v5/node.env

# Slurm reads its config from /etc/slurm; the files come from the repo, read-only.
mkdir -p /etc/slurm
cp /opt/v5/slurm/slurm.conf /opt/v5/slurm-docker/nodes.conf /opt/v5/slurm-docker/cgroup.conf \
   /opt/v5/slurm-docker/site.conf /opt/v5/slurm-docker/gres.conf /etc/slurm/
install -o slurm -g slurm -m 600 /opt/v5/slurm-docker/slurmdbd.conf /etc/slurm/slurmdbd.conf

case "$role" in
  slurmdbd)
    until mariadb-admin ping -h mariadb -u slurm -psimonly --silent; do sleep 1; done
    exec runuser -u slurm -- /usr/sbin/slurmdbd -D
    ;;
  slurmctld)
    until (echo > /dev/tcp/slurmdbd/6819) 2>/dev/null; do sleep 1; done
    # The cluster must exist in the accounting database before slurmctld registers.
    sacctmgr -i add cluster v5sim >/dev/null 2>&1 || true
    exec runuser -u slurm -- /usr/sbin/slurmctld -D -i
    ;;
  slurmd)
    until (echo > /dev/tcp/slurmctld/6817) 2>/dev/null; do sleep 1; done
    # The spot-notice watcher runs on every node (ADR 0002 §5). In the test it reads a file, not EC2 IMDS.
    nohup python3 /opt/v5/slurm/spot_watcher.py >/var/log/slurm/spot_watcher.log 2>&1 &
    touch /dev/v5gpu0 /dev/v5gpu1          # placeholder GPU devices for gres.conf (Docker test only)
    # No systemd in the container (cgroup.conf IgnoreSystemd=yes): make the slice slurmd expects.
    mkdir -p /sys/fs/cgroup/system.slice
    exec /usr/sbin/slurmd -D -N "$(hostname)"
    ;;
esac
