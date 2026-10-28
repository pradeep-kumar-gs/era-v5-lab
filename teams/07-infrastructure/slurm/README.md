# Job queue (`slurm/` and `slurm-docker/`): who runs where, and how the main run restarts by itself

Decision record: [ADR 0002](../docs/adr/0002-job-queue-slurm.md). Checkpoint side:
[../ckpt/README.md](../ckpt/README.md). Health side: [../health/README.md](../health/README.md).
Simulation results: [sim_results.md](../docs/sim_results.md).

> **In one paragraph.** A scheduler is the "front desk" of the cluster. Teams hand in jobs; it decides who runs
> where and when; it keeps the main run first; and when something breaks it puts the main run back in the queue
> so it restarts from the last checkpoint **without anyone typing a command**. We use **Slurm**, the standard for
> training clusters. The instructor asked us to prove it in a **Docker copy of the cluster** first, so we do
> not pay for GPUs while we debug the queue. The Docker copy uses **the same `slurm.conf`** as AWS.

---

## 1. Why Slurm, and why on our own EC2 instances

V4 had **no scheduler**: GPUs from many student accounts, jobs started by hand, a cluster idle for 3 days, and
restarts typed by hand.

| Option | Why not / why |
|---|---|
| No scheduler (V4) | no queue, no fairness, no automatic restart |
| Kubernetes (EKS) | multi-node GPU training needs add-ons; many extra layers |
| AWS ParallelCluster | its own config layer between us and Slurm; harder to copy in Docker. **Our fallback** if running Slurm ourselves is too much work |
| AWS Batch | weak for multi-node jobs, no fair-share |
| **Self-managed Slurm on EC2** ✅ | full control; the norm for LLM training; **the Docker test runs the same file** |

---

## 2. The cluster at a glance

```text
                     ┌──────────────── head (small CPU instances) ────────────────┐
  teams ── sbatch ─► │ slurmctld (primary)   slurmctld2 (backup)   slurmdbd + MariaDB │
  (via svc user)     └───────────────┬────────────────────────────────────────────┘
                                     │ schedules
          ┌──────────────────────────┼───────────────────────────┐
          ▼                          ▼                           ▼
   gpu-1  gpu-2  … (main run)     gpu-N (hot spare:        eval / cpu nodes
   8 × B200 each, EFA             runs team jobs that       (evaluation, data,
                                  the main run can take)    retention, export)
                                     │
                          checkpoints, logs, events ──► S3
```

- **Main run** on on-demand or a Capacity Block, **never spot**, plus **one hot spare** node.
- The **hot spare is not idle**: it runs team jobs that the main run preempts when it needs the node.
- **Idle nodes are stopped** (`State=CLOUD`, `SuspendTime=600`), except exactly one spare. This replaces V4's
  "idle alarm".
- **GPUs are a counted resource (gres)**. A job that asks `--gpus-per-node=8` gets exactly 8, sees only those
  (`CUDA_VISIBLE_DEVICES`), and its GPU-hours are recorded against its account.

### Partitions (queues)

| Partition | For |
|---|---|
| `gpu` | training, whole nodes only |
| `eval` | cheaper GPU nodes for evaluation (never the training nodes) |
| `debug` | 30-minute limit |
| `cpu` | data, tokenizer, retention and export jobs |

---

## 3. Who gets the GPUs: accounts, priority, budgets

```text
root
├── mainrun   QOS=mainrun  Priority 1000, can preempt "normal", no time limit
└── teams     QOS=normal   Priority 10, preemptible (60 s grace to save, then requeued)
    ├── data  tokenizer  ke2  arch  perf  infra  monitoring  eval  posttrain  release  quant
```

In simple words:

1. **Every job is charged to an account.** A job with no valid account is **rejected**.
2. **The main run outranks everyone.** If it needs a node, a team job on that node gets SIGTERM, has 60 s to
   save, and goes back into the queue.
3. **Teams share the rest fairly** (fair-share): a team that used a lot recently waits behind one that used little.
4. **No team can hold more than 2 nodes** (`GrpTRES=node=2`).
5. **Each team has a GPU-hour budget** (`GrpTRESMins`). Alerts at 50 / 75 / 90%, hard stop at 100%.

Students never get SSH or AWS keys. The submit system calls `sbatch` with a team account through one service
user (`svc`).

```bash
# a team job (scaling ladder, benchmark, eval)
sbatch --account=arch --export=ALL,RUN_ID=ladder-1 slurm/team_job.sbatch
# the main run (one script for the first start and every restart)
sbatch slurm/mainrun.sbatch
```

---

## 4. The restart contract (the most important part)

```text
 trainer exits ──► run_job.py (wrapper) ──► exit code ──► Slurm
     0  done                                  0  ─────────► COMPLETED
     75 saved, run me again                   75 ─────────► requeue (RequeueExit=75) ─► restarts from latest.json
     78 NaN / spike / failed checks           78 ─────────► HOLD    (RequeueExitHold=78) ─► page a person
     1, 137, 139… (crash)                     75, unless 3 crashes in 30 min → 78
```

- **One script** (`mainrun.sbatch`) for every start. It always resumes from `latest.json`, so a restart needs no
  human input.
- **The wrapper** (`run_job.py`) maps every trainer outcome to 0, 75 or 78, applies the **crash-loop cap**
  (3 crashes in 30 min → hold: retrying a broken job just burns money), and writes `job.started / ended /
  requeued / held` events.
- **Signals.** The trainer treats SIGUSR1 and SIGTERM the same: save, wait for the commit, exit 75.

| Who sends it | When | What the job loses |
|---|---|---|
| Slurm (SIGTERM) | a team job is preempted by the main run | ~0 (it saves in its 60 s grace) |
| `spot_watcher.py` (SIGUSR1) | AWS 2-minute spot warning | ~0 |
| health check (SIGUSR1) | the node is sick | ~0 |
| nobody (no warning) | process killed, node lost | the work since the last checkpoint |

- A node that stops answering for **30 s** (`SlurmdTimeout`) is marked DOWN and its job is requeued.

### Why a restart takes ~50 s

Slurm will not restart a requeued job until the old job's launch credential has expired (`cred_expire` + 1 s;
the default is ~121 s). We set **`AuthInfo=cred_expire=30`**: measured ~52 s from exit to running, every time.

> **Lesson from simulation run 7.** We first forced an immediate restart (~4 s). Sometimes the job started
> while the old credential was still valid; Slurm killed it ("Job credential revoked") and marked it
> **FAILED**: the main run was lost. We removed that trick. ~50 s per restart (~$13 on 8 nodes at assumed
> prices) is the price of never losing the run.

### When there are not enough nodes

If more nodes fail than there are spares, the main run **waits** (pending), and alert Q-01 pages after 10 min.
It does **not** shrink by itself: fewer GPUs change speed and are not bit-equal, so a person decides. The
operator can restart it on fewer nodes (`sbatch --nodes=1 …`); the rollback drill tests this.

---

## 5. Keeping the control plane alive

| Risk | Protection | Tested |
|---|---|---|
| controller (`slurmctld`) dies | backup `slurmctld2` takes over (shared state on EFS); running jobs keep training | yes: takeover in 82 s, training never stopped |
| accounting database lost | `backup_db.sh` dumps it daily to S3; a backup **counts only after a test restore** with the same tables | yes: 31 of 31 tables restored |
| software upgrade mid-run | only at a checkpoint: emergency save → roll nodes → requeue with `--allow-env-change` (logged). Slurm order: `slurmdbd` → `slurmctld` → `slurmd` | written down (ADR 0002 §11) |

---

## 6. Files

### `slurm/` (production; the Docker test uses the same files where they match)

| File | What it does |
|---|---|
| `slurm.conf` | **shared** by Docker and EC2: controllers, accounting, priorities, preemption, requeue codes, health hooks, gres |
| `nodes.ec2.conf` | EC2 node and partition lines (`p6-b200`, `State=CLOUD`, one spare kept up) |
| `site.ec2.conf`, `cgroup.ec2.conf`, `gres.ec2.conf` | EC2 only: cgroup enforcement (a job uses only its own cores, memory, GPUs), GPU discovery through NVML |
| `mainrun.sbatch` | the main-run job: account `mainrun`, 2 nodes × 8 GPUs, `--requeue`, sets the deterministic env (`OMP_NUM_THREADS=1`) and the NCCL flight recorder |
| `team_job.sbatch` | a preemptible team job |
| `run_job.py` | the wrapper: exit-code mapping, crash-loop cap, job events |
| `prolog.sh` | quick health check before a job starts on a node (see health README) |
| `epilog.sh` | cleans temporary restore files on NVMe after a job |
| `healthcheck.sh` | Slurm's `HealthCheckProgram`, every 30 s |
| `spot_watcher.py` | polls the AWS spot warning (IMDSv2) every 5 s; drains the node and asks jobs to save |
| `setup_accounts.sh` | creates accounts, QOS, limits and the `svc` user |
| `backup_db.sh` | dump and `--verify-latest` the accounting database |

### `slurm-docker/` (the test cluster on a laptop)

| File | What it does |
|---|---|
| `docker-compose.yml` | MariaDB, `slurmdbd`, `slurmctld` + `slurmctld2`, S3 emulator (`s3emu`, moto), nodes `c1–c3` ("GPU") and `c4` (eval/CPU) |
| `Dockerfile` | Ubuntu 24.04, Slurm 23.11, torch 2.5.1 CPU, boto3, moto |
| `entrypoint.sh` | starts each role; creates placeholder GPU device files `/dev/v5gpu0-1` |
| `nodes.conf`, `gres.conf`, `site.conf`, `cgroup.conf` | the Docker versions of the four files that differ from EC2 |
| `slurmdbd.conf` | accounting daemon config (test-only password) |

**Docker vs EC2, the only differences:** CPU instead of GPU (gloo instead of NCCL); 2 placeholder GPUs per node
(allocation and accounting only, no computation); no cgroup enforcement; no Pyxis containers; no power saving;
no EFA. Everything else (requeue, preemption, accounts, budgets, health drain, failover, S3 code path) is the
real thing.

---

## 7. Run it yourself

From `teams/07-infrastructure`, with Docker Desktop running:

```bash
docker compose -f slurm-docker/docker-compose.yml build slurmdbd   # once, ~5 min
python3 sim/run_sim.py                    # every drill, ~25 min; ends with "N/N drills passed"
python3 sim/run_sim.py --keep-up          # same, but leave the cluster running to poke at it
```

With the cluster up, open a shell on the controller
(`docker compose -f slurm-docker/docker-compose.yml exec slurmctld bash`) and try:

```bash
sinfo                                  # nodes and their state (idle / alloc / drain / down)
squeue                                 # what is running and pending, and why pending
scontrol show job <id>                 # one job: restarts, reason, allocated GPUs (AllocTRES)
sacct -a -X -D -o jobid,account,state,elapsed,alloctres   # history incl. every requeue (-D)
sshare -a                              # fair-share per account
sacctmgr show assoc format=account,user,qos,grptres,grptresmins
scontrol ping                          # which controller is in charge
```

---

## 8. What the simulation proved (run 8, 30 of 30 drills)

| Drill | What happened | Recovery time (MTTR p50) | Work lost |
|---|---|---|---|
| kill a training process | wrapper → 75 → requeued → restored | 50 s (3 runs) | 24–45 steps |
| preemption signal | emergency save, 75, requeued | 58 s (3 runs) | 0 |
| spot notice | node drained, save, moved to the spare | 57 s | 0 |
| health fail | drained within one interval, save, moved | 60 s | 0 |
| frozen process (hang) | watchdog at 20 s → 75 → restart | 56 s | 44 steps |
| node unplugged | DOWN after ~30 s, requeued on remaining nodes | 88 s | 24 steps |
| two nodes down, one spare | main run waits (no crash loop), recovers on repair | 58 s after repair | — |
| controller stopped | backup in control in 82 s | — | 0 |
| team jobs vs main run | team jobs saved and requeued; main run started | — | — |
| no account | rejected | — | — |
| GPU allocation | `gres/gpu=4` allocated; over-asking refused | — | — |
| crash loop | requeued twice, then held | — | — |
| budget exceeded | next job pending with the budget reason | — | — |
| fair share | heavy account gets a lower factor | — | — |
| DB backup | 31 of 31 tables restored | — | — |

Low "goodput" in the simulation report is **by design**: we inject a fault every ~30 s, while a real run sees
one every few hours. What matters is the time of each step (detect, reschedule, restore), which
`cost/cost_model.py --measured` projects onto real failure rates.

---

## 9. Metrics to watch

Full list: Q-01 … Q-21 in [metrics_catalog.md](../docs/metrics_catalog.md). These page someone now (HIGH):

| ID | Metric | HIGH when | What to do |
|---|---|---|---|
| Q-01 | main run not running | pending ≥ 10 min | `squeue` reason: no nodes? budget? repair or add a node |
| Q-02 | main run held | any | read the 78 cause (NaN, spike, crash loop); roll back or fix |
| Q-03 | reschedule time | ≥ 10 min | spare missing or prolog failing |
| Q-05 | interruptions | ≥ 10 per day | find the common cause in the incident records |
| Q-06 | idle GPU node-hours | ≥ 4 per day | power saving not working; nodes paid for nothing |
| Q-07 | allocated but not stepping | ≥ 15 min | stuck in restore or data wait |
| Q-09 | drained nodes | ≥ 3 | hardware wave: replace nodes |
| Q-10 | node replacement time | ≥ 15 min | AWS capacity, AMI start |
| Q-11 | slurmctld down | ≥ 60 s | check the backup took over |
| Q-15 | account budget used | ≥ 90% | talk to the team and the lead |
| Q-17 | clock offset | ≥ 500 ms | chrony on the node |
| Q-19 | goodput | ≤ 80% | look at the waste buckets |
| Q-20 | MTTR | ≥ 60 min | which restart step is slow |

---

## 10. Still open

- GPU type and count; Capacity Block or on-demand; real prices.
- AWS account: IAM roles, bucket names, EFA, Pyxis/Enroot, power-saving scripts (`ec2_suspend.sh` /
  `ec2_resume.sh` are week-2 work).
- The submit system's contract (does it call `sbatch` with an account?).
- On-call rota: who acts on a HIGH alert at 4 a.m.
- Capacity calendar for weeks 2–4.

---

## 11. Words used here

| Word | Meaning |
|---|---|
| `slurmctld` | the controller: the brain that schedules |
| `slurmd` | the agent on each node that runs jobs |
| `slurmdbd` | accounting: accounts, budgets, job history |
| partition | a queue of nodes |
| QOS | quality of service: priority and preemption rules |
| preempt | take a node from a lower-priority job (it saves first) |
| requeue | put a job back in the queue to start again |
| hold | keep a job in the queue but do not start it until a person releases it |
| drain | "no new jobs on this node" |
| gres | a counted resource such as GPUs |
| fair-share | priority goes down the more an account used recently |
| MTTR | mean time to recover: fault → training again |
| goodput | share of paid time that trained the model |
