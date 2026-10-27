# ADR 0002: Job queue and cluster operations on self-managed Slurm (EC2), validated in Docker Slurm

- Status: Proposed
- Date: 2026-10-27 (revised from 2026-10-25; second revision after the week-1 review, see [review_week1.md](../review_week1.md))
- Area: Queue (`slurm/`, `slurm-docker/`, `sim/run_sim.py`), cluster image, storage, access, events, goodput
- Related: ADR 0001 (checkpoint, exit codes), ADR 0003 (health, drain), ADR 0004 (cost, metrics)

> **In simple words.** A scheduler is the "front desk" of the cluster: teams hand in jobs, it decides who
> runs where and when, and it restarts the main run by itself when something breaks. We use Slurm, the
> standard for training clusters. The instructor asked us to prove it works in a Docker copy of the
> cluster first, so we do not pay for GPUs while we debug the queue.

## Context

V4 had **no scheduler**. GPUs were reached through many student AWS accounts and jobs were started by hand.
One cluster sat idle for 3 days; the only guard was a CloudWatch CPU-idle alarm. Restarts were typed by hand.
V4 also ran two different software stacks (cu128 pins, and a cu130 nightly plus an S3 tarball).

For V5 the instructor gives **no keys to students**, wants scheduling on **Slurm, validated in Docker Slurm**,
and we will run on **AWS EC2 GPU instances**. The cluster must:

- run the **main run** without letting anything else take its GPUs;
- share the remaining capacity fairly: Team 05's scaling ladder (5–8 runs), Team 06 benchmarks, Team 09 evals,
  Team 12 quantization, Team 10 RL tests, Teams 01–03 CPU data jobs;
- **restart the main run on its own** from the last committed checkpoint (ADR 0001);
- keep sick nodes out of service (ADR 0003);
- account for every GPU-hour and every dollar (ADR 0004).

Team 07 also runs the **two dry runs** (S15 team charter). The S15 reference run lost 9 of its 38 interruptions
to slow storage reads.

## Decision drivers

1. Automatic requeue and restart, with no person typing a command.
2. Hard priority for the main run, fair share for everyone else, and **no idle paid GPUs**.
3. One pinned software stack on every node.
4. Storage that never stalls training.
5. Every lost minute is measured and priced (goodput accounting).
6. Testable on a laptop in week 1, **in Docker Slurm, with the same configuration as production**.

## Options considered

| Option | For | Against |
|---|---|---|
| A. No scheduler, as in V4 (SSH + scripts) | nothing to run | no queue, no fairness, no automatic restart |
| B. Kubernetes (EKS) + Kueue or Volcano | strong for services | gang scheduling and topology for multi-node NCCL jobs are add-ons; extra layers (CNI, device and EFA plugins) |
| C. AWS ParallelCluster (managed Slurm) | AWS-built autoscaling, FSx and EFA wiring | its config layer and update cycle sit between us and Slurm; harder to mirror in Docker |
| D. AWS Batch | managed | weak multi-node gang jobs; no fair-share like Slurm's |
| E. **Self-managed Slurm on EC2** | full control of `slurm.conf`, versions and AMI; the norm for LLM training; `slurmdbd` gives accounting and fair-share; **the Docker test runs the same file** | we operate the control plane ourselves |

Runtime options: (i) bare AMI + venv; (ii) **container image run by Slurm through Pyxis/Enroot**; (iii) Apptainer.

> **In simple words.** We run Slurm ourselves on EC2 (E), because then the Docker copy and the real cluster
> use the same configuration file, and what we test is what we run. ParallelCluster (C) would be a sensible
> fallback if operating Slurm ourselves becomes too much work.

## Decision

**E. Self-managed Slurm on EC2, with containers through Pyxis/Enroot. One `slurm/slurm.conf` for Docker and
EC2; only the node lines differ (`slurm-docker/nodes.conf` vs `slurm/nodes.ec2.conf`).**

### 1. Cluster shape

- **Head instance** (small, CPU): `slurmctld`, `slurmdbd` and MariaDB on an EBS volume with daily snapshots.
  A **backup `slurmctld`** on a second small instance (`SlurmctldHost=slurmctld2`), sharing
  `StateSaveLocation` on EFS. Tested in Docker: the backup takes over, running jobs keep training, new jobs are
  accepted, and the primary takes back control when it returns.
- **Accounting database backup:** `slurm/backup_db.sh` dumps `slurm_acct_db` daily to S3; a backup only counts
  after `--verify-latest` has restored it into a scratch database with the same tables (tested in Docker).
- **GPU nodes:** proposed `p6-b200.48xlarge` (8 × B200, EFA). The instructor said V5 will "handle B200 end to
  end". All nodes in **one cluster placement group**, one AZ, with EFA.
- **Main run on on-demand or an EC2 Capacity Block**, never spot (ADR 0004 §3), plus **one hot spare**.
- **The hot spare is not idle:** it runs preemptible team jobs, which the main run preempts when it needs the node.
- **Idle nodes are stopped:** `State=CLOUD` nodes with `SuspendProgram` / `ResumeProgram` and `SuspendTime=600`;
  `SuspendExcNodes=gpu-[1-9]:1` keeps exactly one idle node up as the spare. This replaces V4's idle alarm.

- **GPUs are a Slurm resource (gres).** `GresTypes=gpu`; on EC2 `gres.conf` uses `AutoDetect=nvml` and
  `cgroup.conf` sets `ConstrainDevices=yes`, so a job sees only its own GPUs (`CUDA_VISIBLE_DEVICES`). Job
  scripts ask `--gpus-per-node=8 --ntasks-per-node=8` (one rank per GPU). `AccountingStorageTRES=gres/gpu`
  records GPU-hours per account, so budgets and reports are in GPU-hours. Docker uses 2 placeholder GPUs per
  node (no computation) to test allocation and accounting.
- **The only Docker/EC2 differences** are four small files: `site.conf` (process tracking and cgroups),
  `nodes.conf`, `gres.conf`, `cgroup.conf` (`slurm-docker/` vs `slurm/*.ec2.conf`). `slurm.conf` is shared.
- **When nodes run out:** if more nodes fail than there are spares, the main run **waits** (pending) and
  Q-01 pages at 10 min. It does **not** shrink by itself: a smaller world changes speed, is not bit-equal, and
  needs a person to agree. The operator may restart it on fewer nodes (`sbatch --nodes=1`), which the
  rollback drill tests.

| Partition | For |
|---|---|
| `gpu` | training, whole nodes (`OverSubscribe=EXCLUSIVE`) |
| `eval` | separate, cheaper GPU nodes (g6e); the training node is never used for evaluation (S6, S15) |
| `debug` | 30-min limit |
| `cpu` | data, tokenizer, retention and export jobs |

### 2. Runtime image

- One image per release, **built in CI** from a lock file, pinned by **digest**. Base: the NVIDIA CUDA image
  matching the AMI driver's major version, plus `aws-ofi-nccl` and the EFA libraries; exact torch, triton and
  NCCL versions and our kernels. At build time the image runs an import and version check (kept from V4).
- Jobs run with `srun --container-image=<digest>`; the AMI holds only the driver, EFA, Enroot/Pyxis and Slurm.
- The digest goes into every checkpoint manifest (ADR 0001); a different digest refuses to resume unless overridden.
- **One stack only.** V4's second "nightly + tarball" path is not allowed. The Docker test image
  (`slurm-docker/Dockerfile`) pins torch 2.5.1 CPU the same way.

### 3. Storage

| Data | Where | Rule |
|---|---|---|
| Training shards | S3 → **local NVMe**, prefetched ahead of use | the next shard downloads while the current one trains (S6). No training read goes to S3 |
| Checkpoints | local NVMe → S3 | ADR 0001 |
| Code, configs, small shared files | the image (or EFS) | never on the training read path |
| Events, health JSON, incident records | local file → shipped by Team 08's agent | in Docker: the shared `s3` volume |

**No shared parallel file system for the main run**: one less component that can be the slow-storage failure,
and one less monthly bill. FSx for Lustre is revisited only if prefetch cannot keep data wait below target (Q-M9).
S3 is reached through a **gateway VPC endpoint** (no NAT data charges, no internet path).

### 4. Accounts, QOS and budgets (`slurmdbd`, `slurm/setup_accounts.sh`)

> **In simple words.** Every job is charged to an account. The main run's account outranks everyone and may
> take nodes from team jobs. Teams share the rest fairly, none can grab more than 2 nodes, and each has a
> GPU-hour budget that Slurm enforces.

```text
root
├── mainrun            QOS=mainrun   (Priority 1000, Preempt=normal, no walltime cap)
└── teams              QOS=normal    (Priority 10, GraceTime=60 s, PreemptMode=REQUEUE)
    ├── data ├── tokenizer ├── ke2 ├── arch ├── perf ├── infra ├── monitoring ├── eval ├── posttrain ├── release ├── quant
```

- `PriorityType=priority/multifactor`, fair-share across the children of `teams`.
- `PreemptType=preempt/qos`, `PreemptMode=REQUEUE`: a preempted team job gets SIGTERM, has `GraceTime` to
  save, and goes back to the queue.
- `AccountingStorageEnforce=associations,limits,qos`: a job with no valid account is rejected.
- `GrpTRES=node=2` per team account, so no team takes all of `teams`.
- **`GrpTRESMins` budget** per account, alerts at 50 / 75 / 90% (Q-15), hard stop at 100%.
- **Capacity calendar:** each week's GPU-hours per team, including Team 05's ladder and the dry runs, agreed
  with the lead.

### 5. Restart contract

> **In simple words.** If the main run stops for any reason, Slurm puts it back in the queue and it starts
> again from the last checkpoint, by itself. Only two cases wait for a person: a NaN, and a job that keeps
> crashing (3 times in 30 minutes), because restarting those just burns money.

- `slurm/mainrun.sbatch`: `--requeue --open-mode=append --exclusive`. The same script for every restart; it
  always resumes from `latest.json`. Rollback passes `TRAIN_EXTRA="--from-step K --skip N --branch B"`.
- `slurm.conf`: `JobRequeue=1`, **`RequeueExit=75`**, **`RequeueExitHold=78`**, `MaxBatchRequeue=1000`.
- `slurm/run_job.py` (the wrapper) maps every trainer outcome to 0, 75 or 78 (ADR 0001 §6), applies the
  **crash-loop cap** (3 crashes in 30 min → 78), and emits `job.started / ended / requeued / held`.
- **Signals.** The trainer treats SIGUSR1 and SIGTERM the same: save, wait for the commit, exit 75.
  - Preemption: Slurm sends SIGTERM, then SIGKILL after `GraceTime` (60 s) / `KillWait` (30 s).
  - Spot notice or health drain: `scancel --signal=USR1 <job>` from the node itself (ADR 0003).
  - The first draft relied on `--signal=USR1@120`. That only fires before a **time limit**, and the main run
    has none, so it was dropped.
- A node that stops answering for `SlurmdTimeout=30` s is marked DOWN and its job is requeued by Slurm.
- **Requeue delay.** Slurm holds every requeued job until the old job's launch credential has expired
  (`cred_expire` + 1 s, default ≈ 121 s). We set `AuthInfo=cred_expire=30`: measured ≈ 52 s from exit to running,
  every requeue succeeded. A first attempt forced an immediate restart with a controller epilog (≈ 4 s); run 7 of
  the simulation showed it sometimes starts the job while the old credential is still valid, and Slurm then kills
  it as "Job credential revoked" and marks it **FAILED**, losing the main run. That hack was removed.

### 6. Health and idle integration

- `HealthCheckProgram=slurm/healthcheck.sh` every `HealthCheckInterval` (30 s in Docker; the main-run value is
  set from its measured cost, ADR 0003). Slurm ignores this program's exit code, so the check drains the node
  itself on `fail` (`scontrol update state=drain`) and sends its jobs SIGUSR1.
- `Prolog=slurm/prolog.sh` runs the quick checks at allocation (`PrologFlags=Alloc`). It exits non-zero only on
  `fail`; Slurm then drains the node and requeues the job (`SchedulerParameters=nohold_on_prolog_fail`).
- `slurm/spot_watcher.py` on every node polls IMDSv2 for a spot notice (a file in Docker).

### 7. Access and secrets

- **No static AWS keys anywhere**, and none in git (gitleaks before every push; pr-guard also checks).
- Nodes use **IAM instance profiles**:

| Role | Can |
|---|---|
| training | read data; write (not delete) checkpoints |
| eval | read checkpoints only |
| retention job | delete non-protected checkpoints (the only delete permission) |

- Students never get SSH or AWS keys. The instructor's submit system calls `sbatch` with a team account through
  one service user (`svc` in Docker).
- Admin access uses SSM Session Manager, not an open port 22. The database password lives in Secrets Manager
  (the Docker file's `simonly` value is for the local container only).

### 8. Event schema and time (shared by all ADRs)

- Every event is one JSON line `{schema, time, host, run_id, job_id, event, ...}`, `time` in UTC ISO-8601
  (`common/events.py`). Each process writes its own file, so writers never interleave.
- Events: `checkpoint.started / committed / failed / backpressure / retention`, `restore.done / failed / fatal`,
  `export.done / failed`, `train.started / step / stop_requested / watchdog / nonfinite / exit`,
  `job.started / ended / requeued / held / signal / signalled / rejected`, `node.drained / resumed / spot_notice`,
  `health.report`, and in the test `drill.injected`.
- chrony runs on every node (Amazon Time Sync); the health check reports the offset (Q-17).

### 9. Goodput accounting

> **In simple words.** We split every second of the main run into "trained the model" and named kinds of
> waste, and put a dollar figure on each. Then we know where the money went.

| Bucket | Source |
|---|---|
| productive training (kept steps) | the rest of the wall clock |
| checkpoint stall | `checkpoint.committed.stall_s` (+ backpressure) |
| lost work (steps redone) | step at fault − restored step, × step time |
| detection (fault → job ends) | fault time → `job.ended` |
| reschedule (job end → job running) | `job.ended` → `job.started` |
| restore (job start → first productive step) | `job.started` → first `train.step` |

```text
goodput = productive / wall-clock
planning model:  waste ≈ stall/interval + (interval/2 + MTTR) / MTBF
```

`metrics/evaluate.py` rebuilds one **incident record** per interruption from the events:
`{kind, cause_class, t_fault, detect_s, reschedule_s, restore_s, mttr_s, step_at_fault, restored_step,
lost_steps, lost_s, human_action}`. `cause_class` uses the Llama 3 / S15 taxonomy. MTBF, MTTR and the paper's
failure table come from these records only. Team 08 stores and displays them.

### 10. Dry runs and drills (Team 07 runs both)

Each dry run, and the Docker simulation, runs these drills, timed and recorded as incidents:

| Drill | How | Expected |
|---|---|---|
| preemption | team jobs on all GPU nodes, then submit the main run | team jobs save on SIGTERM and requeue; main run starts |
| kill | `kill -9` one training process | job requeued; resumes from the last commit |
| SIGUSR1 | `scancel --signal=USR1` | emergency save; 0 steps lost |
| spot notice | notice on one node | node drained; save; job moves to the spare |
| health fail | forced `fail` on one node | node drained by the health check; save; job moves to the spare |
| hang | `SIGSTOP` one rank | step watchdog exits 75; job restarts |
| node loss | stop a node's container (instance) | Slurm marks it DOWN; job requeued on the remaining nodes + spare |
| NaN | NaN at a set step | no save; exit 78; held |
| rollback + smaller world | `--from-step --skip 5 --branch b1` on 1 node instead of 2 | restored, resharded, branch b1; run completes |
| crash loop | a job that crashes at step 3 | requeued twice, then held |
| budget | `GrpTRESMins` exceeded | next job pending with a GrpCPUMins reason |
| fair share | heavy vs unused account | heavy account's fair-share factor is lower |
| retention, export | retention job; safetensors export on the eval partition | no protected checkpoint deleted; probe bit-equal |
| GPU allocation | main run on 2 nodes × 2 GPUs; a job asking 3 GPUs per node | `gres/gpu=4` allocated; the over-ask refused at submit |
| kill during save | slow the upload, kill all ranks while a save is in flight | the half-uploaded checkpoint is never used |
| kill during restore | slow the restore, kill while restoring | next attempt restores the same checkpoint cleanly |
| two nodes down | force `fail` on both main-run nodes (one spare) | main run waits (pending), no crash loop; recovers when nodes return |
| controller failover | stop the primary `slurmctld` | backup in control, training continues, new jobs accepted, primary returns |
| DB backup | dump and test-restore `slurm_acct_db` | same table count in the restored copy |
| loss spike | loss × 10 for 5 steps (no NaN) | no save during the spike; exit 78; held; rollback completes |
| repeats | kill and SIGUSR1 three times each | MTTR p50 / p95 per fault type |
| live alerts | `metrics/watch.py` runs during the whole simulation | HIGH alerts fire for the spot notice, the held run and the NaN |

### 11. Operations (added after the week-1 review)

> **In simple words.** How we change things without breaking the run, where logs live, how someone gets woken
> up, and what is still only on paper.

- **Upgrades** (image, driver, Slurm) never happen in the middle of a run segment. Order: build and test the new
  image in CI → run the checkpoint tests on one drained node → wait for a checkpoint commit → `scancel
  --signal=USR1` the main run (emergency save, exit 75) → roll the nodes → requeue with
  `TRAIN_EXTRA=--allow-env-change`, which is logged in the restore event. Slurm itself: `slurmdbd` first, then
  `slurmctld`, then `slurmd`, as Slurm's upgrade guide requires.
- **Logs:** job stdout/stderr and events go to node-local files, shipped by Team 08's agent to S3
  (`s3://…/logs/`, 90 days Standard, then Glacier). In Docker they are on the shared `/s3` volume.
- **Paging path:** `metrics/watch.py` runs on the head node, recomputes the catalog every 30 s, and on a new
  severity writes `alert.fired` and posts to `V5_ALERT_WEBHOOK` (Team 08's pager or a chat webhook). HIGH pages
  the on-call person. Tested in the simulation (alerts fired live for the drills).
- **Security, designed but not yet tested on AWS:** munge key from Secrets Manager at boot; IAM roles per job type
  (§7); the submit service; SSM-only admin access. These need the AWS account (week 2).

### 12. Metrics

All queue metrics with LOW / MEDIUM / HIGH thresholds are Q-01 … Q-21 in `metrics/catalog.yaml`
([metrics_catalog.md](../metrics_catalog.md)). The ones that page immediately (HIGH):

| ID | Metric | HIGH when |
|---|---|---|
| Q-01 | Main run not running | pending ≥ 10 min |
| Q-02 | Main run held | any |
| Q-03 | Reschedule latency | ≥ 10 min |
| Q-05 | Interruptions | ≥ 10 per day |
| Q-06 | Idle GPU node-hours | ≥ 4 per day |
| Q-07 | Allocated but not stepping | ≥ 15 min |
| Q-09 | Drained nodes | ≥ 3 |
| Q-10 | Node replacement time | ≥ 15 min |
| Q-11 | slurmctld down | ≥ 60 s |
| Q-15 | Account budget used | ≥ 90% |
| Q-17 | Clock offset | ≥ 500 ms |
| Q-19 | Goodput | ≤ 80% |
| Q-20 | MTTR | ≥ 60 min |

Targets: goodput ≥ 90%, MTTR ≤ 15 min, reschedule ≤ 2 min with the spare, node replacement ≤ 5 min,
human interventions ≈ 0 (Llama 3: 3 of 466).

## Consequences

Good:

- The main run cannot be starved, the rest is shared fairly and within budget, and idle nodes are stopped.
- Restarts are automatic, use one script, and lose ≈ 0 work when there is a warning signal.
- One pinned stack; training never reads from S3 directly.
- Goodput, MTTR and dollars come from records, so the paper's failure table is factual.

Bad and accepted:

- **Ops load:** we operate slurmctld, slurmdbd and MariaDB, plus image CI.
- **Container cost:** Pyxis/Enroot adds a component, and the first image pull on a node costs time (cache on NVMe, Q-18).
- **Preemption cost:** preempted team jobs lose the work since their last save (Q-14).
- **Spare node cost:** a hot spare costs one node of GPU-hours; it runs preemptible work to recover some of that.
- **Local disk only:** with no shared FS, a node's local checkpoint copy is lost with the node; S3 is the copy that counts.
- **Docker is not EC2:** no GPUs, EFA, Pyxis, power saving or cgroup enforcement in the test (§Validation).

## Validation

`sim/run_sim.py` starts `slurm-docker/` (MariaDB, slurmdbd, slurmctld, 4 compute containers) with the
production `slurm.conf`, runs every drill in §10, then `metrics/evaluate.py`. Results are in
`sim/results/` (`drills.json`, `report.md`, `metrics.json`, `alerts.json`); see [sim_results.md](../sim_results.md).

**Result (2026-10-27): 17 of 17 drills pass in 10 minutes.** Measured per incident (2 nodes × 2 ranks, tiny model):

| Drill | Detect | Reschedule | Restore | MTTR | Steps lost |
|---|---|---|---|---|---|
| kill -9 | 0.2 s | 4.7 s | 1.2 s | 6 s | 92 (since last commit) |
| SIGUSR1 | 0.4 s | 1.1 s | 1.3 s | 3 s | 0 (emergency save) |
| spot notice | 5.3 s | 40 s (spare freed by preempting its team job) | 1.3 s | 47 s | 0 |
| health fail | 26 s (health interval 30 s) | 3.1 s | 1.7 s | 31 s | 0 |
| hang (SIGSTOP) | 21 s (watchdog 20 s) | 1.1 s | 2.0 s | 24 s | 42 |
| node loss | 55 s (`SlurmdTimeout` 30 s + kill) | 1.4 s | 1.3 s | 57 s | 12 |

The first simulation run measured ≈ 121 s reschedule on every incident (Slurm's requeue `BeginTime`);
`cred_expire=30` (§5) cuts it to ≈ 50 s. Goodput in the simulation is 63% only because a fault is injected every
≈ 30 s of training; ADR 0004 projects the measured durations onto real failure rates.

Differences from EC2, by design: CPU instead of GPU (gloo instead of NCCL), `proctrack/linuxproc` and
`task/none` instead of cgroup enforcement (`cgroup.conf` uses `IgnoreSystemd=yes` because there is no
systemd), no Pyxis (the image already holds the code), no `State=CLOUD` power saving, one controller.

## Open points

- GPU type and count; Capacity Block or on-demand.
- The AWS account, IAM roles and S3 bucket names.
- The submit system's contract: does it call `sbatch` with an account?
- The on-call rota (who acts on a HIGH alert at 4 a.m.).
- The capacity calendar for weeks 2–4.
- Automatic shrink to fewer nodes (decided against for now, §1); revisit if spares prove too expensive.
- Security items in §11, which need the AWS account.
