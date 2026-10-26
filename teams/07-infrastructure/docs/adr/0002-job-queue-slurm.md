# ADR 0002: Job queue — self-managed Slurm on EC2 with two account tiers

- Status: Proposed
- Date: 2026-10-25
- Area: Queue (`docs/queue_design.md`, `slurm-docker/`)

## Context

V4 had **no scheduler**. GPUs were reached through many student AWS accounts, and jobs were started by hand.
Team 15's notes say "infra managed manually, run on few accounts". One cluster sat idle for 3 days because
someone forgot it, and the only guard was a CloudWatch CPU-idle alarm that stopped the instance. Restarts were
typed by hand, with long command lines and config paths that changed after a crash.

For V5 the instructor gives **no keys to students**. Teams submit work, it runs on the cluster, and they get
the result. The cluster must therefore:

- run the **main run** without letting anything else take its GPUs;
- share the remaining capacity fairly between teams (ablations, kernel benchmarks, evals, quantization);
- **restart the main run on its own** after a failure, from the last committed checkpoint (ADR 0001);
- keep sick nodes out of service (ADR 0003).

The first shared GPU node arrives in week 2; the main run starts on 28 Nov.

## Decision drivers

1. Automatic requeue and restart, with no person typing a command.
2. Hard priority for the main run, and fair share for everyone else.
3. Per-team usage records (who spent which GPU-hours) for budget and the paper.
4. Testable on a laptop in week 1.
5. The same configuration on the real cluster: no rewrite between the test and production.

## Options considered

| Option | For | Against |
|---|---|---|
| A. No scheduler, as in V4 (SSH + scripts) | nothing to run | no queue, no fairness, no automatic restart; the failures V4 had |
| B. Kubernetes (EKS) + Kueue or Volcano | strong for services; Team 10 uses Docker anyway | gang scheduling and topology for multi-node NCCL jobs are add-ons; extra layers (CNI, device plugin, EFA plugin); most LLM training tooling assumes Slurm |
| C. AWS ParallelCluster (managed Slurm) | Slurm, plus AWS-built autoscaling, FSx and EFA wiring | its config language and update cycle sit between us and Slurm; less control over versions |
| D. AWS Batch | managed, pay per job | weak multi-node gang jobs; no fair-share between accounts like Slurm's; poor fit for one long-running job |
| E. **Self-managed Slurm on EC2** | full control of `slurm.conf`, versions and the AMI; the industry norm for LLM training; `slurmdbd` gives accounting and fair-share; the Docker test runs the **same** config files | we run slurmctld, slurmdbd and MariaDB ourselves (backup, upgrades, HA) |

## Decision

**E. Self-managed Slurm on EC2**, with the same `slurm.conf` used in the Docker test.

**Cluster shape**

- One head instance runs `slurmctld`, `slurmdbd` and MariaDB, on an EBS volume with snapshots.
- A second instance runs a backup `slurmctld` (`SlurmctldHost` listed twice) once the main run starts.
- GPU instances are compute nodes, built from one pinned AMI or container image. Its digest is recorded in every
  checkpoint manifest.
- Partitions:
  - `gpu`: training nodes, exclusive;
  - `debug`: short limit, for smoke tests;
  - `cpu`: data and eval helpers.

**Two account tiers, set up in `slurmdbd` with `sacctmgr`**

```text
root
├── mainrun            QOS=mainrun   (priority high, Preempt=normal, GrpTRES all GPUs, no walltime cap)
└── teams              QOS=normal    (PreemptMode=REQUEUE)
    ├── arch    ├── perf    ├── eval    ├── quant    ├── posttrain   … one child per team
```

- `PriorityType=priority/multifactor`, with fair-share across the children of `teams`, so heavy users decay and
  light users rise.
- `PreemptType=preempt/qos`. The `mainrun` QOS can take nodes from `normal` jobs, which get SIGUSR1 (save),
  then are requeued.
- `AccountingStorageEnforce=associations,limits,qos`. A job with no valid account is rejected.

**Restart contract** (shared with ADR 0001)

- Main-run jobs are submitted with `--requeue --signal=USR1@120 --open-mode=append`.
- An epilog script maps exit codes:
  - **75**: checkpoint done, requeue;
  - **1**: fatal, hold the job and page;
  - **0**: done.
- The job script always resumes from `latest`. There is one command and no hand-edited paths, which fixes V4's
  "restart command hell".
- A cap on requeues per hour stops a crash loop.

**Health integration (ADR 0003)**

- `HealthCheckProgram` runs the quick checks every `HealthCheckInterval`.
- A `Prolog` script runs them before every job.
- A failed node goes to `DRAIN` with the reason, and the job requeues onto healthy nodes.

**Idle control**

- `SuspendProgram` and `ResumeProgram` stop idle cloud nodes after a timeout. This replaces V4's CloudWatch
  idle alarm inside the scheduler itself.

**Student access**

- Students never get SSH or AWS keys. The instructor's submit system calls `sbatch` with the team account.

## Consequences

Good:

- The main run cannot be starved.
- Teams share the rest fairly.
- Every GPU-hour is attributed to an account (`sacct`/`sreport`).
- Restarts are automatic and use one command.
- Bad nodes leave the pool on their own.

Bad and accepted:

- We operate the control plane ourselves: database backups, Slurm upgrades, and controller failover.
- Preempted team jobs lose up to their own checkpoint interval. Their code must handle SIGUSR1, or accept
  restarting from zero.
- Multi-node performance needs EFA, placement groups and NCCL settings on the AMI. That belongs to the GPU
  design, not this ADR.

## Validation

In week 1, `slurm-docker/` runs the controller, slurmdbd, MariaDB and two compute containers on a laptop, using
the production `slurm.conf` minus the GPU lines. The tests:

- **Accounts:** submissions from `mainrun` and from two `teams` children are accepted; a job with no account is
  rejected.
- **Fair-share:** two `teams` children submit equal loads; `sshare` shows usage, and priority moves toward the
  lighter one.
- **Preemption:** a `mainrun` job preempts a running `teams` job, which gets SIGUSR1, exits 75 and is
  requeued.
- **Requeue:** a fake training job that exits 75 is requeued and resumes from its saved step; a job that exits
  1 is held.
- **Drain:** a health check that fails marks the node `DRAIN` with its reason.

## Open points

- Does the instructor's submit system wrap `sbatch`, or replace it?
- Spot or on-demand for training nodes? With spot, the EC2 notice must feed the same SIGUSR1 path.
- How many GPU nodes, and does the main run use all of them, or keep one spare for fast replacement?
