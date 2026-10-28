# Team 07 submission: checkpoint, job queue and node health for the V5 main run

What we deliver: the infrastructure that runs the V5 main run on **AWS EC2 GPU nodes, scheduled by Slurm**. It
is designed so that **money already spent on training is never lost**. It comes with LOW / MEDIUM / HIGH
metrics that tell people what to do. Every decision is explained in simple words. Everything here was proven
in a Docker copy of the Slurm cluster (31 of 31 drills). That evidence is in the optional
[`../support/`](../support/README.md) folder.

> **In simple words.** Training costs money every hour, and at this scale machines break every few hours.
> We **save** the training often and prove every saved copy is exact (checkpoint). We let **Slurm** restart the
> job from that copy by itself (queue). We take **broken GPUs** out before they waste paid hours (health). And
> we tell a person **what to do** only when the system stopped on purpose (monitoring).

## What is here

```text
submission/
├── README.md        ← you are here
├── adr/             the 4 decisions, each with an "In simple words" box
├── ckpt/            checkpoint      ckpt.py (save, commit, restore checks, retention, export), store.py (folder or S3)
├── queue/           job queue       job_queue.py (job wrapper, exit codes, crash loop, spot watcher, drain, status)
├── slurm/           Slurm setup     slurm.conf, EC2 files, job scripts, prolog/epilog/health hooks, accounts, DB backup
├── health/          node health     health_check.py, xid_scan.py, allreduce_test.py, thresholds.yaml
├── metrics/         monitoring      catalog.yaml (74 metrics), watch.py (live alerts), evaluate.py, explain.py, prices.yaml
└── common/          shared          events.py (event format), reasons.py (every code → plain words + what to do)
```

| Area | Decision | Read | Main file |
|---|---|---|---|
| Checkpoint | [ADR 0001](adr/0001-checkpoint-format-and-commit.md) | [ckpt/README.md](ckpt/README.md) | `ckpt/ckpt.py` |
| Job queue | [ADR 0002](adr/0002-job-queue-slurm.md) | [queue/README.md](queue/README.md) | `queue/job_queue.py`, `slurm/slurm.conf` |
| Node health | [ADR 0003](adr/0003-node-health-checks.md) | [health/README.md](health/README.md) | `health/health_check.py` |
| Cost and monitoring | [ADR 0004](adr/0004-cost-model-and-alerting.md) | [metrics/README.md](metrics/README.md) | `metrics/catalog.yaml`, `metrics/watch.py` |

Reading order: [adr/README.md](adr/README.md) → the README of the area you care about → for the evidence,
[review_week1.md](../support/docs/review_week1.md) (what the simulation showed, the questions we expect, and what
is still open).

## The decisions in one table

| Question | Decision | Why, in simple words |
|---|---|---|
| How do we save? | PyTorch DCP shards + our manifest; the manifest is written **last**, then `latest.json` | a half-written save can never be picked |
| How do we know a restore is right? | 4 proofs: state digest, **bit-equal** probe loss, next-batch hash, frozen hashes | catches last year's silent misload |
| How often? | every 30 min (Young's formula, capped); emergency save on any warning | warnings lose ≈ 0 work, the rest ≤ one interval |
| Who schedules? | self-managed **Slurm** on EC2; the same `slurm.conf` in Docker | we test exactly what we run |
| How does it restart? | exit **75** = Slurm requeues; **78** = hold for a person; 3 crashes in 30 min = hold | no human for normal faults, no money burnt in loops |
| Who gets the GPUs? | main run first (preempts); teams by fair share, ≤ 2 nodes, GPU-hour budgets | the main run cannot starve; no team grabs everything |
| How do we find broken GPUs? | checks before every job and every 30 s; drain + save + move to the hot spare | a bad GPU costs seconds, not a crash |
| How does a person know what to do? | every job, alert and drained node says what happened, what was done, what to do | `python metrics/explain.py <anything>` |

## For monitoring and on-call

```bash
python queue/job_queue.py status       # every job and drained node, in plain words
python metrics/explain.py 78           # or 137, loss_spike, CK-05, v5:xid, JobHeldUser ...
```

| Exit | Means | You do |
|---|---|---|
| 0 | done | nothing |
| 75 | saved (or crashed once), restarts by itself in ~50 s | nothing |
| 78 | **held on purpose**: retrying would waste GPU hours | read the job comment; it says what to do |

Full guide: [metrics/README.md](metrics/README.md).

## Paths in this folder

Commands in these READMEs run from `submission/` (for example `python ckpt/ckpt.py list`). On the cluster the
team folder is mounted at `/opt/v5`, so `slurm.conf` and the job scripts use `/opt/v5/submission/...`. Paths
starting with `support/` point to the optional folder next to this one.

## Not done yet (honest list)

- Real GPUs: the Kaggle 2×T4 run is ready, not yet run; B200 numbers come in week 2.
- AWS account: EFA, Pyxis/Enroot, power-saving scripts, IAM roles, AWS Budgets, real S3 speed.
- Truly sharded state (FSDP/ZeRO-3, expert parallel) needs Team 05's model; the real data loader must pass
  `ckpt/loader_contract.py` (Teams 01/02).
- Prices in `metrics/prices.yaml` are assumptions until checked; on-call rota and alert channel need people.
