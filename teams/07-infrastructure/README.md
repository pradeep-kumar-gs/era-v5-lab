# Team 07-infrastructure

Put your work in a **topic folder**: `teams/07-infrastructure/<topic>/<file>`, for example `teams/07-infrastructure/harness/run.py`.
The team lead decides the topic names. Do not make personal folders. Do not make new folders under `teams/`.
Open a pull request; the lead reviews and merges it. The pr-guard check must pass.

## What we deliver

Checkpoint, job queue and node health for the V5 main run on AWS EC2 GPU nodes, scheduled by Slurm, designed so
that **money already spent on training is never lost**, with LOW / MEDIUM / HIGH metrics that tell people what to
do. Every decision is explained in simple words.

## How the folder is organised

```text
teams/07-infrastructure/
├── README.md                 ← you are here
├── docs/                     the decisions and the evidence
│   ├── adr/                  ADR 0001–0004 (start with adr/README.md)
│   ├── review_week1.md       instructor questions, and the status of each
│   ├── metrics_catalog.md    all 74 metrics with thresholds (generated)
│   ├── codes_and_actions.md  every exit code / cause / drain reason → what to do (generated)
│   └── sim_results.md        the latest simulation run (generated)
│
├── ckpt/      ┐
├── slurm/     │  THE SUBMISSION: what runs on the cluster
├── health/    │  each has a README that explains it in plain words
├── metrics/   │
├── cost/      │
├── common/    ┘
│
└── simulation/               OPTIONAL: the kit we used to prove it (Docker Slurm, test trainer, drills,
                              Kaggle GPU checks). Not needed to use or review the design.
```

| Folder | What | Read | Decision |
|---|---|---|---|
| `ckpt/` | checkpoint save, commit, restore checks, retention, export; folder or S3 store | [ckpt/README.md](ckpt/README.md) | ADR 0001 |
| `slurm/` | `slurm.conf` (shared by Docker and EC2), EC2-only files, job scripts, job wrapper, prolog/epilog, spot watcher, accounts, DB backup | [slurm/README.md](slurm/README.md) | ADR 0002 |
| `health/` | node health checks, Xid scan, all-reduce and NCCL-timeout tests, thresholds | [health/README.md](health/README.md) | ADR 0003 |
| `metrics/` | **monitoring**: metric catalog, live alerts, report, `explain.py` | [metrics/README.md](metrics/README.md) | ADR 0004 |
| `cost/` | prices (assumptions), cost model, instances-outside-Slurm check | ADR 0004 | ADR 0004 |
| `common/` | the event format (`events.py`) and the plain-words reasons (`reasons.py`) | — | ADR 0002 §8 |
| `simulation/` | optional proof kit | [simulation/README.md](simulation/README.md) | ADR 0002 §10 |

## Reading order

1. [docs/adr/README.md](docs/adr/README.md): the four decisions; each has an **"In simple words"** box.
2. The README of the area you care about: [checkpoint](ckpt/README.md), [queue](slurm/README.md),
   [health](health/README.md), [monitoring](metrics/README.md).
3. [docs/review_week1.md](docs/review_week1.md): what the simulation showed, the instructor's likely questions,
   and what is fixed, decided or still open.

## For monitoring and on-call: what does this code mean?

Every job, alert and drained node explains itself in the same three parts: **what happened, what the system
already did, what you do**.

```bash
squeue -o "%.10i %.12T %.20r %k"        # the last column is our plain-words status
python metrics/explain.py 78            # or 137, loss_spike, CK-05, v5:xid, JobHeldUser ...
```

| Exit | Means | You do |
|---|---|---|
| 0 | done | nothing |
| 75 | saved (or crashed once), restarts by itself in ~50 s | nothing |
| 78 | **held on purpose**: retrying would waste GPU hours | read the job comment; it says what to do |

Full guide: [metrics/README.md](metrics/README.md). All codes: [docs/codes_and_actions.md](docs/codes_and_actions.md).

## How to check it yourself

Needs Docker (Docker Desktop on a Mac is fine). Nothing needs AWS.

```bash
cd teams/07-infrastructure
docker compose -f simulation/slurm-docker/docker-compose.yml build slurmdbd       # once, ~5 min

# 1. unit tests: checkpoint T1-T13, health, cost, monitoring (~1 min)
docker run --rm -v "$PWD":/opt/v5:ro --entrypoint bash v5-slurm-sim:1 \
  -c 'cd /opt/v5 && python -m pytest -q -p no:cacheprovider -W ignore ckpt health cost metrics'

# 2. the same checkpoint tests against S3 (the moto emulator, real boto3 code path)
docker run --rm -v "$PWD":/opt/v5:ro --entrypoint bash v5-slurm-sim:1 -c '
  cd /opt/v5; moto_server -p 9000 >/dev/null 2>&1 & sleep 3
  export V5_S3_ENDPOINT=http://127.0.0.1:9000 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_REGION=us-east-1
  python -c "import boto3; boto3.client(\"s3\", endpoint_url=\"http://127.0.0.1:9000\").create_bucket(Bucket=\"v5-test\")"
  V5_TEST_REMOTE=s3://v5-test/ckpt python -m pytest -q -p no:cacheprovider -W ignore ckpt/test_ckpt.py'

# 3. optional: the full Docker Slurm simulation, every drill (~25 min)
python3 simulation/run_sim.py            # results in simulation/results/, summary in docs/sim_results.md

# 4. cost model (prices are assumptions until cost/prices.yaml is checked)
docker run --rm -v "$PWD":/opt/v5 --entrypoint bash v5-slurm-sim:1 \
  -c 'cd /opt/v5 && python cost/cost_model.py --measured simulation/results/metrics.json'

# after editing metrics/catalog.yaml or common/reasons.py, regenerate the docs pages
docker run --rm -v "$PWD":/opt/v5 --entrypoint bash v5-slurm-sim:1 -c 'cd /opt/v5 &&
  python metrics/evaluate.py --render-md docs/metrics_catalog.md && python metrics/explain.py --all docs/codes_and_actions.md'
```

What "pass" looks like: all unit tests pass in steps 1 and 2; step 3 ends with `N/N drills passed`.

## Review checklist (for the lead)

- [ ] Every decision in ADR 0001–0004 has an "In simple words" box a non-infra student can follow.
- [ ] Each item in `docs/review_week1.md` is marked fixed / tested / open, with a reason.
- [ ] Unit tests pass (steps 1–2); the simulation passes if you run it (step 3).
- [ ] Every HIGH alert and every exit-78 cause says what to do (`metrics/README.md`, `docs/codes_and_actions.md`).
- [ ] `cost/prices.yaml` is checked against the AWS pricing page before any dollar figure is used.
- [ ] Open points in each ADR have an owner and a date.
