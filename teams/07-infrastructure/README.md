# Team 07-infrastructure

Put your work in a **topic folder**: `teams/07-infrastructure/<topic>/<file>`, for example `teams/07-infrastructure/harness/run.py`.
The team lead decides the topic names. Do not make personal folders. Do not make new folders under `teams/`.
Open a pull request; the lead reviews and merges it. The pr-guard check must pass.

## Start here (reading order for reviewers)

1. [docs/adr/README.md](docs/adr/README.md): the four decisions. Each has an **"In simple words"** box.
2. [docs/review_week1.md](docs/review_week1.md): what the simulation showed, the questions we expect from the
   instructor, and the status of each one (fixed, tested, or still open and why).
3. [docs/sim_results.md](docs/sim_results.md): every drill and its measured result.
4. [docs/metrics_catalog.md](docs/metrics_catalog.md): all 74 metrics with LOW / MEDIUM / HIGH thresholds.

To learn one area in depth (plain-language walk-throughs with diagrams, files, commands, tests and metrics):

- **Checkpoint:** [ckpt/README.md](ckpt/README.md)
- **Queue (Slurm):** [slurm/README.md](slurm/README.md), which also covers `slurm-docker/`
- **Health:** [health/README.md](health/README.md)

## What is here

| Folder | What | Decision |
|---|---|---|
| `docs/adr/` | the decisions | start with [docs/adr/README.md](docs/adr/README.md) |
| `ckpt/` | checkpoint save / commit / restore / retention / export; `store.py` (folder or S3); loader contract; tests T1–T13; save-speed benchmark | ADR 0001 |
| `trainer/` | a tiny test trainer that follows every checkpoint, signal, watchdog and loss-spike rule (CPU or GPU) | ADR 0001, 0003 |
| `slurm/` | `slurm.conf` (shared), EC2-only files (`*.ec2.conf`), job scripts, job wrapper, prolog/epilog, health hook, spot watcher, accounts, DB backup | ADR 0002 |
| `slurm-docker/` | the Docker Slurm test cluster: 2 controllers, accounting DB, S3 emulator, 4 nodes | ADR 0002 |
| `health/` | node health checks, Xid scan, all-reduce and NCCL-timeout tests, thresholds, tests | ADR 0003 |
| `metrics/` | the metrics catalog, the evaluator (report) and the live alert watcher | ADR 0004 |
| `cost/` | prices (assumptions), cost model and report, instances-outside-Slurm check, tests | ADR 0004 |
| `common/` | the shared event format | ADR 0002 §8 |
| `sim/` | the drill simulation, the architecture team's `config_v0.yaml`, and results | ADR 0002 §10 |
| `kaggle/` | GPU validation on Kaggle 2×T4 (run separately; results go in `kaggle/results/`) | ADR 0001, 0003 |

## How to check it yourself

Needs Docker (Docker Desktop on a Mac is fine). Everything runs in the `v5-slurm-sim` image; nothing needs AWS.

```bash
cd teams/07-infrastructure
docker compose -f slurm-docker/docker-compose.yml build slurmdbd        # once, ~5 min

# 1. unit tests: checkpoint T1-T13, health, cost (~1 min)
docker run --rm -v "$PWD":/opt/v5:ro --entrypoint bash v5-slurm-sim:1 \
  -c 'cd /opt/v5 && python -m pytest -q -p no:cacheprovider -W ignore ckpt health cost'

# 2. the same checkpoint tests against S3 (the moto emulator, real boto3 code path)
docker run --rm -v "$PWD":/opt/v5:ro --entrypoint bash v5-slurm-sim:1 -c '
  cd /opt/v5; moto_server -p 9000 >/dev/null 2>&1 & sleep 3
  export V5_S3_ENDPOINT=http://127.0.0.1:9000 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_REGION=us-east-1
  python -c "import boto3; boto3.client(\"s3\", endpoint_url=\"http://127.0.0.1:9000\").create_bucket(Bucket=\"v5-test\")"
  V5_TEST_REMOTE=s3://v5-test/ckpt python -m pytest -q -p no:cacheprovider -W ignore ckpt/test_ckpt.py'

# 3. the full Docker Slurm simulation: every drill, live alerts, metrics report (~30 min)
python3 sim/run_sim.py                   # results in sim/results/, summary in docs/sim_results.md

# 4. cost model (prices are assumptions until cost/prices.yaml is checked)
docker run --rm -v "$PWD":/opt/v5 --entrypoint bash v5-slurm-sim:1 \
  -c 'cd /opt/v5 && python cost/cost_model.py --measured sim/results/metrics.json'

# after editing metrics/catalog.yaml, regenerate the catalog page
docker run --rm -v "$PWD":/opt/v5 --entrypoint bash v5-slurm-sim:1 \
  -c 'cd /opt/v5 && python metrics/evaluate.py --render-md docs/metrics_catalog.md'
```

What "pass" looks like: all unit tests pass in steps 1 and 2; step 3 ends with `N/N drills passed`.

## Review checklist (for the lead)

- [ ] Every decision in ADR 0001–0004 has an "In simple words" box a non-infra student can follow.
- [ ] Each item in `docs/review_week1.md` is marked fixed / tested / open, with a reason.
- [ ] Unit tests and the simulation pass on your machine (steps 1–3).
- [ ] `cost/prices.yaml` is checked against the AWS pricing page before any dollar figure is used.
- [ ] Open points in each ADR have an owner and a date.
