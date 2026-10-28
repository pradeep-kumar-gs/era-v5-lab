# Team 07-infrastructure

Put your work in a **topic folder**: `teams/07-infrastructure/<topic>/<file>`, for example `teams/07-infrastructure/harness/run.py`.
The team lead decides the topic names. Do not make personal folders. Do not make new folders under `teams/`.
Open a pull request; the lead reviews and merges it. The pr-guard check must pass.

## Two folders

| Folder | What | Who needs it |
|---|---|---|
| **[submission/](submission/README.md)** | **what we submit**: the 4 ADRs, a README per area, and the code that runs on the cluster: `ckpt/` (checkpoint), `queue/` + `slurm/` (job queue), `health/` (node health), `metrics/` (monitoring) and `common/` | the instructor, reviewers, other teams |
| [support/](support/README.md) | **optional**: what we used to prove it: tests, the Docker Slurm simulation, the test trainer, Kaggle GPU checks, the cost model, generated reports and the week-1 review | anyone who wants to re-run the evidence |

Start here: **[submission/README.md](submission/README.md)**.

`submission/` never needs `support/` to run on the cluster. The one link between them is a default: the job
wrapper starts the test trainer from `support/simulation/trainer/` unless `V5_TRAIN_SCRIPT` names the real one.
