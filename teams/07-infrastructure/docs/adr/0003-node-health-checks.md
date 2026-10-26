# ADR 0003: Node health checks — JSON report, skip-not-fail, drain on hard faults

- Status: Proposed
- Date: 2026-10-25
- Area: Health (`health/launch.sh`, `health/gpu_health.sh`, `health/xid_scan.py`)

## Context

Most interruptions in a large run come from the machine, not the model. Llama 3's 419 unexpected interruptions
were:

| Cause | Share |
|---|---|
| Faulty GPU | 30.1% |
| GPU memory (HBM) | 17.2% |
| Software | 12.9% |
| Network | 8.4% |
| Other (NIC, SRAM, NCCL timeouts, power) | 22.4% |
| Silent data corruption | 1.4% |

The Session 15 reference run had 10 of 38 interruptions from GPU or node hardware.

Session 15 names the tools: DCGM health and Xid alerts, `dcgmi diag` before the run, the NCCL flight recorder
for hangs, and per-GPU step timing for stragglers. Its alert table says: **Xid 79, 94 or 95, or an ECC error →
page, remove the node.**

V4's repository has **no GPU health check at all**: no nvidia-smi parsing, no Xid scan, no NCCL test. Its
monitoring watched loss and CPU idle.

The scripts must also run where tools are missing: a laptop (no GPU), Colab or Kaggle (no DCGM, no `dmesg`
access), and the real node.

## Decision drivers

1. Catch a bad node **before** it joins a job, and while a job runs.
2. Machine-readable output that the queue (ADR 0002) and Monitoring (Team 8) can consume.
3. Never crash, and never claim "pass", because a tool is absent.
4. Fast enough to run in a Slurm prolog: seconds, not minutes.
5. One set of scripts from laptop to cluster.

## Options considered

| Option | For | Against |
|---|---|---|
| A. DCGM only (`dcgmi health`, `dcgmi diag`) | NVIDIA's own; deep diagnostics | not on Colab, Kaggle or laptops; `diag -r 3` takes minutes; no NCCL multi-node test |
| B. A ready-made stack (e.g. a Kubernetes node-problem-detector or a vendor health agent) | less code | we are not on Kubernetes (ADR 0002); hides what is checked |
| C. **Our own small scripts that wrap nvidia-smi, DCGM if present, kernel-log Xid scan and an NCCL all-reduce, with one JSON schema** | runs everywhere; each check visible and testable; DCGM is used when present | we maintain the Xid table and thresholds |

## Decision

**C. Three scripts and one JSON contract.**

**`gpu_health.sh`: per-node checks.** One JSON object, written to stdout and to a file:

- driver and CUDA versions, and the GPU count against the expected count;
- per GPU, from `nvidia-smi -q`/`--query-gpu`:
  - temperature;
  - power;
  - **ECC volatile and aggregate uncorrectable errors**;
  - retired or remapped rows pending or failed;
  - clocks and throttle reasons;
  - PCIe link width and generation;
- NVLink status (`nvidia-smi nvlink -s`) where present;
- `dcgmi diag -r 1` (quick) when `dcgmi` exists. Level 2 and 3 run only on demand.

**`xid_scan.py`: kernel-log Xid scan.**

- Reads `dmesg` or `journalctl -k`, or a file given as an argument (used for tests and on hosts without access).
- Parses `NVRM: Xid (PCI:…): <code>` lines and classifies each code from a table:
  - **fatal → drain:** 48 (double-bit ECC), 63/64 (row remap), 74 (NVLink), **79** (fell off the bus),
    **94/95** (contained/uncontained ECC), 119/120 (GSP);
  - **warn:** 13, 31, 43 (usually an application fault);
  - unknown codes are reported as warn.
- Reports counts per GPU and per code since boot or since a given time.

**NCCL all-reduce test** (run from `launch.sh`):

- Uses `torch.distributed` across all local GPUs (and across nodes when given).
- Checks **correctness**: the reduced sum equals the expected value exactly.
- Reports algbw and **busbw** = algbw × 2(n−1)/n for sizes 1 MB to 256 MB.
- If `nccl-tests` `all_reduce_perf` is installed, it runs that too.
- Fails on a wrong result, a timeout, or busbw below a configured floor. The floor is set from the week-2
  baseline.
- With no GPU it falls back to gloo on CPU, which proves the code path and is marked as such.

**`launch.sh`: the entry point.**

- Runs the checks in order, merges them into one report, and prints a verdict.
- Starts the given training command only if the verdict is not `fail`.
- Modes: `--prolog` (quick checks, for Slurm), `--full` (adds DCGM diag and the NCCL sweep, before a dry run),
  `--report-only`.

**JSON contract** (shared by all three scripts):

```json
{
  "schema": "v5-health/1", "host": "...", "time": "ISO-8601", "mode": "prolog|full",
  "checks": [
    {"name": "ecc_uncorrectable", "status": "pass|warn|fail|skipped",
     "detail": "...", "metrics": {}, "gpu": 0}
  ],
  "verdict": "pass|warn|fail",
  "drain_reason": null
}
```

- A missing tool gives **`skipped`** with the reason (e.g. `"nvidia-smi not found"`). `skipped` never turns
  into `pass` or `fail`.
- Verdict rule: any `fail` gives `fail`. Otherwise, any `warn`, or a skipped check that is *required* for the
  mode, gives `warn`. Otherwise `pass`. On a GPU node, a skipped GPU check is required, so a GPU node where
  nvidia-smi is missing is not silently healthy.
- Exit codes: 0 pass, 1 warn, 2 fail. Slurm drains the node on 2, using `drain_reason` as the reason string.

## Consequences

Good:

- Bad GPUs are removed before they cost a restart.
- Monitoring gets the same JSON from every node.
- The scripts run and are tested on a laptop, Kaggle and the real node.

Bad and accepted:

- **A healthy node is not a correct GPU:** these checks do not detect silent data corruption. That needs
  step replay on another GPU (Session 15), which is a later ADR.
- **The Xid table needs care:** it must be kept current with NVIDIA's list, and the thresholds (temperature,
  busbw floor) need the week-2 baseline before they mean anything.
- **Root may be needed:** `dmesg` can require root, so on the cluster `xid_scan.py` runs from the Slurm prolog
  as root, or reads `journalctl` through a group.

## Validation

- **Mac:** every GPU check reports `skipped`, the verdict is `warn`, and the JSON is valid against the schema.
  The NCCL test runs the gloo path on 2 CPU processes.
- **Fixture logs:** `xid_scan.py` on saved kernel-log samples (Xid 79, 94, 13 and an unknown code) gives the
  expected classes. These are unit tests.
- **Kaggle 2×T4:** real nvidia-smi values, plus a 2-rank NCCL all-reduce with a correct result and measured
  busbw.
- **Week-2 node:** the full mode, including DCGM diag; this sets the baseline thresholds.
- **Slurm Docker:** a forced `fail` drains the node (ADR 0002).

## Open points

- Who receives a page, and how (Team 8 owns alerting)?
- Should a `warn` node stay in the main-run pool?
- How often does `HealthCheckInterval` run during the main run, and what does it cost in step time?
