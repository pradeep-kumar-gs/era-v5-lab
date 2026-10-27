# ADR 0001: Checkpoint format, commit protocol and restore checks

- Status: Proposed
- Date: 2026-10-27 (revised from 2026-10-25; second revision after the week-1 review, see [review_week1.md](../review_week1.md))
- Area: Checkpoint (`ckpt/ckpt.py`, `ckpt/test_ckpt.py`, `trainer/train.py`, `docs/checkpoint_design.md`)
- Related: ADR 0002 (queue, exit codes, events), ADR 0003 (health, hang policy), ADR 0004 (cost, metrics)

> **In simple words.** Training costs money every hour. A checkpoint is a saved copy of the training, so
> if a machine breaks we continue from the copy instead of starting again. This ADR decides how to save
> (quickly, without stopping training for long), how to make sure a copy is complete and correct before
> we trust it, and how to prove a restore gives exactly the same training as before.

## Context

The lab's goal for Team 07: the cluster saves, restores and restarts the main run with no lost work.
Interruptions are routine at scale. The Session 15 reference run had one every 12.9 h and lost 5.6% of its
time; Llama 3 had 466 in 54 days. Every hour lost to a bad restart is paid for and trains nothing (ADR 0004).

V4 (LightningLM 0.1V) ran on one 8-GPU node with DeepSpeed checkpoints. Its code picks "latest" as the
highest step in an S3 listing, has no checksums, saves the CUDA RNG of the current device only, and checks
resume by eye. During the run V4 hit:

- a checkpoint damaged during save (step 3009, rolled back to 2909);
- a silent misload: a converter wrote wrong names, the load left weights random, and five cheap checks still
  passed (step-1 loss 2.95 against 2.55);
- library drift on a new machine.

V5 is larger and multi-node, so the world size changes when a node is lost. Several teams read checkpoints
(Teams 09, 11, 12), and several write state we save (Teams 01/02 data loader, 04 KE2, 05 architecture,
06 kernels). The architecture team's `config_v0.yaml` fixes `run.seed=1234` ("two runs with this seed must
match"), `attention.impl=eager` (fast attention backward does not repeat), `run.grad_clip=1.0` and AdamW.
Model sizes and precision are still open (TSAI fills them on 31 Oct).

## Decision drivers

1. A restore must be provably exact. The week-1 pass condition is **bit-equal loss on CPU**.
2. A partial or damaged checkpoint must never be picked.
3. The stall per save is small, and the work (money) lost per failure is bounded.
4. A checkpoint must load on a different number of GPUs, and on one CPU process.

## Options considered

| Option | For | Against |
|---|---|---|
| A. `torch.save` of a gathered state dict on rank 0 | simple; one file | gathers 1–7 TB on one rank (V4's OOM-during-gather); no parallel write; slow |
| B. DeepSpeed `save_checkpoint` (V4) | proven in V4; sharded | tied to DeepSpeed's engine and ZeRO stage; resharding needs `zero_to_fp32`; no integrity layer |
| C. **`torch.distributed.checkpoint` (DCP) plus our own manifest and commit layer** | every rank writes in parallel; **reshards on load** to any world size, including 1 CPU; works with FSDP2 and DDP; no engine lock-in | newer API; the manifest/verify layer is our code |
| D. Megatron distributed checkpoint | reshards; used at scale | brings in Megatron-Core, which we may not use |

> **In simple words.** We pick C: PyTorch's own sharded format, where every GPU writes its own part at
> the same time, and the checkpoint can later be loaded on a different number of GPUs. On top we add a
> small "receipt" (the manifest) that lists every file and its checksum.

## Decision

**C. DCP for the tensor data, wrapped in a commit protocol we own.**

### 1. State saved

- Model weights (fp32 master weights), including frozen tensors and buffers.
- Optimizer state (AdamW moments and step), scheduler and gradient scaler.
- **Every** RNG: Python, NumPy, torch CPU, and CUDA for each local device.
- Data position: the loader's ledger offset and the rollback skip list (the offset is the authority, not the step).
- Identity: run id, branch id, step, tokens seen.
- Environment: git sha, image digest, torch, CUDA, NCCL, tokenizer hash, **config sha256**, **`OMP_NUM_THREADS`**.
- Hashes of frozen tensors (KE2 tables).

Not saved: gradients, activations, NCCL state.

> **Found in the simulation.** On CPU, `OMP_NUM_THREADS` changes the last bits of the loss even with
> `torch.set_num_threads(1)`: torchrun sets it to 1, a plain launch does not, and the two runs differed from
> step 16. The thread count is therefore part of the environment; job scripts set it, and a mismatch refuses
> to resume.

### 2. Save protocol

> **In simple words.** Training pauses only for the quick copy to local disk. The upload to S3 runs in the
> background. The checkpoint "counts" only when every file is in S3 *and* re-checked; the receipt is written
> last, and only then does the `latest` pointer move. If anything dies halfway, the pointer still shows the
> previous good checkpoint.

1. Rank 0 decides when to save, and all ranks agree in one all-reduce (a stop signal on any rank counts).
2. A NaN/Inf guard runs first: a bad state is never saved (exit 78).
3. A free-space check: free NVMe must be ≥ 1.2 × this node's share of the last checkpoint, or the save is
   skipped with `checkpoint.failed cause=nvme_full` (alert CK-03).
4. **Stall:** DCP writes each rank's shard to node-local NVMe. On GPU this becomes `dcp.async_save`, so the
   stall is only the GPU-to-host copy into pinned host RAM.
5. **Background:** each rank uploads its own files to `s3://…/<run>/<branch>/step_<N>/` through `ckpt/store.py`
   (boto3: multipart above 64 MB, 16 parallel parts, standard retry mode for `503 SlowDown`). Each object
   carries its sha256 in metadata and S3 verifies a SHA256 checksum on upload. Then the rank writes
   `sums_rank<r>.json`.
6. Rank 0 waits for every rank's sums, re-checks every object in S3, writes `MANIFEST.json` **last**, then moves
   `latest.json` and appends to `commits.jsonl`, and emits `checkpoint.committed` (ADR 0002 §8).
   Re-check mode `V5_VERIFY_MODE`: `full` re-reads and hashes every object (tests, small checkpoints);
   `head` compares the checksum S3 stored at upload, which costs no download (TB-size checkpoints).

**One interface, two stores** (`ckpt/store.py`): `LocalStore` (a folder) and `S3Store` (boto3). The Docker
cluster runs `S3Store` against an S3 emulator (moto), so the code that will run on AWS has already run through
every drill. S3 has no append: `commits.jsonl` is read-modify-written by rank 0 of the one job that owns the run.

> **Node-local disk is a cache, never a copy that counts.** p6 instances' NVMe is *instance store*: it is wiped
> when the instance stops (including Slurm power-saving) or is replaced. A restore always works from S3; the
> local copy only saves a download when the job comes back on the same, still-running node.

S3 has no rename, so there is no `.tmp` directory: a prefix without a manifest is simply not a checkpoint.
The retention job (§5) deletes such orphans after one hour, because they are billed like any other object.
Only one save is in flight at a time; a save that has to wait for the previous one is counted (CK-10).

### 3. Restore protocol

> **In simple words.** We never guess which checkpoint is newest by listing folders. We follow the pointer,
> check the receipt, and then prove that what we loaded is exactly what we saved. If the newest checkpoint
> fails a check, we fall back to the one before it automatically.

1. Read `latest.json`; never list directories. Fallback order comes from `commits.jsonl`.
2. Check every file against the manifest checksums (rank 0), then copy it to local NVMe once per node.
3. Check the environment; a mismatch refuses (exit 78) unless `--allow-env-change` is given, which is logged.
4. **Strict key check:** every missing, extra, wrong-shape or wrong-dtype key aborts, and all of them are listed.
5. DCP load (reshards to the current world size), then RNG, data position and skip list.
6. Run the **checks in §8**. A damaged file or a failed check falls back to the previous commit; a strict-key or
   environment error does not (every checkpoint would fail the same way) and exits 78.

### 4. Interval

```text
interval = min( Young(stall, MTBF), eval cadence, RPO cap )
Young    = √(2 × stall × MTBF)
```

> **In simple words.** Saving often costs a little on every save; saving rarely costs a lot on every failure.
> Young's formula finds the balance. With today's failure rates the balance point is far (hours) for small
> clusters, so in practice the cap on how much work we accept losing decides.

`cost/cost_model.py` computes this per model size and cluster size (ADR 0004). With the S15 failure rate and
8 nodes, Young gives 3–17 min (1B–30B parameters). With 2 nodes the **30-min RPO cap** binds for 7B and up.
**Start value: 30 min**, or Young when Young is shorter. Revisit after the dry runs with measured stall and MTBF.

Work lost per failure: on average **interval/2 + stall** (≈ 15 min). The worst case is
**interval + commit latency** (≈ 40 min), when the failure hits just before a commit finishes. Signals that
give warning (preemption, spot notice, health drain) trigger an **emergency save**, so they lose ≈ 0 (§6).

### 5. Retention and storage

> **In simple words.** Storage costs money every month, so we keep only a few recent checkpoints, plus
> the ones we can never re-buy. Only a separate job may delete, so a bug in training can never wipe them.

| Tier | Keeps |
|---|---|
| Local NVMe | last 2 committed (fast restore on the same node) |
| S3 Standard | last 3 committed + **protected**: every N-th (Team 09 sets N), the final one, any pinned (`ckpt.py pin`) |
| S3 Glacier Flexible | protected checkpoints older than X days (≈ 6× cheaper per GB-month) |

- S3 **versioning** on, plus a lifecycle rule that aborts incomplete multipart uploads after 1 day.
- **Second copy of protected checkpoints:** S3 Replication to a bucket in a second region, filtered on the tag
  `v5:protected=true` (set by `ckpt.py pin` and by retention when it marks a checkpoint protected). Deletes are
  not replicated, so a wrong delete in the main bucket cannot remove the copy.
- Delete permission belongs **only** to the retention job's IAM role (`ckpt.py retention`, ADR 0002 §7).
- Retention never deletes a protected checkpoint or the one `latest.json` points to (CK-15 is HIGH).

### 6. Exit codes (shared with ADR 0002 and 0003)

> **In simple words.** One number tells Slurm what to do next: done, run again, or wait for a person.

| Code | Meaning | Slurm |
|---|---|---|
| 0 | done | complete |
| **75** | retry: the state is safe (just saved, or nothing to save); requeue, possibly on another node | `RequeueExit=75` |
| **78** | fatal: NaN, **loss spike**, failed checks, strict-load or environment error; retrying will not help | `RequeueExitHold=78`: hold and page |
| anything else | unexplained crash (exit 1, SIGKILL 137, segfault 139) | the job wrapper turns it into 75, or 78 after 3 crashes in 30 min |

**Loss-spike guard** (added after the week-1 review). A NaN check does not see a model that diverges with finite
numbers. If the loss is above **1.5 × the median of the last 100 steps** for **3 steps in a row**, the trainer
stops with 78, and **no checkpoint is written while a spike is in progress**, so `latest` stays before the spike.
The operator then rolls back (§9). Thresholds are flags (`--spike-ratio`, `--spike-steps`) and get tuned on the
dry runs; T12 checks that 300 normal steps never trip it. Metric H-24 (HIGH).

Changed from the first draft, which used 1 for fatal: an uncaught Python error (for example from
`TORCH_NCCL_ASYNC_ERROR_HANDLING`, ADR 0003 §2) also exits 1, and most of those are one-off faults worth one retry.

### 7. Contracts with other teams

| Team | Contract |
|---|---|
| 01/02 Data | the loader exposes `state_dict()` / `load_state_dict()` returning JSON-serialisable state, and `next_batch_id()` with a content hash of the next batch. The ledger offset is the authority on the data position |
| 05 Architecture | every tensor is trainable, frozen or a buffer, with a stable name. The parallel layout is given: ZeRO/FSDP stage, expert-parallel size. **Experts are stored by global expert id** (DTensor or an explicit index), so the checkpoint reshards when the EP size changes. Reversible models keep dropout at 0 |
| 04 KE2 | Kronecker tables and other non-trained tensors are declared frozen buffers and hashed in the manifest |
| 05 + 06 Kernels | **every kernel has a deterministic mode**: no unordered atomics, a fixed reduction order, deterministic reversible recompute. Resume tests and the CI restore probe run in deterministic mode; production may run fast mode. A kernel without a deterministic mode is flagged in its docstring and in the manifest |
| 06 Performance | owns the **GPU-to-host stall time** against our budget (≤ 2% of the interval); Team 07 owns correctness |
| 09 Eval, 12 Quant, 11 Release | each reads the S3 path from `checkpoint.committed`, read-only, never on a training node. **Team 07 owns the `dcp → safetensors` export**, with the same strict-key and probe-loss checks: **bit-equal** when the dtype is unchanged; a dtype change (quantization) belongs to Team 12, which states its own tolerance |
| 08 Monitoring | ships `checkpoint.*`, `restore.*` and `export.*` events and pages on the HIGH alerts in ADR 0004 |

### 8. Restore checks

> **In simple words.** Four independent proofs that the restored training is the saved training. Any one
> failing stops us from continuing on a wrong model, which is how V4 wasted a run.

| Check | Rule |
|---|---|
| **State digest** | sha256 over the raw bytes of every tensor (weights, optimizer, buffers) in sorted key order, at save and after restore; **must be equal**. After resharding it is computed on the logical tensors, so it still must be equal |
| **Probe loss** | loss of one fixed held-out probe batch, in deterministic mode, stored **per device class** (`cpu`, `cuda:<model>`) because CPU and GPU arithmetic differ; compared as raw float bits, not rounded |
| **Next batch** | `next_batch_id()` and its content hash equal the saved ones |
| **Frozen hashes** | each equals the manifest |

Any failure prints both values, emits `restore.failed`, and the next attempt uses the previous commit.

### 9. Rollback (operator action, not crash resume)

One job script supports `--from-step K --skip N --branch <id>`: it restores step K, adds the next N batch
ids to the skip list, and continues on a new branch. The run can therefore be replayed exactly later
(Session 6: resume / replay / fork). The simulation does this after the NaN drill.

### 10. Metrics

All checkpoint metrics, with LOW / MEDIUM / HIGH thresholds, are in `metrics/catalog.yaml` (CK-01 … CK-20)
and in [metrics_catalog.md](../metrics_catalog.md). The ones that page immediately (HIGH):

| ID | Metric | HIGH when |
|---|---|---|
| CK-01/02 | Time / dollars at risk since the last commit | ≥ 2 × interval |
| CK-03 | Consecutive failed saves | ≥ 2 |
| CK-04 | Checksum mismatch | any |
| CK-05 | Restore check failure | any |
| CK-06 | Strict-load or environment refusal | any |
| CK-07 | NaN guard trip | any |
| CK-08 | Save stall | ≥ 5% of the interval |
| CK-13 | Restore time | ≥ 30 min |
| CK-15 | Protected checkpoint missing | any |

Targets: mean work lost ≤ interval/2 + stall; RTO ≤ 15 min; commit latency ≤ 10 min; stall ≤ 2% of interval;
0 restores from an uncommitted checkpoint; export probe bit-equal.

## Consequences

Good:

- A half-written or damaged checkpoint cannot be chosen, and a bad newest checkpoint falls back by itself.
- A silent misload fails within one forward pass, with both numbers printed.
- A checkpoint saved on 4 ranks restores on 2 or 1 (tested), so losing a node does not stop the run.
- A warned interruption (preemption, spot, health drain) loses ≈ 0 work thanks to the emergency save.
- Other teams get one checked export format.

Bad and accepted:

- **Host RAM:** async staging needs pinned host memory for one per-node shard.
- **Kernel cost:** every custom kernel needs a deterministic mode (extra work for Teams 05/06, and slower).
- **Resharded continuation is not bit-equal:** after a world-size change the reduction order changes. The
  restored state is still digest-checked, and the branch is marked `<branch>-resharded-w<N>`.
- **Probe per device class:** the first save on new hardware records its own value (CK-20).
- **Extra code:** we own the manifest, verify, commit, retention and export code, and their tests.

## Validation

`python -m pytest ckpt/test_ckpt.py` runs on CPU, with gloo for more than one rank, inside the
`v5-slurm-sim` image. **All 14 pass with the local store, and the 14 also pass with
`V5_TEST_REMOTE=s3://…` against the S3 emulator** (2026-10-27).

| Test | What it does | Result |
|---|---|---|
| T1 | straight run vs save → new process → restore, on 1 and 2 ranks: loss bit-equal at every step; digest and next batch checked | pass |
| T2 | save on 4 ranks, restore on 1: digest equal, CPU probe bit-equal, marked resharded | pass |
| T3 | one flipped byte in the newest checkpoint: verify fails, previous checkpoint used | pass |
| T4 | process dies after the upload, before the manifest: `latest.json` unchanged; retention removes the orphan | pass |
| T5 | renamed key (the V4 converter bug): strict load aborts and names both keys | pass |
| T6 | SIGUSR1 to one of 2 ranks: all ranks save, exit 75; continuation bit-equal to a run never stopped | pass |
| T7 | NaN: no save, exit 78 | pass |
| T8 | export to safetensors: strict keys pass, probe loss bit-equal | pass |
| T9 | a changed config refuses to resume (78); `--allow-env-change` resumes and logs it | pass |
| T10 | rollback `--from-step K --skip 3 --branch b1` | pass |
| T11 | loss × 10 for 5 steps (no NaN): no save during the spike, exit 78, `latest` before the spike | pass |
| T12 | 300 normal steps never trip the spike guard | pass |
| T13 | the data-loader contract (`ckpt/loader_contract.py`) that Teams 01/02's real loader must pass | pass |

Simulation drills that exercise the checkpoint in Slurm (ADR 0002 §10): kill **during** a save (the half-uploaded
checkpoint is never used), kill **during** a restore (the next attempt restores the same checkpoint cleanly),
restore onto a smaller world, retention and export, all against the S3 emulator.

The same code runs inside the Docker Slurm cluster in `sim/run_sim.py` (ADR 0002 §Validation).
GPU: `V5_DEVICE=cuda` runs T1–T10 with the trainer on CUDA in deterministic mode (`CUBLAS_WORKSPACE_CONFIG` set),
and `ckpt/bench_save.py` measures the GPU-to-host stall and `dcp.async_save` times. Both run on Kaggle 2×T4 via
`kaggle/run_gpu_checks.py` (see `kaggle/README.md`). Week 2: the same on the B200 node for a randomly initialised
model of the final shape.

## Open points

- **Not tested yet:** truly sharded state. The test trainer uses DDP, where every rank holds a full copy, so
  "4 ranks → 2" is the easy case. FSDP/ZeRO-3 and expert-parallel resharding need Team 05's model (week 2).
- Real S3 throughput and `head` verification on a multi-GB checkpoint (week-2 node; the emulator is in-memory).
- Parallel layout: ZeRO-2 on 32 GPUs or ZeRO-3 on 8, and the EP degree. This sets the per-node share and the NVMe size.
- Protected cadence N (Team 09).
- The RPO cap the instructor accepts (start: 30 min).
- Model size and precision (config_v0.yaml, 31 Oct), which set the checkpoint size and stall.
