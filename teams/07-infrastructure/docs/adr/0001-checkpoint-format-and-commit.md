# ADR 0001: Checkpoint format, commit protocol and restore checks

- Status: Proposed
- Date: 2026-10-25
- Area: Checkpoint (`docs/checkpoint_design.md`, `ckpt/ckpt.py`)

## Context

The lab's goal for Team 07 is a cluster that saves, restores and restarts the main run with no lost work.
Interruptions are routine at this scale: the reference run in Session 15 had one every 12.9 h and lost 5.6%
of its time; Llama 3 had 466 in 54 days. Every hour lost to a bad restart comes straight out of goodput.

V4 (LightningLM 0.1V, repo `The-School-of-AI/LLM`) ran on one 8-GPU node with DeepSpeed checkpoints. Its
code picks "latest" as the highest step in the S3 listing, has no checksums, saves CUDA RNG for the current
device only, and checks resume by eye. During the run, V4 hit:

- a checkpoint damaged during save (step 3009, rolled back to 2909);
- a silent misload: the converter wrote the wrong names, and the load left weights random while every cheap
  check passed (step-1 loss 2.95 against 2.55);
- library drift on a new machine.

V5 is larger and will run on several nodes, so the world size can change when a node is lost.

## Decision drivers

1. A restore must be provably exact: the week-1 pass condition is a **bit-equal loss on CPU**.
2. A partial or damaged checkpoint must never be picked.
3. The training stall per save must be small (Young's formula: interval ≈ √(2 × stall × MTBF)).
4. A checkpoint must load on a different number of GPUs, and on one CPU process.
5. It must be usable by eval and quantization (Teams 9 and 12) without touching the training node.

## Options considered

| Option | For | Against |
|---|---|---|
| A. `torch.save` of a gathered state dict on rank 0 | simple; one file | gathers 1–7 TB to one rank (V4's OOM-during-gather failure); no parallel write; slow |
| B. DeepSpeed `save_checkpoint` (V4's choice) | proven in V4; sharded | tied to DeepSpeed's engine and ZeRO stage; resharding needs the separate `zero_to_fp32` step; no integrity layer |
| C. **`torch.distributed.checkpoint` (DCP)** with async save, plus our own manifest and commit layer | per-rank parallel writes; **reshards on load** to any world size, including 1 CPU; async staging to host RAM; works with FSDP2 and plain DDP; no lock-in to one engine | newer API; our manifest and verify layer is extra code |
| D. Megatron distributed checkpoint | reshards; used at scale | brings in Megatron-Core, which we may not use for the model |

## Decision

**C. DCP for the tensor data, wrapped in a commit protocol that we own.**

1. **State:** weights, fp32 master weights, optimizer state, scheduler, scaler, **every** RNG (Python, NumPy,
   torch CPU, CUDA for each local device), sampler and ledger offset, the rollback skip list, run, branch and
   step ids, the environment (git sha, image digest, torch/CUDA/NCCL versions), and frozen-weight hashes.
2. **Save:** rank 0 decides when to save and broadcasts the decision (kept from V4). A NaN or Inf guard means a
   bad state is never saved. The only stall is DCP's async stage to pinned host RAM. In the background: write to
   `step_k.tmp/`, fsync, hash every file into `MANIFEST.json`, then **atomically rename** to `step_k/`. A
   separate uploader process (kept from V4) uploads the files first and the manifest last, re-checks them in S3,
   and only then writes `latest`. One save is in flight at a time.
3. **Restore:** read `latest` (never a listing), verify every sha256, check the environment against the
   manifest, then do a **strict** DCP load (any missing, extra, wrong-shape or wrong-dtype key aborts). Then
   restore every RNG and the data position, and recompute the loss of a fixed **probe batch**; it must be
   bit-equal to the value stored in the manifest at save time.
4. **Retention:** local NVMe keeps the last 2; S3 keeps the last 3 plus protected checkpoints (every N-th,
   the final one, any pinned); protected checkpoints move to Glacier by a lifecycle rule. Before a save, check
   free space ≥ 1.2 × checkpoint size.
5. **Exit codes** shared with the queue: 0 done, **75 retry** (save done, requeue), 1 fatal (NaN, failed verify;
   do not requeue).
6. **Start interval: 30 min.** It is replaced by Young's formula once the week-2 node and the dry runs give a
   measured stall time and MTBF.

## Consequences

Good:

- A half-written or damaged checkpoint cannot be chosen.
- A silent misload fails in one forward pass, with both numbers printed.
- A checkpoint saved on 64 GPUs restores on 56 GPUs or on 1 CPU process.
- Training stalls only for the device→host copy.

Bad and accepted:

- **Host RAM:** async staging needs pinned host memory of one per-node shard, about 14 B per trained parameter
  ÷ number of nodes.
- **Determinism costs speed:** bit-equal continuation needs deterministic algorithms (some kernels are slower)
  and a stateful sampler.
- **Resharded continuation is not bit-equal:** after a world-size change, continued training differs because the
  reduction order changes. The restored state is still checked bit-equal with the probe; such a branch is marked
  `resume-resharded`.
- **Extra code to own:** the manifest, verify and commit layer, with a test for each failure mode (T1–T7 below).

## Validation

These tests are in `ckpt/ckpt.py` and run on CPU with the gloo backend:

| Test | What it does |
|---|---|
| T1 | straight run vs save → new process → restore: loss bit-equal at every step |
| T2 | save on 4 ranks, restore on 1: probe loss bit-equal |
| T3 | one flipped byte → verify fails → falls back to the previous checkpoint |
| T4 | kill between write and seal → `latest` unchanged |
| T5 | renamed key → strict load aborts and names the key |
| T6 | SIGUSR1 → save and exit 75 |
| T7 | NaN → no save, exit 1 |

Week 2: the same tests on GPU, plus measured stall, write and upload times for a randomly initialised model of
the final shape.

## Open points

The parallel layout (ZeRO-2 on 32 GPUs or ZeRO-3 on 8; the expert-parallel degree), the S3 bucket and IAM,
the cadence for protected checkpoints (ask Team 9), and spot vs on-demand.
