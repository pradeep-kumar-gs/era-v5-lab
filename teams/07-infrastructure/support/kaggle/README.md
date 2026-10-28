# GPU validation on Kaggle (2 × T4, free)

In simple words: a Mac has no NVIDIA GPU, so the Docker Slurm test ran everything on CPU. This folder runs
the GPU-only parts on Kaggle's free 2-GPU machine. Total time is about 20–30 minutes.

## Steps

1. kaggle.com → **Create → New Notebook**.
2. Right panel → **Settings**: *Accelerator* = **GPU T4 x2**, *Internet* = **on** (phone verification needed once).
3. Get the code into the notebook. Pick one:
   - the branch is pushed to GitHub:
     ```
     !git clone -b infrastructure --depth 1 https://github.com/<your-fork>/era-v5-lab.git
     %cd era-v5-lab
     ```
   - or zip `teams/07-infrastructure`, upload it as a Kaggle **Dataset**, add it to the notebook, then
     `!cp -r /kaggle/input/<dataset>/07-infrastructure /kaggle/working/ && cd /kaggle/working`
     and run the script from there (the path below changes to `07-infrastructure/support/kaggle/...`).
4. Run one cell:
   ```
   !python teams/07-infrastructure/support/kaggle/run_gpu_checks.py
   ```
5. Download `/kaggle/working/gpu_results/summary.md` and `summary.json` (Output panel, right side), and add
   them to `teams/07-infrastructure/support/kaggle/results/`. The `*.log` files there help if a step failed.

No AWS keys, tokens or personal data are used or written.

## What each step proves

| Step | Proves | ADR |
|---|---|---|
| health_prolog / health_full | the health check reads real `nvidia-smi` values; missing DCGM/dmesg are `skipped`, never `pass` | 0003 §1 |
| allreduce | NCCL across 2 GPUs gives exact sums; measured bus bandwidth | 0003 §1 |
| nccl_timeout | a stuck GPU becomes a clean failure about 20 s after the timeout, with a flight-recorder trace | 0003 §2 |
| ckpt_tests_cuda | T1–T10 with the trainer on GPU: **bit-equal resume on CUDA** in deterministic mode | 0001 §8 |
| bench_save | GPU → host copy speed (the stall) and save speed, replacing the cost model's assumptions | 0001 §2, 0004 §1 |

## Not covered on Kaggle

T4 is a 2018 GPU on PCIe, not B200 with NVLink and EFA. Bandwidth and speed numbers are T4 numbers.
No DCGM, no kernel log (Xid), no row remapping (T4 does not support it), no Slurm. These wait for the
week-2 EC2 node.
