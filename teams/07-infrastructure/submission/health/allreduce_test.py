"""All-reduce test: correctness and bus bandwidth (ADR 0003 §1).

In simple words: every GPU adds up a known set of numbers with every other GPU. If the answer is
wrong, the hardware or network is broken. If it is slow, the node would slow down the whole training
run (and every slow minute is paid for), so it fails the check.

Each rank fills its tensor with the whole number rank+1. The sum n(n+1)/2 is exact in fp32 in any
reduction order, so a wrong value is a fault, not rounding.

Run:  torchrun --nproc_per_node=<gpus> health/allreduce_test.py [--max-mb 1024]
      without GPUs it uses gloo on CPU and says so (backend: gloo).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mb", type=float, default=1)
    ap.add_argument("--max-mb", type=float, default=1024)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--floor-gbps", type=float, default=0.0, help="fail if busbw is below this at the largest size")
    ap.add_argument("--out", help="write the JSON result here (rank 0)")
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    r, n = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(dev)

    results, wrong = [], []
    size = int(a.min_mb * 2 ** 20)
    while size <= a.max_mb * 2 ** 20:
        x = torch.full((size // 4,), float(r + 1), dtype=torch.float32, device=dev)
        expect = n * (n + 1) / 2
        for _ in range(a.warmup):
            y = x.clone()
            dist.all_reduce(y)
        times = []
        for _ in range(a.iters):
            y = x.clone()
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(y)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        if not torch.all(y == expect).item():
            wrong.append(size)
        t = sorted(times)[len(times) // 2]
        algbw = size / t / 1e9
        busbw = algbw * 2 * (n - 1) / n          # nccl-tests convention
        results.append({"bytes": size, "time_s": round(t, 6), "algbw_gbps": round(algbw, 3),
                        "busbw_gbps": round(busbw, 3)})
        size *= 2

    ok_vals = torch.tensor([0 if not wrong else 1], device=dev)
    dist.all_reduce(ok_vals)
    status = "pass"
    detail = "exact sums"
    if ok_vals.item():
        status, detail = "fail", f"wrong sum at sizes {wrong} on some rank"
    elif a.floor_gbps and results and results[-1]["busbw_gbps"] < a.floor_gbps:
        status, detail = "fail", f"busbw {results[-1]['busbw_gbps']} < floor {a.floor_gbps}"
    out = {"name": "nccl_allreduce", "status": status, "backend": "nccl" if cuda else "gloo", "ranks": n,
           "detail": detail, "metrics": {"sizes": results}}
    if r == 0:
        print(json.dumps(out))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(out, f)
    dist.destroy_process_group()
    return 0 if status == "pass" else 4


if __name__ == "__main__":
    sys.exit(main())
