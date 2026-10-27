"""The data-loader contract every team's loader must pass before the main run (ADR 0001 §7).

In simple words: a checkpoint can only resume training exactly if the data loader can say where it is,
and later continue from exactly that place. Teams 01/02 own the real loader; this file is the test they run
on it. Our synthetic loader passes it today (see test at the bottom).

A loader must offer:
    state_dict()        -> JSON-serialisable dict (ledger offset, skip list, shard/worker positions)
    load_state_dict(d)  -> continue from that state
    next_batch_id()     -> {"position": ..., "sha256": <hash of the next batch's content>}, without consuming it
    next()              -> the next batch (tensor or anything hashable via bytes)
    add_skip(n)         -> skip the next n batches (rollback)

    python ckpt/loader_contract.py --factory mypkg.loader:make_loader     (factory(rank, world) -> loader)
"""
import argparse
import hashlib
import importlib
import json
import sys


def _bytes(b):
    if hasattr(b, "numpy"):
        return b.cpu().numpy().tobytes()
    return json.dumps(b, sort_keys=True, default=str).encode()


def check(factory, steps=20, cut=7, world=2) -> list:
    """Returns a list of problems; empty means the loader passes."""
    problems = []
    for r in range(world):
        a = factory(r, world)
        straight = [hashlib.sha256(_bytes(a.next())).hexdigest() for _ in range(steps)]
        b = factory(r, world)
        for _ in range(cut):
            b.next()
        st = b.state_dict()
        try:
            st = json.loads(json.dumps(st))
        except TypeError as e:
            problems.append(f"rank {r}: state_dict is not JSON-serialisable: {e}")
            continue
        peek = b.next_batch_id()
        if b.next_batch_id() != peek:
            problems.append(f"rank {r}: next_batch_id() changes when called twice (it must not consume)")
        c = factory(r, world)
        c.load_state_dict(st)
        if c.next_batch_id() != peek:
            problems.append(f"rank {r}: next_batch_id after load_state_dict differs from before save")
        resumed = [hashlib.sha256(_bytes(c.next())).hexdigest() for _ in range(steps - cut)]
        if resumed != straight[cut:]:
            first = next(i for i, (x, y) in enumerate(zip(resumed, straight[cut:])) if x != y)
            problems.append(f"rank {r}: resumed batch {cut + first} differs from the uninterrupted run")
        d = factory(r, world)
        d.load_state_dict(st)
        d.add_skip(2)
        skipped = [hashlib.sha256(_bytes(d.next())).hexdigest() for _ in range(3)]
        if skipped != straight[cut + 2:cut + 5]:
            problems.append(f"rank {r}: add_skip(2) does not skip exactly the next 2 batches")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", required=True, help="module:function, function(rank, world) -> loader")
    a = ap.parse_args()
    mod, fn = a.factory.split(":")
    probs = check(getattr(importlib.import_module(mod), fn))
    print(json.dumps({"pass": not probs, "problems": probs}, indent=1))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
