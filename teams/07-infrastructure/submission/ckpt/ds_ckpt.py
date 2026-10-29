"""DeepSpeed backend for the V5 checkpoint library (support/docs/deepspeed_impact.md, DS-1 to DS-7).

In simple words:
- DeepSpeed owns the training state (weights, ZeRO optimizer pieces), so DeepSpeed writes the files:
  `engine.save_checkpoint`. Everything around it stays ours: the commit protocol (upload, checksums,
  MANIFEST last, then latest.json), the four restore proofs, and the exit codes.
- DeepSpeed's own `latest` file is never written or read. Our manifest alone decides what counts.
- DeepSpeed's save is synchronous: the stall is the whole write to the node's disk, not just a copy to
  host memory. We measure it (stall_s) instead of claiming it is small.
- `load_checkpoint` returns (None, None), or fails an assert, when it cannot load. That must never become
  a fresh start (it would train from step 0 over good checkpoints): here it is an IntegrityError, so
  restore falls back to the previous commit.
- ZeRO optimizer pieces depend on the world size. A resume at another world size is refused
  (ReshardError, exit 78): an operator converts the checkpoint with ds_to_universal.py first.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import threading

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ckpt.ckpt import (MANIFEST, SCHEMA, Checkpointer, IntegrityError, NonFiniteError, ReshardError,  # noqa: E402
                       RestoreCheckError, StrictLoadError, EnvMismatchError, ENV_MUST_MATCH, _fingerprint_piece,
                       all_finite, all_true, barrier, broadcast_obj, device_class, env_manifest, fingerprints,
                       float_bits, key_spec, local_rank, rank, rng_capture, rng_restore, tensor_items, world)
from common.events import Timer, emit  # noqa: E402

DS_SCHEMA = SCHEMA + "+deepspeed/1"
_OWNER = (re.compile(r"zero_pp_rank_(\d+)_"), re.compile(r"^extra_rank(\d+)\.json$"))


def is_deepspeed(model) -> bool:
    return hasattr(model, "save_checkpoint") and hasattr(model, "load_checkpoint") and hasattr(model, "module")


def ds_env(engine) -> dict:
    """DS-7: the DeepSpeed version and its config change the numbers, so they must match to resume."""
    import deepspeed
    cfg = json.dumps(engine.config, sort_keys=True, default=str)
    return {"deepspeed": deepspeed.__version__, "ds_config_sha256": hashlib.sha256(cfg.encode()).hexdigest(),
            "zero_stage": engine.zero_optimization_stage()}


def ds_digest(engine) -> tuple:
    """(digest, model state). Weights are a full copy on every rank under ZeRO 0-2, so the usual logical
    fingerprint works. The ZeRO optimizer state is this rank's own piece: each rank fingerprints its
    piece and the digest lists them in rank order (a digest of this world size, by design: DS-5)."""
    msd = {"model": engine.module.state_dict()}
    model_fp = fingerprints(msd)
    osd = engine.optimizer.state_dict() if engine.optimizer is not None else {}
    mine = hashlib.sha256()
    for k, t in tensor_items(osd):
        a, b = _fingerprint_piece(t, (0,) * t.dim(), tuple(t.shape)).tolist()
        mine.update(f"{k}|{t.dtype}|{tuple(t.shape)}|{a:08x}-{b:08x};".encode())
    parts = [None] * world()
    if dist.is_initialized():
        dist.all_gather_object(parts, mine.hexdigest())
    else:
        parts = [mine.hexdigest()]
    h = hashlib.sha256()
    for k, t in tensor_items(msd):
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|{model_fp[k]};".encode())
    for r, p in enumerate(parts):
        h.update(f"optim_rank{r}|{p};".encode())
    return h.hexdigest(), msd, osd


class DSCheckpointer(Checkpointer):
    """Same commit protocol and restore proofs as Checkpointer; DeepSpeed writes and reads the tensors."""

    def save(self, step: int, model, optim, loader, probe_fn, tokens: int = 0, reason: str = "interval") -> bool:
        t = Timer()
        if rank() == 0:
            emit("checkpoint.started", step=step, reason=reason, branch=self.branch, backend="deepspeed")
        waited = 0.0
        if self._thread is not None and self._thread.is_alive():
            tw = Timer()
            self._thread.join()
            waited = tw.s()
            if rank() == 0:
                emit("checkpoint.backpressure", step=step, waited_s=waited)

        digest, msd, osd = ds_digest(model)
        if not all_true(all_finite(msd) and all_finite(osd)):
            if rank() == 0:
                emit("checkpoint.failed", step=step, cause="nonfinite", stall_s=t.s())
            raise NonFiniteError(f"NaN/Inf in the state at step {step}; not saved")

        save_dir = os.path.join(self.local_run, self.branch)
        tag = f"step_{step:08d}"
        local_dir = os.path.join(save_dir, tag)
        os.makedirs(save_dir, exist_ok=True)
        free = shutil.disk_usage(save_dir).free
        need = self.last_rank_bytes * int(os.environ.get("SLURM_NTASKS_PER_NODE", "1")) * self.nvme_headroom
        if not all_true(free >= need):
            if rank() == 0:
                emit("checkpoint.failed", step=step, cause="nvme_full", free_bytes=free, need_bytes=need)
            return False

        frozen = fingerprints(msd, {f"model.{k}" for k in self.frozen_keys})
        probe = float_bits(probe_fn(model))
        ld = loader.state_dict()
        loader_global = all_true(json.dumps(ld, sort_keys=True) == broadcast_obj(json.dumps(ld, sort_keys=True)))

        # Synchronous: every rank writes its ZeRO piece, data-parallel rank 0 writes the weights. All ranks
        # must call it (a rank-0-only call hangs). save_latest=False: DeepSpeed's `latest` is not our pointer.
        tw = Timer()
        model.save_checkpoint(save_dir, tag=tag, client_state={"v5_step": step}, save_latest=False)
        write_s = tw.s()
        extra = {"step": step, "tokens": tokens, "world_size": world(), "rank": rank(),
                 "rng": rng_capture(), "loader": ld}
        with open(os.path.join(local_dir, f"extra_rank{rank()}.json"), "w") as f:
            json.dump(extra, f)

        meta = None
        if rank() == 0:
            env = dict(env_manifest(self.config_path), **ds_env(model))
            meta = {
                "schema": DS_SCHEMA, "backend": "deepspeed", "run_id": self.run_id, "branch": self.branch,
                "step": step, "tokens": tokens, "world_size": world(), "reason": reason, "env": env,
                "keys": key_spec(msd), "state_digest": digest,
                "frozen": {k[len("model."):]: v for k, v in frozen.items()},
                "probe_loss": {device_class(model.module): probe},
                "next_batch": loader.next_batch_id(), "loader_global": loader_global,
                "async_save": False, "ds_write_s": write_s,
            }
        stall = t.s()
        self._ok = True
        self._thread = threading.Thread(target=self._upload_and_commit,
                                        args=(step, local_dir, meta, stall, waited, None), daemon=True)
        self._thread.start()
        return True

    def _own_files(self, local_dir: str) -> list:
        """zero_pp_rank_<r>_* and extra_rank<r>.json belong to rank r; the weights file to rank 0."""
        r = rank()
        out = []
        for n in os.listdir(local_dir):
            if not os.path.isfile(os.path.join(local_dir, n)) or n.endswith(".part"):
                continue
            owner = next((int(m.group(1)) for m in (p.search(n) for p in _OWNER) if m), 0)
            if owner == r:
                out.append(n)
        return sorted(out)

    def _restore_one(self, c, model, optim, loader, probe_fn, skip, allow_env_change) -> dict:
        t = Timer()
        st = self.store
        rel = c["path"]
        man = st.get_json(f"{rel}/{MANIFEST}")
        if man.get("backend") != "deepspeed":
            raise StrictLoadError(f"step {man['step']} was not saved by DeepSpeed; this run uses DeepSpeed")
        if rank() == 0:
            emit("restore.started", step=man["step"], path=rel, backend="deepspeed")

        # 1. Checksums in S3. The .pt files are pickles (they can run code when loaded): this check is a
        #    security control too, so it runs before anything is unpickled.
        tv = Timer()
        bad = None
        if rank() == 0:
            bad = [n for n, s in man["files"].items() if st.sha256(f"{rel}/{n}", self.verify_mode) != s["sha256"]]
        bad = broadcast_obj(bad)
        if bad:
            raise IntegrityError(f"step {c['step']}: checksum mismatch in {bad}")
        verify_s = tv.s()

        # 2. Environment, now including the DeepSpeed version and config.
        now = dict(env_manifest(self.config_path), **ds_env(model))
        keys = tuple(ENV_MUST_MATCH) + ("deepspeed", "ds_config_sha256", "zero_stage")
        diff = {k: [man["env"].get(k), now.get(k)] for k in keys if man["env"].get(k) != now.get(k)}
        if diff and not allow_env_change:
            raise EnvMismatchError(f"environment differs from the checkpoint: {diff}")

        # 3. ZeRO pieces are per rank: another world size needs the operator's universal conversion.
        if man["world_size"] != world():
            raise ReshardError(f"step {c['step']} was saved by DeepSpeed at world size {man['world_size']}; "
                               f"resume at {world()} needs ds_to_universal.py first (an operator job)")

        # 4. Download, checking each copy.
        td = Timer()
        save_dir = os.path.join(self.local_run, "restore", c["branch"])
        tag = os.path.basename(rel)
        local_dir = os.path.join(save_dir, tag)
        if local_rank() == 0:
            os.makedirs(local_dir, exist_ok=True)
            for n, s in man["files"].items():
                sha, _ = st.get_file(f"{rel}/{n}", os.path.join(local_dir, n))
                if sha != s["sha256"]:
                    raise IntegrityError(f"local copy of {n} is damaged")
        barrier()
        download_s = td.s()

        # 5. Strict key check on the weights.
        want, have = man["keys"], key_spec({"model": model.module.state_dict()})
        missing = sorted(set(want) - set(have))
        unexpected = sorted(set(have) - set(want))
        changed = sorted(k for k in set(want) & set(have) if want[k] != have[k])
        if missing or unexpected or changed:
            raise StrictLoadError(f"strict load failed: missing in model={missing} "
                                  f"not in checkpoint={unexpected} shape/dtype differs={changed}")

        # 6. DeepSpeed loads the tensors. It says "could not load" two ways: it returns None (a warning only)
        #    or an assert fails (a file is missing). Both mean: try the previous commit, never start at step 0.
        tl = Timer()
        try:
            path, _ = model.load_checkpoint(save_dir, tag=tag, load_module_strict=True,
                                            load_optimizer_states=True, load_lr_scheduler_states=True)
            why = None if path is not None else "returned None"
        except (AssertionError, FileNotFoundError, KeyError, RuntimeError) as e:
            why = f"{type(e).__name__}: {e}"[:300]
        if not all_true(why is None):
            raise IntegrityError(f"step {c['step']}: DeepSpeed load_checkpoint could not load ({why or 'on another rank'})")
        with open(os.path.join(local_dir, f"extra_rank{rank()}.json")) as f:
            extra = json.load(f)
        loader.load_state_dict(extra["loader"])
        rng_restore(extra["rng"])
        load_s = tl.s()

        # 7. The four proofs: digest, probe loss bits, next batch, frozen tables.
        tc = Timer()
        problems = []
        digest, msd, _ = ds_digest(model)
        if digest != man["state_digest"]:
            problems.append(f"state digest saved={man['state_digest']} restored={digest}")
        dc = device_class(model.module)
        probe_now = float_bits(probe_fn(model))
        probe_saved = man["probe_loss"].get(dc)
        if probe_saved is not None and probe_now != probe_saved:
            problems.append(f"probe loss bits saved={probe_saved} restored={probe_now} ({dc})")
        nb = loader.next_batch_id()
        if nb != man["next_batch"]:
            problems.append(f"next batch saved={man['next_batch']} restored={nb}")
        frozen_now = fingerprints(msd, {f"model.{k}" for k in man["frozen"]})
        for k, h in man["frozen"].items():
            if frozen_now.get(f"model.{k}") != h:
                problems.append(f"frozen tensor {k} fingerprint differs")
        if not all_true(not problems):
            raise RestoreCheckError(f"step {c['step']}: " + "; ".join(problems or ["check failed on another rank"]))
        checks_s = tc.s()

        if skip:
            loader.add_skip(skip)
        return {"step": man["step"], "tokens": man["tokens"], "branch": c["branch"], "from_path": rel,
                "resharded": False, "saved_world": man["world_size"], "world": world(),
                "env_override": sorted(diff) if diff else [], "skip": skip,
                "probe_recorded": probe_saved is not None, "store": st.kind, "backend": "deepspeed",
                "verify_s": verify_s, "download_s": download_s, "load_s": load_s, "checks_s": checks_s,
                "restore_s": t.s()}
