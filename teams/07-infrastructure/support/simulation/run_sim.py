"""Docker Slurm simulation: runs every drill from ADR 0002 §10 and checks the result.

In simple words: we start a small Slurm cluster in Docker, start a "main run" and some team jobs, then
break things on purpose, one at a time: kill a process (also in the middle of a save and of a restore), send
a preemption signal, fake a spot notice, fake a broken GPU (on one node, then on two at once), freeze a
process, pull the plug on a node, stop the Slurm controller, inject a NaN and a loss spike. After each one we
check that the main run came back by itself from its last checkpoint, and measure how long it took and how
much work was lost. Checkpoints go to an S3 emulator through the real S3 code. At the end,
metrics/evaluate.py turns the events into the metrics report, and metrics/watch.py (running all along) shows
which alerts fired live.

Run from the repo:  python3 teams/07-infrastructure/support/simulation/run_sim.py   (needs Docker; about 25 minutes)
Only the Python standard library is needed on the host.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSE = os.path.join(HERE, "slurm-docker", "docker-compose.yml")
RESULTS = os.path.join(HERE, "results")
DS = False          # --deepspeed: the trainer runs a DeepSpeed ZeRO engine (support/docs/deepspeed_impact.md)
RUN = "v5main"
PY = "/opt/venv/bin/python"
REMOTE = "s3://v5-ckpt/ckpt"

# Main-run settings for the test: steps are slowed to 50 ms so there is time to break things.
# Time is compressed: the save interval is 8 s instead of 30 min, so the crash-loop window (3 crashes in 30 min
# in production) is compressed to 10 s too; otherwise the drills themselves trip it (found in run 4: the third
# kill drill held the main run, which is the cap working as designed).
MAIN_ENV = {"RUN_ID": RUN, "STEPS": "100000", "INTERVAL_S": "8", "STEP_SLEEP": "0.05", "WATCHDOG_S": "20",
            "CRASHLOOP_WINDOW_S": "10"}
TEAM_ENV = {"STEPS": "100000", "INTERVAL_S": "5", "STEP_SLEEP": "0.05"}
# The job scripts default to EC2 (8 GPUs, 8 ranks per node); the Docker nodes have 2 placeholder GPUs.
MAIN_ARGS = "--ntasks-per-node=2 --gpus-per-node=2"
TEAM_ARGS = "--ntasks-per-node=1 --gpus-per-node=2"
REPEATS = 3

results = []
CTL = {"host": "slurmctld"}   # which controller container runs our commands (changes during failover)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dc(*args, check=True, capture=True):
    p = subprocess.run(["docker", "compose", "-f", COMPOSE, *args], capture_output=capture, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p


def sh(service, cmd, check=False):
    p = dc("exec", "-T", service, "bash", "-c", cmd, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"{service}: {cmd}\n{p.stdout}\n{p.stderr}")
    return p


def ctl(cmd, check=False):
    return sh(CTL["host"], cmd, check)


def sbatch(script, env=None, args="", user="svc"):
    exports = ",".join(["ALL"] + [f"{k}={v}" for k, v in (env or {}).items()])
    p = ctl(f"cd /tmp && runuser -u {user} -- sbatch --parsable --export={shlex.quote(exports)} {args} {script}")
    if p.returncode != 0:
        return None, (p.stdout + p.stderr).strip()
    return p.stdout.strip().split(";")[0], ""


def job(jid) -> dict:
    out = ctl(f"scontrol show job -o {jid}").stdout
    d = dict(re.findall(r"(\w+)=(\S*)", out))
    m = re.search(r"Comment=(.*?)(?= \w+=|$)", out)   # the plain-words status from submission/queue/job_queue.py (has spaces)
    if m:
        d["Comment"] = m.group(1)
    return d


def events() -> list:
    out = ctl("cat /s3/events/*.jsonl 2>/dev/null").stdout
    ev = []
    for line in out.splitlines():
        try:
            ev.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    ev.sort(key=lambda e: e["time"])
    return ev


def emit(event, **fields):
    args = json.dumps(fields)
    out = ctl(f"RUN_ID={RUN} V5_EVENTS_DIR=/s3/events {PY} -c \"import sys,json; sys.path.insert(0,'/opt/v5/submission'); "
              f"from common.events import emit; print(emit('{event}', **json.loads(sys.argv[1]))['time'])\" '{args}'")
    return out.stdout.strip()


def mark(drill, **fields):
    return emit("drill.injected", drill=drill, **fields)


def latest(run=RUN) -> dict:
    out = ctl(f". /etc/v5/node.env; {PY} -c \"import sys,json; sys.path.insert(0,'/opt/v5/submission'); "
              f"from ckpt.store import open_store; print(json.dumps(open_store('{REMOTE}/{run}').get_json('latest.json')))\"")
    return json.loads(out.stdout)


def drill_file(name, value=None):
    """Test-only switches the trainer reads from /s3/drills/<run>/ (ADR 0001 drill hooks)."""
    if value is None:
        ctl(f"rm -f /s3/drills/{RUN}/{name}")
    else:
        ctl(f"mkdir -p /s3/drills/{RUN} && echo {value} > /s3/drills/{RUN}/{name}")


def wait(pred, timeout, what, poll=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(poll)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")


def ev_after(t, name, run=RUN, **match):
    for e in events():
        if e["time"] > t and e["event"] == name and (run is None or e.get("run_id") == run) and \
                all(e.get(k) == v for k, v in match.items()):
            return e
    return None


def last_step_before(t):
    """Highest step of the current incarnation before t (redone steps after a restore are lower)."""
    ev = events()
    since = max([e["time"] for e in ev if e["event"] in ("restore.done", "train.started") and e["run_id"] == RUN
                 and e["time"] <= t], default="")
    steps = [e["step"] for e in ev if e["event"] == "train.step" and e["run_id"] == RUN and since <= e["time"] <= t]
    return max(steps) if steps else 0


def record(drill, ok, **detail):
    results.append({"drill": drill, "pass": bool(ok), **detail})
    log(f"{'PASS' if ok else 'FAIL'} {drill} {json.dumps(detail)}")


def run_drill(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - a failed drill is a result, not the end of the simulation
        record(name, False, error=str(e)[:400])


def main_job():
    out = ctl("squeue -h -n v5-mainrun -o '%A %T %N %r'").stdout.split()
    return (out[0], out[1], out[2] if len(out) > 3 else "", out[-1]) if out else (None, None, None, None)


def wait_recovered(t_fault, timeout=240):
    """The main run is back: a restore after the fault, then a first productive step."""
    rd = wait(lambda: ev_after(t_fault, "restore.done"), timeout, "restore.done")
    fs = wait(lambda: ev_after(rd["time"], "train.step", first_after_start=True), 60, "first step")
    return rd, fs


def node_of_main(index=0):
    jid, st, nodes, _ = wait(lambda: (lambda j: j if j[1] == "RUNNING" else None)(main_job()), 300, "main run RUNNING")
    hosts = ctl(f"scontrol show hostnames {nodes}").stdout.split()
    return jid, hosts[index], hosts


def wait_fresh_commit():
    """A checkpoint committed by the current incarnation, so a kill has something recent to fall back on."""
    ev = events()
    since = max([e["time"] for e in ev if e["event"] in ("restore.done", "train.started") and e["run_id"] == RUN],
                default="")
    return wait(lambda: ev_after(since, "checkpoint.committed"), 120, "fresh commit")


def kill_trainer(hosts):
    for h in hosts:
        sh(h, "pkill -9 -f '[t]rain.py --run-id v5main' ; true")


def resume_node(host):
    ctl(f"scontrol update nodename={host} state=resume")
    emit("node.resumed", node=host, by="drill")


def time_from_iso(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


# --------------------------------------------------------------------------- drills

def d_accounts():
    jid, err = sbatch("--wrap hostname", args="--account=arch -p debug")
    _, err2 = sbatch("--wrap hostname", args="-p debug", user="noacct")
    if err2:
        emit("job.rejected", user="noacct")
    record("accounts", jid is not None and "Invalid account" in err2,
           accepted_job=jid, rejected_no_account=err2[:120])


def d_preemption():
    team = []
    for acct in ("arch", "perf", "eval"):
        jid, err = sbatch("/opt/v5/submission/slurm/team_job.sbatch", dict(TEAM_ENV, RUN_ID=f"ladder-{acct}"),
                          f"--account={acct} {TEAM_ARGS}")
        team.append(jid)
    wait(lambda: all(job(j).get("JobState") == "RUNNING" for j in team), 120, "3 team jobs running")
    wait(lambda: ev_after("0", "train.step", run="ladder-arch"), 60, "team job stepping")
    t_submit = mark("preemption", team_jobs=team)
    jid, err = sbatch("/opt/v5/submission/slurm/mainrun.sbatch", MAIN_ENV, MAIN_ARGS)
    wait(lambda: job(jid).get("JobState") == "RUNNING", 180, "main run RUNNING")
    t_run = time.time()
    wait(lambda: ev_after(t_submit, "train.started"), 60, "main run training")
    stops = [e for e in events() if e["event"] == "train.stop_requested" and e["time"] > t_submit
             and e["run_id"].startswith("ladder-")]
    preempted = sorted({e["run_id"] for e in stops})
    record("preemption", len(preempted) >= 2, main_job=jid, preempted_team_runs=preempted,
           team_saved_before_exit=all(e.get("signal") == "SIGTERM" for e in stops),
           seconds_submit_to_running=round(t_run - time_from_iso(t_submit), 1))


def d_gpu_allocation():
    """GPUs are allocated and accounted as a resource (gres): the main run holds 2 nodes x 2 GPUs,
    and a job asking more GPUs than a node has is refused at submit time."""
    jid, _, _ = node_of_main(0)
    # The controller's live view (the accounting record is written a moment later).
    alloc = job(jid).get("AllocTRES", "")
    _, err = sbatch("--wrap hostname", args="--account=arch -p debug --gpus-per-node=3")
    record("gpu_allocation", "gres/gpu=4" in alloc and "not available" in err,
           main_run_alloc=alloc, over_ask_rejected=err[:100])


def d_kill(i):
    def f():
        jid, host, _ = node_of_main(0)
        wait_fresh_commit()
        t = mark("kill", target=host, cause_class="software", repeat=i)
        kill_trainer([host])
        rd, fs = wait_recovered(t)
        record(f"kill_process_{i}", rd["step"] > 0, node=host, restored_step=rd["step"],
               lost_steps=max(0, last_step_before(t) - rd["step"]))
    return f


def d_usr1(i):
    def f():
        jid, host, _ = node_of_main(0)
        t = mark("sigusr1", cause_class="preemption", repeat=i)
        ctl(f"scancel --signal=USR1 {jid}")
        stop = wait(lambda: ev_after(t, "train.stop_requested"), 60, "stop requested")
        rd, fs = wait_recovered(t)
        record(f"sigusr1_save_and_requeue_{i}", rd["step"] == stop["step"], saved_step=stop["step"],
               restored_step=rd["step"], lost_steps=rd["step"] - stop["step"])
    return f


def d_kill_during_save():
    """Kill while the background upload is still running: the half-uploaded checkpoint must never be used."""
    jid, host, hosts = node_of_main(0)
    wait_fresh_commit()
    before = latest()
    drill_file("upload_delay_s", 2)      # each file upload now takes 2 s, so the save stays in flight
    t = mark("kill_during_save", cause_class="software")
    started = wait(lambda: ev_after(t, "checkpoint.started"), 60, "a save to start")
    time.sleep(1.0)
    kill_trainer(hosts)
    drill_file("upload_delay_s")
    rd, fs = wait_recovered(t)
    inflight = started["step"]
    committed = [e for e in events() if e["event"] == "checkpoint.committed" and e["run_id"] == RUN
                 and e["step"] == inflight and e["time"] > t]
    record("kill_during_save", rd["step"] != inflight and not committed and rd["step"] >= before["step"],
           inflight_step=inflight, restored_step=rd["step"], previous_latest=before["step"])


def d_kill_during_restore():
    """Kill while a restore is running: the next attempt must restore cleanly from the same checkpoint."""
    jid, host, hosts = node_of_main(0)
    wait_fresh_commit()
    drill_file("restore_delay_s", 6)
    t = mark("kill_during_restore", cause_class="software")
    kill_trainer(hosts)
    rs = wait(lambda: ev_after(t, "restore.started"), 120, "restore started")
    time.sleep(2)
    _, _, hosts2 = node_of_main(0)
    kill_trainer(hosts2)
    drill_file("restore_delay_s")
    t2 = emit("drill.note", note="killed during restore")
    rd, fs = wait_recovered(t2)
    bad = [e for e in events() if e["event"] in ("restore.failed", "restore.fatal") and e["time"] > t]
    record("kill_during_restore", rd["step"] == rs["step"] and not bad, restore_interrupted_at=rs["step"],
           restored_step=rd["step"], restore_failures=len(bad))


def d_spot():
    jid, host, _ = node_of_main(1)
    t = mark("spot_notice", target=host, cause_class="host")
    sh(host, "echo 'terminate at +120s (drill)' > /var/lib/v5-health/spot_notice")
    drained = wait(lambda: ev_after(t, "node.drained", run=None, node=host), 60, "node drained")
    rd, fs = wait_recovered(t)
    _, _, nodes, _ = main_job()
    new_hosts = ctl(f"scontrol show hostnames {nodes}").stdout.split()
    sh(host, "rm -f /var/lib/v5-health/spot_notice")
    resume_node(host)
    record("spot_notice", host not in new_hosts and rd["step"] > 0, node=host, drained=bool(drained),
           new_nodes=new_hosts, lost_steps=max(0, last_step_before(t) - rd["step"]))


def d_health():
    jid, host, _ = node_of_main(0)
    t = mark("health_fail", target=host, cause_class="gpu")
    sh(host, "echo 'drill: GPU 3 Xid 79 (fallen off the bus)' > /var/lib/v5-health/force_fail")
    drained = wait(lambda: ev_after(t, "node.drained", run=None, node=host), 90, "node drained by health check")
    rd, fs = wait_recovered(t)
    _, _, nodes, _ = main_job()
    new_hosts = ctl(f"scontrol show hostnames {nodes}").stdout.split()
    sh(host, "rm -f /var/lib/v5-health/force_fail")
    resume_node(host)
    record("health_fail_drain_and_replace", host not in new_hosts, node=host, reason=drained.get("reason"),
           new_nodes=new_hosts, lost_steps=max(0, last_step_before(t) - rd["step"]))


def d_two_nodes_down():
    """Two nodes fail and there is one spare: the main run cannot start. It must wait (no crash loop,
    no partial start), the Q-01 alert must show the wait, and it must come back once nodes return."""
    jid, _, hosts = node_of_main(0)
    t = mark("two_nodes_down", targets=hosts, cause_class="gpu")
    for h in hosts:
        sh(h, "echo 'drill: two nodes down' > /var/lib/v5-health/force_fail")
    for h in hosts:
        wait(lambda: ev_after(t, "node.drained", run=None, node=h), 90, f"{h} drained")
    pend = wait(lambda: (lambda j: j if j[1] == "PENDING" else None)(main_job()), 120, "main run pending")
    time.sleep(30)
    still = main_job()
    t_back = emit("drill.note", note="nodes repaired")
    for h in hosts:
        sh(h, "rm -f /var/lib/v5-health/force_fail")
        resume_node(h)
    rd, fs = wait_recovered(t_back, timeout=300)
    record("two_nodes_down_waits_then_recovers", still[1] == "PENDING" and rd["step"] > 0,
           drained=hosts, pending_reason=still[3], waited_s=round(time_from_iso(t_back) - time_from_iso(t)),
           restored_step=rd["step"])


def d_hang():
    jid, host, _ = node_of_main(1)
    t = mark("hang", target=host, cause_class="software")
    sh(host, "pkill -STOP -f '[t]rain.py --run-id v5main' ; true")
    wd = wait(lambda: ev_after(t, "train.watchdog"), 90, "watchdog")
    rd, fs = wait_recovered(t)
    record("hang_watchdog", bool(wd), node=host, watchdog_idle_s=wd["idle_s"], restored_step=rd["step"],
           lost_steps=max(0, last_step_before(t) - rd["step"]))


def d_node_loss():
    jid, host, _ = node_of_main(1)
    t = mark("node_loss", target=host, cause_class="host")
    dc("kill", host, check=False)
    rd, fs = wait_recovered(t, timeout=300)
    _, _, nodes, _ = main_job()
    new_hosts = ctl(f"scontrol show hostnames {nodes}").stdout.split()
    dc("start", host, check=False)
    wait(lambda: any(s in ctl(f"sinfo -h -n {host} -o %T").stdout for s in ("idle", "alloc", "mix")),
         120, f"{host} back")
    record("node_loss", host not in new_hosts, node=host, new_nodes=new_hosts,
           lost_steps=max(0, last_step_before(t) - rd["step"]))


def d_ctld_failover():
    """Stop the primary Slurm controller: the backup takes over, training keeps stepping, new jobs are
    accepted; then the primary comes back and takes control again."""
    jid, _, _ = node_of_main(0)
    t = mark("ctld_failover", cause_class="host")
    dc("stop", "slurmctld", check=False)
    CTL["host"] = "slurmctld2"
    took = wait(lambda: "is UP" in ctl("scontrol ping").stdout and ctl("squeue -h -o %A").returncode == 0,
                180, "backup controller in control")
    t_took = emit("ctld.backup_in_control", ctld="slurmctld2")
    stepping = wait(lambda: ev_after(t, "train.step"), 60, "training still stepping")
    jt, err = sbatch("--wrap hostname", args="--account=arch -p debug")
    accepted = jt is not None and wait(lambda: job(jt).get("JobState") == "COMPLETED", 120, "job via backup")
    dc("start", "slurmctld", check=False)
    CTL["host"] = "slurmctld"
    back = wait(lambda: "Slurmctld(primary) at slurmctld is UP" in ctl("scontrol ping").stdout
                and ctl("squeue -h -o %A").returncode == 0, 240, "primary back")
    still = main_job()
    record("controller_failover", bool(took and stepping and accepted and back) and still[1] == "RUNNING",
           takeover_s=round(time_from_iso(t_took) - time_from_iso(t), 1), job_via_backup=jt,
           main_run_state_after=still[1])


def d_db_backup():
    """Back up the accounting database, restore it into a scratch database, compare table counts."""
    out = sh("slurmdbd", "DB_ROOT_PASSWORD=simonly bash /opt/v5/submission/slurm/backup_db.sh /s3/backups && "
                         "DB_ROOT_PASSWORD=simonly bash /opt/v5/submission/slurm/backup_db.sh --verify-latest /s3/backups")
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    res = json.loads(lines[-1]) if lines else {}
    if res:
        emit("db.backup", **res)
    ok = res.pop("ok", None) is True
    record("db_backup_restore", ok, **res)


def d_nan_rollback_reshard():
    jid, host, _ = node_of_main(0)
    cur = last_step_before("9999")
    nan_at = cur + 25
    t = mark("nan", nan_at_step=nan_at, cause_class="software")
    drill_file("nan_at_step", nan_at)
    held = wait(lambda: ev_after(t, "job.held"), 120, "job held")
    st = job(jid)
    lt = latest()
    record("nan_guard_hold", lt["step"] <= nan_at and held["cause"] == "nonfinite", nan_at_step=nan_at,
           latest_committed=lt["step"], job_state=st.get("JobState"), reason=st.get("Reason"))
    # Monitoring: the held job explains itself in Slurm, in plain words, with the next step.
    c = st.get("Comment", "")
    record("held_job_explains_itself", "HELD (78)" in c and "NaN" in c and "roll back" in c, comment=c)
    # Operator rollback (ADR 0001 §9): from the last good step, skip 5 batches, new branch,
    # and on half the world size (1 node instead of 2): the restore must reshard.
    drill_file("nan_at_step")
    ctl(f"scancel {jid}")
    t2 = mark("rollback", from_step=lt["step"], cause_class="operator")
    env = dict(MAIN_ENV, STEPS=str(lt["step"] + 300),
               TRAIN_EXTRA=f"--from-step {lt['step']} --skip 5 --branch b1")
    jid2, err = sbatch("/opt/v5/submission/slurm/mainrun.sbatch", env, f"{MAIN_ARGS} --nodes=1")
    if DS:
        # DeepSpeed: ZeRO pieces are per rank, so a smaller world must be REFUSED (exit 78, ReshardError) until an
        # operator converts the checkpoint (ds_to_universal.py). Then the rollback runs again at the saved size.
        held = wait(lambda: ev_after(t2, "job.held"), 180, "job held on world change")
        fatal = ev_after(t2, "restore.fatal") or {}
        record("restore_smaller_world", held["cause"] == "ReshardError" and "ds_to_universal" in fatal.get("detail", ""),
               cause=held["cause"], detail=fatal.get("detail", "")[:160])
        ctl(f"scancel {jid2}")
        t2 = mark("rollback", from_step=lt["step"], cause_class="operator")
        jid2, err = sbatch("/opt/v5/submission/slurm/mainrun.sbatch", env, MAIN_ARGS)
    rd, fs = wait_recovered(t2)
    record("rollback_skip_branch", rd["skip"] == 5 and rd["branch"] == "b1", restored_step=rd["step"],
           branch=rd["branch"])
    if not DS:
        record("restore_smaller_world", rd["resharded"] and rd["saved_world"] == 4 and rd["world"] == 2,
               saved_world=rd["saved_world"], world=rd["world"])
    # Loss spike on the new branch: no NaN, but the loss jumps x10. The guard must hold the job.
    spike_at = last_step_before("9999") + 120
    t3 = mark("loss_spike", spike_at_step=spike_at, cause_class="software")
    drill_file("spike_at_step", spike_at)
    held2 = wait(lambda: ev_after(t3, "job.held"), 180, "job held on loss spike")
    lt2 = latest()
    ex = ev_after(t3, "train.exit")
    record("loss_spike_hold", held2["cause"] == "loss_spike" and ex and ex.get("cause") == "loss_spike"
           and lt2["step"] <= spike_at, spike_at_step=spike_at, latest_committed=lt2["step"])
    drill_file("spike_at_step")
    ctl(f"scancel {jid2}")
    t4 = mark("rollback", from_step=lt2["step"], cause_class="operator")
    env = dict(MAIN_ENV, STEPS=str(lt2["step"] + 60), TRAIN_EXTRA=f"--from-step {lt2['step']} --skip 5 --branch b2")
    jid3, err = sbatch("/opt/v5/submission/slurm/mainrun.sbatch", env, MAIN_ARGS if DS else f"{MAIN_ARGS} --nodes=1")
    done = wait(lambda: ev_after(t4, "train.exit", code=0), 240, "main run finished")
    record("main_run_completes", done["committed"], final_step=done["step"], job=jid3)


def d_crash_loop():
    t = mark("crash_loop", cause_class="software")
    jid, err = sbatch("/opt/v5/submission/slurm/team_job.sbatch",
                      dict(RUN_ID="crashloop", STEPS="50", INTERVAL_S="999", CRASH_AT="3",
                           **({"V5_DEEPSPEED": "1"} if DS else {})),
                      "--account=arch -p cpu --ntasks-per-node=1 --gpus-per-node=0")
    if jid is None:
        raise RuntimeError(err)
    held = wait(lambda: ev_after(t, "job.held", run="crashloop"), 480, "crash loop hold")
    record("crash_loop_hold", "crash_loop" in held["cause"], job=jid, cause=held["cause"],
           restarts=held["restart_count"])


def d_budget_fairshare():
    ctl("sacctmgr -i modify account quant set GrpTRESMins=cpu=2")
    j1, _ = sbatch("--wrap 'sleep 45'", args="--account=quant -p cpu -c 4")
    wait(lambda: job(j1).get("JobState") == "COMPLETED", 120, "budget job done")
    j2, _ = sbatch("--wrap 'hostname'", args="--account=quant -p cpu")
    reason = wait(lambda: (lambda r: r if "Min" in r else None)(job(j2).get("Reason", "")), 120, "budget block")
    ctl(f"scancel {j2}")
    record("budget_limit", "Min" in reason, reason=reason)
    # LevelFS: below 1 = used more than its share. The heavy account must rank below the unused one.
    out = ctl("sshare -n -P -A arch,posttrain -o account,user,rawusage,levelfs").stdout
    rows = {r.split("|")[0].strip(): r.split("|") for r in out.splitlines() if r.strip() and not r.split("|")[1]}
    arch, post = rows.get("arch"), rows.get("posttrain")
    lv = lambda r: float(r[3]) if r and r[3] else None  # noqa: E731
    ok = arch and post and lv(arch) is not None and lv(post) is not None and lv(arch) < lv(post)
    record("fair_share", bool(ok), arch=arch, posttrain=post)
    # GPU-minutes per account from the job records (sreport's hourly roll-up is empty in a short test).
    out = ctl("sacct -a -X -D -n -P -S now-1days -o account,elapsedraw,alloctres").stdout   # -D: every requeue segment
    used = {}
    for line in out.splitlines():
        parts = line.split("|")
        m = re.search(r"gres/gpu=(\d+)", parts[-1]) if len(parts) == 3 else None
        if m and parts[1]:
            used[parts[0]] = round(used.get(parts[0], 0) + int(m.group(1)) * int(parts[1]) / 60, 1)
    record("gpu_hours_accounting", used.get("mainrun", 0) > 0, gpu_minutes_by_account=used)


def d_retention_export():
    jid, _ = sbatch(f"--wrap '. /etc/v5/node.env; {PY} /opt/v5/submission/ckpt/ckpt.py retention --run {REMOTE}/{RUN} "
                    f"--keep 3 --protect-every 5'", args="--account=infra -p cpu", user="root")
    ret = wait(lambda: ev_after("0", "checkpoint.retention", run=None), 120, "retention")
    exp = ("import sys; sys.path[:0] = ['/opt/v5/submission', '/opt/v5/support/simulation']; from ckpt.ckpt import export_safetensors; "
           "from trainer.tiny import TinyLM, make_probe_fn; "
           f"print(export_safetensors('{REMOTE}/{RUN}', lambda: TinyLM(1234), make_probe_fn(1234), "
           f"'/s3/exports/{RUN}.safetensors'))")
    ctl("mkdir -p /s3/exports && chmod 1777 /s3/exports")
    ctl(f"cat > /s3/exports/export_job.py <<'EOF'\n{exp}\nEOF")
    sbatch(f"--wrap '. /etc/v5/node.env; OMP_NUM_THREADS=1 RUN_ID={RUN} {PY} /s3/exports/export_job.py'",
           args="--account=eval -p eval")
    ex = wait(lambda: ev_after("0", "export.done") or ev_after("0", "export.failed"), 120, "export")
    record("retention", not any(p in ret["deleted"] for p in ret["protected"]) and ret.get("store") == "s3",
           deleted=len(ret["deleted"]), protected=len(ret["protected"]), freed_bytes=ret["freed_bytes"],
           store=ret.get("store"))
    record("export_safetensors", ex["event"] == "export.done", step=ex.get("step"))


def d_live_alerts():
    """metrics/watch.py ran on c4 during the whole simulation: the drills must have paged (alert.fired)."""
    fired = [e for e in events() if e["event"] == "alert.fired"]
    high = sorted({e["id"] for e in fired if e["severity"] == "high"})
    want = {"H-21", "Q-02", "CK-07"}         # spot notice, held main run, NaN guard
    record("live_alerts", want <= set(high), high_alerts_fired=high, total_fired=len(fired))


# --------------------------------------------------------------------------- main

def up():
    log("starting the Docker Slurm cluster (fresh volumes)")
    dc("down", "-v", check=False)
    dc("up", "-d", "--build")
    wait(lambda: "4" in ctl("sinfo -h -t idle -o %D -p debug").stdout, 180, "4 idle nodes")
    ctl("bash /opt/v5/submission/slurm/setup_accounts.sh >/dev/null", check=True)
    ctl("mkdir -p /s3/logs /s3/events /s3/runs /s3/drills /s3/backups && chmod 1777 /s3/*", check=True)
    # Live alerting, as Team 08's pager would run it: every 15 s, new alerts become alert.fired events.
    dc("exec", "-d", "c4", "bash", "-c",
       f"cd /opt/v5/submission && V5_EVENTS_DIR=/s3/events RUN_ID={RUN} {PY} metrics/watch.py --events /s3/events "
       f"--run {RUN} --nodes 2 --interval-s {MAIN_ENV['INTERVAL_S']} --every 15 --sim > /s3/logs/watch.log 2>&1")


def write_summary(summary):
    """docs/sim_results.md: the drill table plus the metrics report, for the PR and the ADRs."""
    mode = " (DeepSpeed ZeRO-1 trainer)" if DS else ""
    L = [f"# Docker Slurm simulation results{mode}", "",
         f"Generated by `support/simulation/run_sim.py{' --deepspeed' if DS else ''}` on "
         f"{time.strftime('%Y-%m-%d %H:%M %Z')}. "
         f"**{summary['passed']} of {summary['total']} drills passed** in {summary['duration_s']} s. "
         f"Raw files: `support/simulation/{os.path.basename(RESULTS)}/` "
         "(drills.json, metrics.json, alerts.json, report.md).", "",
         "In simple words: we broke the test cluster on purpose in every way the ADRs plan for, and checked that "
         "the main run came back by itself from its last checkpoint. Checkpoints went to an S3 emulator through "
         "the real S3 code; GPUs were placeholders (no computation), see ADR 0002 §Validation.", "",
         "| Drill | Result | Details |", "|---|---|---|"]
    for r in summary["drills"]:
        d = {k: v for k, v in r.items() if k not in ("drill", "pass")}
        L.append(f"| {r['drill']} | {'PASS' if r['pass'] else '**FAIL**'} | `{json.dumps(d)[:300]}` |")
    report = os.path.join(RESULTS, "report.md")
    if os.path.exists(report):
        body = open(report).read().split("\n", 1)[1]
        L += ["", "## Metrics report", "", body.replace("\n## ", "\n### ")]
    with open(os.path.join(HERE, "..", "docs", "sim_results_deepspeed.md" if DS else "sim_results.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-up", action="store_true", help="leave the cluster running afterwards")
    ap.add_argument("--only", nargs="+", metavar="DRILL", help="run only these drills (debugging; no summary written)")
    ap.add_argument("--deepspeed", action="store_true",
                    help="every training job uses DeepSpeed ZeRO-1; results go to results_deepspeed/")
    a = ap.parse_args()
    global RESULTS, DS
    if a.deepspeed:
        DS = True
        RESULTS = os.path.join(HERE, "results_deepspeed")
        for e in (MAIN_ENV, TEAM_ENV):
            e.update(V5_DEEPSPEED="1", V5_ZERO_STAGE="1")
    if not shutil.which("docker"):
        sys.exit("docker is needed")
    t0 = time.time()
    up()
    plan = [("accounts", d_accounts), ("preemption", d_preemption), ("gpu_allocation", d_gpu_allocation),
            ("crash_loop", d_crash_loop)]
    plan += [(f"kill_{i}", d_kill(i)) for i in range(1, REPEATS + 1)]
    plan += [(f"sigusr1_{i}", d_usr1(i)) for i in range(1, REPEATS + 1)]
    plan += [("kill_during_save", d_kill_during_save), ("kill_during_restore", d_kill_during_restore),
             ("spot", d_spot), ("health", d_health), ("two_nodes_down", d_two_nodes_down), ("hang", d_hang),
             ("node_loss", d_node_loss), ("ctld_failover", d_ctld_failover), ("db_backup", d_db_backup),
             ("nan", d_nan_rollback_reshard), ("budget", d_budget_fairshare), ("retention", d_retention_export),
             ("live_alerts", d_live_alerts)]
    if a.only:
        plan = [(n, f) for n, f in plan if n in a.only]
    for name, fn in plan:
        log(f"--- drill: {name}")
        run_drill(name, fn)
    if a.only:
        log(f"{sum(r['pass'] for r in results)}/{len(results)} selected drills passed (no summary written)")
        return 0 if all(r["pass"] for r in results) else 1
    ctl("scancel -u svc; true")
    log("evaluating metrics")
    p = ctl(f"cd /opt/v5/submission && {PY} metrics/evaluate.py --events /s3/events --out /s3/report "
            f"--prices metrics/prices.yaml --nodes 2 --interval-s {MAIN_ENV['INTERVAL_S']} --run {RUN} --sim")
    print(p.stdout[-4000:], p.stderr[-2000:])
    os.makedirs(RESULTS, exist_ok=True)
    for f in ("metrics.json", "alerts.json", "report.md"):
        out = ctl(f"cat /s3/report/{f}").stdout
        with open(os.path.join(RESULTS, f), "w") as fh:
            fh.write(out)
    summary = {"duration_s": round(time.time() - t0), "passed": sum(r["pass"] for r in results),
               "total": len(results), "drills": results}
    with open(os.path.join(RESULTS, "drills.json"), "w") as f:
        json.dump(summary, f, indent=1)
    write_summary(summary)
    log(f"{summary['passed']}/{summary['total']} drills passed in {summary['duration_s']} s; results in {RESULTS}")
    if not a.keep_up:
        dc("down", "-v", check=False)
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
