# Team 07 briefing: what we built, what we decided, and why

For every Team 07 member, including those who did not get the chance to work on the code. You know
infrastructure; you do not need to read Python. After this page you should be able to explain any decision we made,
say why we made it, and know what is still open. The full records are the four ADRs in [adr/](adr/README.md); how
to use the system day to day is in the [user guide](USER_GUIDE.md).

## 1. The problem in one paragraph

The V5 main run trains one large model on many GPUs for weeks. GPU hours are the most expensive thing in the
course. At this scale something breaks every few hours: a GPU, a node, the network, a software bug, a bad batch of
data (the reference run in Session 15 had one interruption every 12.9 hours). In the previous course run a
checkpoint once loaded "fine" while part of the model was silently random, restarts were typed by hand, and an
idle cluster kept costing money for days. Our job: **make sure money
already spent on training is never lost, failures fix themselves, broken hardware is taken out early, and when a
person is needed the system says what to do.**

## 2. The answer in one page

| Area | What we decided | Why, in one line |
|---|---|---|
| **Save often, prove every save** (ADR 0001) | Save the full training state every 30 min and on every warning; a save only counts once every file is stored, re-checked and a pointer is moved **last**; every restore is **proven** equal to what was saved | a half-written or wrongly loaded checkpoint is worse than none, because it fails silently |
| **Restart by itself** (ADR 0002) | Run on **Slurm**; three exit codes: 0 done, **75 = "saved, restart me"**, **78 = "stop, a person must look"** | most failures need no human; the few that do must not be retried blindly |
| **Share the GPUs fairly** (ADR 0002) | The main run first; every team gets a GPU share, a budget and per-job limits | the main run must never wait, and no team can take everything |
| **Take out broken hardware early** (ADR 0003) | Check every node before each job and every 30 s; a failing node is drained and its jobs save and move | one bad GPU slows or crashes the whole run |
| **Say what to do** (ADR 0004) | 74 measurements with LOW / MEDIUM / HIGH thresholds; every code and alert explains itself in plain words | a 3 a.m. page must tell the on-call person the next step |
| **Start in the right order** | One script, `ops/launch.sh`, brings the cluster up step by step and stops at the first problem | a wrong start order is a common cause of a broken first day |
| **No dependence on a provider** | Runs on bare metal or any cloud; storage is a shared file system or any S3-compatible store | the course may run on bare metal; nothing should assume one vendor |

**Evidence.** We cannot test on the real GPU cluster yet, so we built a small copy of it in Docker (a real Slurm
with fake GPUs) and broke it on purpose in 44 ways: killed processes, pulled nodes, faked GPU faults, stopped the
scheduler, injected bad numbers. **All 44 drills pass** with the checkpoints on an S3-compatible store; the same run
with a shared folder (the bare-metal case) is in progress and its result will be added here. Each drill keeps a log of every command it ran, so anyone can check the claim.

## 3. The decisions in detail

Each decision below: **what** we chose, **why**, what we **rejected** and why, and what it **means for you**.

### 3.1 Checkpoints: save, commit, prove (ADR 0001)

**What.**
- A checkpoint holds *everything* needed to continue exactly: the model weights, the optimizer state, the
  learning-rate schedule, every random-number generator on every GPU, and the exact position in the data.
- Saving has two parts. Training pauses only for a quick copy to the node's memory; writing to disk and copying to
  the shared store run in the background while training continues.
- **A checkpoint counts only when it is committed:** every file is stored, every file's checksum is re-checked, a
  receipt listing all files is written **last**, and only then is the pointer "latest" moved. If anything dies
  halfway, the pointer still shows the previous good checkpoint.
- **Every restore is proven** with four checks: a fingerprint of the whole state, the loss on one fixed test batch
  compared bit for bit, the next data batch, and the frozen parts of the model. Any mismatch falls back to the
  previous checkpoint by itself.
- **Only one copy of a run may write** ("fencing"). If a node loses its network, Slurm may start a second copy
  while the first still runs. Each copy takes a ticket number when it starts; the store accepts commits only from
  the newest ticket, so the old copy can never overwrite the new one's work.
- Extra guards: a save is refused if a part that must never change (the KE2 tables) has changed; a restore is
  refused if the software, the configuration, the embedding policy or the data list is different.

**Why.** In the previous run, a converted checkpoint loaded without error while some weights were wrong; only the
loss after one step showed it. A checkpoint you cannot trust costs more than a lost hour. The commit rule and the
four proofs turn silent failures into loud ones.

**Rejected.**
- *Picking "latest" by listing folders:* a half-uploaded folder can look newest.
- *Checking a resume by eye ("loss looks close"):* small errors pass unseen.
- *Saving only the weights:* the resumed run would be a different run.

**For you.** A held job that says "restore check failed" or "environment changed" is the system protecting the run,
not a bug. The [user guide §5.2](USER_GUIDE.md#52-a-job-is-held-exit-78) says what to do for each case.

### 3.2 The job queue: Slurm and three exit codes (ADR 0002)

**What.**
- **Slurm** schedules every job: the main run and every team's jobs.
- Every training job ends with one of three codes: **0** finished; **75** "I saved, run me again" (Slurm puts it
  back in the queue and it continues from the last checkpoint); **78** "stop, a person must decide" (Slurm holds it).
- A job that crashes once is restarted; **three crashes in 30 minutes** means something is really wrong, so it is
  held instead of burning GPU hours in a loop.
- Every held job writes a plain-words comment in Slurm: what happened, what was already done, what to do.

**Why.** Most failures (a killed process, a lost node, a preemption) only need "restart from the last save". A few
(bad numbers in training, a changed environment) get worse with every retry. Three codes keep that difference
simple for everyone: the scheduler, the scripts and the on-call person.

**Rejected.**
- *Kubernetes or a managed batch service:* weaker for many GPUs working as one job; harder to copy in a test.
- *A provider's managed Slurm:* it ties us to one provider, and the course may run on bare metal.
- *Treating every crash as fatal:* a 3 a.m. network blip would stop the run until morning.

**For you.** Jobs that stop with 75 need nothing from you. Jobs held with 78 always say why in `squeue` /
`scontrol show job`.

### 3.3 Sharing the GPUs fairly (ADR 0002)

**What.**
- The **main run** has its own account and top priority. If it needs a node, team jobs on it get a signal, have
  60 s to save, and go back to the queue.
- Every **team** has its own account with a **GPU share** (how many GPUs it may hold at once), a **budget**
  (GPU-hours; a job starts only if it can finish inside what is left), and **per-job limits** (at most 24 h per
  job, a cap on queued jobs). A job that breaks a per-job limit is refused at once with the reason, instead of
  waiting forever.
- A **debug** queue gives one quick 1-GPU job ahead of everyone, for 30 minutes.
- Long team jobs get a warning 5 minutes before their time limit, save, and continue in the next segment
  automatically.

**Why.** The main run is the course's product and must never wait behind a test job. Teams still need fair, predictable
access, and a budget that cannot be overspent by one last job.

**Rejected.**
- *No preemption, separate nodes per team:* fine with many nodes, but on a shared node the main run would wait.
- *Node caps only:* meaningless on one 8-GPU node; GPU shares work on any size.

**For you.** If your job waits, `squeue` shows the reason (for example "your team's GPU share is used up"); the
[user guide §2.3](USER_GUIDE.md#23-when-your-job-does-not-start) lists every reason and the fix.

### 3.4 Node health (ADR 0003)

**What.**
- **Before every job** a quick check (≤ 30 s); **every 30 seconds** a light check while jobs run; a **deep check**
  before a dry run or after a repair.
- What is checked: GPU count, temperature, memory errors (ECC), failed memory repairs, GPU-to-GPU links (NVLink),
  GPU driver error codes (Xid), local disk space, clock, and four kinds of GPU-to-GPU data exchange (the ones
  training actually uses) for correct results and speed.
- **On a failure** the node is drained (no new jobs), its jobs save and move to a healthy node, and the reason is
  written next to the node.
- A missing tool is reported as "skipped", **never as "pass"**.

**Why.** In published reports of large training runs (Session 15), faulty GPUs and GPU memory cause about half of
all interruptions, and one slow GPU slows every GPU in the job. Finding it before a job, or within 30 s, costs seconds instead of a crash.

**Rejected.**
- *The vendor's deep diagnostic every 30 s:* it loads the GPUs and can falsely fail a busy node, draining the main
  run's nodes again and again.
- *Treating every GPU error code as fatal:* some only mean "restart the job"; one of them now drains a node only if
  it repeats within 24 h.

**For you.** A drained node shows its reason in `sinfo -R`; the [user guide §5.3](USER_GUIDE.md#53-a-node-is-drained)
says what to do for each.

### 3.5 Monitoring, cost and alerts (ADR 0004)

**What.**
- 74 measurements (catalogue in `metrics/catalog.yaml`), each with LOW / MEDIUM / HIGH thresholds and a first action.
- **HIGH** pages a person now; **MEDIUM** is for today; **LOW** for the weekly review.
- Every exit code, error and alert can be explained in plain words: `python metrics/explain.py <anything>`.
- Cost: GPU-hours per team, the cost of a lost hour, and a check for paid machines that Slurm does not know
  about (the "forgotten test box").
- Prices in the repo are **assumptions** until checked against the real price list.

**Why.** An alert that only says "CK-05" at 3 a.m. wastes the first 20 minutes of an incident. Every alert names the
next step.

**For you.** If you are on call, the first thing to run is `python queue/job_queue.py status`.

### 3.6 Storage: bare metal or any cloud

**What.** Checkpoints and logs go to "the shared store", one setting (`V5_REMOTE_ROOT`):
- a **folder on a shared file system** every node mounts (NFS, Lustre, GPFS, CephFS): the default, and the
  bare-metal case;
- or an **S3-compatible object store** (self-hosted, or a cloud's): optional.

The node's local disk is only a fast cache; the shared store holds the copy that counts. The launch script checks
the store before any GPU hour is spent.

**Why.** We do not know yet whether the course runs on bare metal or in a cloud. Tying the design to one storage
product or provider would force a rewrite later.

**Evidence.** All 44 drills pass with the S3-compatible store; the shared-folder run is in progress.

### 3.7 Starting the cluster (operations)

**What.** `ops/launch.sh` starts everything in order and checks each step before the next: logins between machines
(munge), the accounting database, a backup of it **that is test-restored**, the schedulers, the accounts, the
GPU nodes, a deep health check of every node, the background services, a check of the shared store, and finally
the main run. It stops at the first failure and says how to continue.

**Why.** Most first-day outages are order problems (a scheduler started before its database, a node with a different
login key). A script that checks each step makes the first day boring.

### 3.8 What we merged from team members

This submission combines three team members' repos (details in [support/docs/team07_merge_plan.md](support/docs/team07_merge_plan.md)):

| From | What we took |
|---|---|
| pradeep-kumar-gs | the checkpoint design, the exit-code contract, health engine, monitoring, launch script, Docker simulation |
| millermuttu | the four GPU data-exchange checks, the NVLink check, fake GPU tools for testing, frozen-part and data-list checks, the 90-second quick test |
| uz9ai | per-job limits and the debug queue, time-limit self-requeue, the evidence log per test, the reasons to launch without a launcher, budgets as a hard stop |

## 4. What is still open

| Open item | Why it is open | Who / when |
|---|---|---|
| Real GPU numbers | no GPU cluster yet; a 2-GPU check package is ready to run on Kaggle | Team 07, now; the shared GPU node from 1 Nov |
| The real shared store | tested on a folder and on an S3-compatible emulator only | the launch script checks it on day one |
| GPU share per team, time limits | set by TSAI | TSAI, before the main run |
| Whether the main run shares its node with team jobs | decides whether preemption is needed | team decision |
| An installer that builds a node from one config file | uz9ai's installer, merge phase B | uz9ai + Team 07 |
| Model size and precision | decide checkpoint size and save time | Team 05, config v1 |

## 5. How you can help now

- **Read and question the ADRs**: every "In simple words" box should make sense to you; if one does not, that is a
  bug in the document.
- **Run the quick test** on a laptop with Docker: `python3 support/simulation/run_sim.py --smoke` (about 5 minutes).
- **Own an open item** from §4, for example the Kaggle GPU check (`support/kaggle/README.md`) or the on-call rota.
- **Rehearse an incident**: plant a fault in the Docker cluster (user guide §7.2) and follow the on-call steps.

## 6. Words used here

| Word | Meaning |
|---|---|
| checkpoint | a saved copy of everything training needs to continue |
| commit | the moment a checkpoint counts: all files stored and checked, receipt written last, pointer moved |
| fencing | making sure only the newest copy of a run can save |
| Slurm | the scheduler: decides which job runs on which node |
| requeue / hold | Slurm puts a job back in the queue / keeps it until a person releases it |
| drain | no new jobs on a node; its running jobs save and move |
| QOS, account | Slurm's names for a set of limits and priority, and for who pays |
| preemption | the main run takes a node from a lower-priority job, which saves first |
| Xid, ECC, NVLink | a GPU driver error code; GPU memory error correction; the fast link between GPUs |
| drill | a failure we cause on purpose in the test cluster to check the system recovers |
| shared store | the place every node can reach where checkpoints and logs are kept |
