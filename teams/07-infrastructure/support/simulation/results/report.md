# Simulation metrics report: run `v5main`

Cluster price used: 2 × p6-b200.48xlarge = **$227.86/h** (assumed price, see metrics/prices.yaml). Interval: 8 s (one interval = $0.51).

> In the simulation a fault is injected about once a minute, while a real run sees one every few hours or days. So goodput and cost-waste numbers here are far worse than production will be. The simulation proves the *mechanism* and measures each step's *duration*; `support/cost/cost_model.py --measured` projects those durations onto real failure rates.

Metrics in catalog: 74; computed from simulation events: 40.

## Alerts

| Severity | ID | Metric | Value | Caused by the test? | Action |
|---|---|---|---|---|---|
| HIGH | CK-07 | NaN guard trips | 1 count | yes | Rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | Q-02 | Main run held (exit 78 / crash loop) | 3 count | yes | Read the job.held cause; rollback or fix, then release. |
| HIGH | Q-05 | Main-run interruptions per day | 895.5766 count/day | fault rate | Group incidents by node and cause_class; drain repeat offenders. |
| HIGH | Q-19 | Goodput of the main run | 20.8728 % | fault rate | See the largest non-productive bucket in the report. |
| HIGH | H-21 | Spot interruption notice | 1 count | yes | Automatic: drain + save + requeue. Page only if no commit within 90 s. |
| HIGH | H-24 | Loss-spike guard trips | 1 count | yes | Look at the batches before the spike; rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | C-02 | Cost per productive GPU-hour vs list price | 4.7909 ratio | fault rate | Fix the largest waste bucket (C-03). |
| HIGH | C-03 | Wasted share of main-run spend | 79.1272 % | fault rate | The report lists each bucket in USD. |
| HIGH | C-04 | Cost of one incident | 21.41 x interval cost | yes | Large values: missing emergency save, slow restore, or no spare. |
| MEDIUM | CK-03 | Consecutive failed saves | 1 count | no | Read the failure cause (nvme_full, IntegrityError, upload timeout) and fix it now. |
| MEDIUM | Q-04 | Crash-loop holds | 1 count | yes | Read the logs of the 3 crashes; for the main run this is also Q-02 (high). |
| MEDIUM | Q-09 | Drained nodes | 2 count | yes | Re-test with --full; repair, replace or stop the instance. |
| MEDIUM | Q-21 | Human interventions | 4 count/week | yes | Automate the step that needed a person. |
| MEDIUM | H-14 | Step watchdog fired (hang) | 1.964 count/h | fault rate | Read the stack dump in the job log and the NCCL flight-recorder trace; drain the rank that stopped. |
| LOW | CK-01 | Time since last committed checkpoint | 1.3133 x interval | no | Check checkpoint.failed events; if saves fail, fix storage before the next failure costs a full interval. |
| LOW | CK-02 | Compute dollars at risk (since last commit) | 0.665 USD | no | Same as CK-01. Thresholds are multiples of the cost of one interval. |
| LOW | CK-14 | Work lost per incident | 0.955 x interval | no | Check that SIGUSR1/SIGTERM reached the trainer and the emergency save committed. |
| LOW | CK-19 | Resharded resumes | 1 count | yes | Note it in the run log and the paper's failure table. |
| LOW | Q-16 | Jobs rejected for no valid account | 1 count | yes | Check the submit system's account mapping. |
| LOW | H-15 | Straggler ratio (rank step time / median) | 1.0875 ratio | no | Medium for 3 windows: drain at the next checkpoint. |

## Where the main run's wall-clock time (and money) went

| Bucket | Seconds | Share | USD |
|---|---|---|---|
| checkpoint_stall | 1.27 | 0.1% | $0.08 |
| lost_work | 22.38 | 1.3% | $1.42 |
| detection | 139.1 | 7.9% | $8.80 |
| reschedule | 1025.77 | 58.0% | $64.93 |
| restore | 210.89 | 11.9% | $13.35 |
| productive | 369.15 | 20.9% | $23.37 |
| wall_clock | 1768.56 | 100.0% | $111.94 |

## Recovery time by fault type (MTTR, seconds)

| Fault | n | p50 | p95 | max |
|---|---|---|---|---|
| hang | 1 | 88.88 | 88.88 | 88.88 |
| health_fail | 1 | 89.53 | 89.53 | 89.53 |
| kill | 3 | 81.48 | 89.17 | 89.17 |
| kill_during_restore | 1 | 169.93 | 169.93 | 169.93 |
| kill_during_save | 1 | 82.03 | 82.03 | 82.03 |
| node_loss | 1 | 117.85 | 117.85 | 117.85 |
| resume_after_fence | 1 | 3.37 | 3.37 | 3.37 |
| rollback | 2 | 47.63 | 91.38 | 91.38 |
| sigusr1 | 3 | 88.11 | 89.43 | 89.43 |
| spot_notice | 1 | 87.36 | 87.36 | 87.36 |
| two_nodes_down | 1 | 113.7 | 113.7 | 113.7 |

## Incidents

| Fault | Cause class | Detect s | Reschedule s | Restore s | MTTR s | Lost steps | Restored step | Human |
|---|---|---|---|---|---|---|---|---|
| kill | software | 0.25 | 86.65 | 2.27 | 89.17 | 73 | 2907 | no |
| kill | software | 0.24 | 79.17 | 2.07 | 81.48 | 2 | 3038 | no |
| kill | software | 0.28 | 77.55 | 2.52 | 80.35 | 31 | 3169 | no |
| sigusr1 | preemption | 0.39 | 84.67 | 1.58 | 86.64 | 0 | 3211 | no |
| sigusr1 | preemption | 0.48 | 87.28 | 1.67 | 89.43 | 0 | 3226 | no |
| sigusr1 | preemption | 0.5 | 85.76 | 1.85 | 88.11 | 0 | 3264 | no |
| kill_during_save | software | 10.05 | 69.36 | 2.63 | 82.03 | 4 | 3396 | no |
| kill_during_restore | software | 0.26 | 77.32 | 92.34 | 169.93 | 23 | 3527 | no |
| spot_notice | host | 0.6 | 84.6 | 2.15 | 87.36 | 0 | 3577 | no |
| health_fail | gpu | 24.31 | 63.1 | 2.12 | 89.53 | 0 | 4021 | no |
| two_nodes_down | gpu | 21.59 | 90.07 | 2.04 | 113.7 | 0 | 4417 | no |
| hang | software | 20.95 | 65.61 | 2.33 | 88.88 | 23 | 4417 | no |
| node_loss | host | 41.11 | 74.63 | 2.11 | 117.85 | 33 | 4417 | no |
| split_brain | software | 1.76 | None | None | None | None | None | no |
| resume_after_fence | operator | 1.71 | 0.0 | 1.66 | 3.37 | 129 | 6651 | no |
| nan | software | 1.41 | None | None | None | None | None | yes |
| rollback | operator | 2.36 | 0.0 | 1.53 | 3.89 | 59 | 6651 | yes |
| loss_spike | software | 7.09 | None | None | None | None | None | yes |
| rollback | operator | 3.76 | 0.0 | 90.02 | 91.38 | 1 | 6779 | yes |

## Checkpoint

- committed: 57, failed: 1, restores: 16
- stall per save: mean 0.0223 s, p95 0.0266 s; commit latency mean 0.1075 s; size 1151948 bytes (tiny model)

## All computed metrics

| ID | Metric | Value | Unit | Severity |
|---|---|---|---|---|
| CK-01 | Time since last committed checkpoint | 1.3133 | x interval | LOW |
| CK-02 | Compute dollars at risk (since last commit) | 0.665 | USD | LOW |
| CK-03 | Consecutive failed saves | 1 | count | MEDIUM |
| CK-04 | Checksum mismatches (commit or restore) | 0 | count | OK |
| CK-05 | Restore check failures (digest, probe, next batch, frozen) | 0 | count | OK |
| CK-06 | Strict-load or environment refusals | 0 | count | OK |
| CK-07 | NaN guard trips | 1 | count | HIGH |
| CK-08 | Save stall as share of interval | 0.2789 | % | OK |
| CK-09 | Commit latency (stall end to committed) | 0.2291 | s | OK |
| CK-10 | Save backpressure (previous save still running) | 0 | count/day | OK |
| CK-13 | Restore time (job start to first productive step) | 92.34 | s | OK |
| CK-14 | Work lost per incident | 0.955 | x interval | LOW |
| CK-15 | Protected checkpoint missing or deleted | 0 | count | OK |
| CK-16 | Orphan uncommitted objects in S3 | 0.0 | GB | OK |
| CK-18 | Export parity failures (safetensors) | 0 | count | OK |
| CK-19 | Resharded resumes | 1 | count | LOW |
| CK-20 | Probe value missing for this device class | 0 | count | OK |
| Q-01 | Main run not running (pending time) | 90.07 | s | OK |
| Q-02 | Main run held (exit 78 / crash loop) | 3 | count | HIGH |
| Q-03 | Reschedule latency (job end to running again) | 90.07 | s | OK |
| Q-04 | Crash-loop holds | 1 | count | MEDIUM |
| Q-05 | Main-run interruptions per day | 895.5766 | count/day | HIGH |
| Q-07 | Allocated but not stepping | 92.337 | s | OK |
| Q-09 | Drained nodes | 2 | count | MEDIUM |
| Q-10 | Node replacement time (drain to spare running the job) | 92.41 | s | OK |
| Q-14 | Team GPU-hours lost to preemption | 0.0 | node-h/week | OK |
| Q-16 | Jobs rejected for no valid account | 1 | count | LOW |
| Q-19 | Goodput of the main run | 20.8728 | % | HIGH |
| Q-20 | MTTR (fault to first productive step) | 85.1937 | s | OK |
| Q-21 | Human interventions | 4 | count/week | MEDIUM |
| H-01 | Fatal Xid on a node | 0 | count | OK |
| H-14 | Step watchdog fired (hang) | 1.964 | count/h | MEDIUM |
| H-15 | Straggler ratio (rank step time / median) | 1.0875 | ratio | LOW |
| H-17 | Health check runtime (prolog) | 0.015 | s | OK |
| H-18 | Required check skipped | 0 | count | OK |
| H-21 | Spot interruption notice | 1 | count | HIGH |
| H-24 | Loss-spike guard trips | 1 | count | HIGH |
| C-02 | Cost per productive GPU-hour vs list price | 4.7909 | ratio | HIGH |
| C-03 | Wasted share of main-run spend | 79.1272 | % | HIGH |
| C-04 | Cost of one incident | 21.41 | x interval cost | HIGH |
