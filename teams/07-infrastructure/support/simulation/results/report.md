# Simulation metrics report: run `v5main`

Cluster price used: 2 × p6-b200.48xlarge = **$227.86/h** (assumed price, see metrics/prices.yaml). Interval: 8 s (one interval = $0.51).

> In the simulation a fault is injected about once a minute, while a real run sees one every few hours or days. So goodput and cost-waste numbers here are far worse than production will be. The simulation proves the *mechanism* and measures each step's *duration*; `support/cost/cost_model.py --measured` projects those durations onto real failure rates.

Metrics in catalog: 74; computed from simulation events: 40.

## Alerts

| Severity | ID | Metric | Value | Caused by the test? | Action |
|---|---|---|---|---|---|
| HIGH | CK-07 | NaN guard trips | 1 count | yes | Rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | Q-02 | Main run held (exit 78 / crash loop) | 3 count | yes | Read the job.held cause; rollback or fix, then release. |
| HIGH | Q-05 | Main-run interruptions per day | 936.5984 count/day | fault rate | Group incidents by node and cause_class; drain repeat offenders. |
| HIGH | Q-19 | Goodput of the main run | 11.9911 % | fault rate | See the largest non-productive bucket in the report. |
| HIGH | H-01 | Fatal Xid on a node | 1 count | no | Automatic drain + save; open a hardware ticket; replace the instance. |
| HIGH | H-21 | Spot interruption notice | 1 count | yes | Automatic: drain + save + requeue. Page only if no commit within 90 s. |
| HIGH | H-24 | Loss-spike guard trips | 1 count | yes | Look at the batches before the spike; rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | C-02 | Cost per productive GPU-hour vs list price | 8.3395 ratio | fault rate | Fix the largest waste bucket (C-03). |
| HIGH | C-03 | Wasted share of main-run spend | 88.0089 % | fault rate | The report lists each bucket in USD. |
| HIGH | C-04 | Cost of one incident | 21.39 x interval cost | yes | Large values: missing emergency save, slow restore, or no spare. |
| MEDIUM | CK-03 | Consecutive failed saves | 1 count | no | Read the failure cause (nvme_full, IntegrityError, upload timeout) and fix it now. |
| MEDIUM | Q-04 | Crash-loop holds | 1 count | yes | Read the logs of the 3 crashes; for the main run this is also Q-02 (high). |
| MEDIUM | Q-09 | Drained nodes | 2 count | yes | Re-test with --full; repair, replace or stop the instance. |
| MEDIUM | Q-10 | Node replacement time (drain to spare running the job) | 423.229 s | yes | Keep the hot spare up; check preemption grace time on the spare's team job. |
| MEDIUM | Q-21 | Human interventions | 4 count/week | yes | Automate the step that needed a person. |
| MEDIUM | H-14 | Step watchdog fired (hang) | 1.8583 count/h | fault rate | Read the stack dump in the job log and the NCCL flight-recorder trace; drain the rank that stopped. |
| LOW | CK-14 | Work lost per incident | 0.935 x interval | no | Check that SIGUSR1/SIGTERM reached the trainer and the emergency save committed. |
| LOW | CK-19 | Resharded resumes | 1 count | yes | Note it in the run log and the paper's failure table. |
| LOW | Q-16 | Jobs rejected for no valid account | 1 count | yes | Check the submit system's account mapping. |
| LOW | H-15 | Straggler ratio (rank step time / median) | 1.0821 ratio | no | Medium for 3 windows: drain at the next checkpoint. |

## Where the main run's wall-clock time (and money) went

| Bucket | Seconds | Share | USD |
|---|---|---|---|
| checkpoint_stall | 1.33 | 0.1% | $0.08 |
| lost_work | 25.12 | 1.5% | $1.59 |
| detection | 300.98 | 17.9% | $19.05 |
| reschedule | 1030.49 | 61.3% | $65.22 |
| restore | 122.55 | 7.3% | $7.76 |
| productive | 201.71 | 12.0% | $12.77 |
| wall_clock | 1682.18 | 100.0% | $106.47 |

## Recovery time by fault type (MTTR, seconds)

| Fault | n | p50 | p95 | max |
|---|---|---|---|---|
| hang | 1 | 88.69 | 88.69 | 88.69 |
| health_fail | 1 | 89.79 | 89.79 | 89.79 |
| kill | 3 | 82.15 | 88.48 | 88.48 |
| kill_during_restore | 1 | 170.64 | 170.64 | 170.64 |
| kill_during_save | 1 | 80.83 | 80.83 | 80.83 |
| node_loss | 1 | 118.29 | 118.29 | 118.29 |
| resume_after_fence | 1 | 2.7 | 2.7 | 2.7 |
| rollback | 2 | 2.99 | 3.04 | 3.04 |
| sigusr1 | 3 | 89.27 | 89.6 | 89.6 |
| spot_notice | 1 | 87.8 | 87.8 | 87.8 |
| two_nodes_down | 1 | 113.74 | 113.74 | 113.74 |

## Incidents

| Fault | Cause class | Detect s | Reschedule s | Restore s | MTTR s | Lost steps | Restored step | Human |
|---|---|---|---|---|---|---|---|---|
| nvlink_down | gpu | None | None | None | None | None | None | no |
| time_limit | planned | 156.2 | 0.0 | None | None | None | None | no |
| kill | software | 0.33 | 85.57 | 2.58 | 88.48 | 98 | 2742 | no |
| kill | software | 0.27 | 76.85 | 1.82 | 78.94 | 27 | 2873 | no |
| kill | software | 0.26 | 79.62 | 2.27 | 82.15 | 6 | 3004 | no |
| sigusr1 | preemption | 0.38 | 86.56 | 2.34 | 89.27 | 0 | 3018 | no |
| sigusr1 | preemption | 0.39 | 86.72 | 2.49 | 89.6 | 0 | 3033 | no |
| sigusr1 | preemption | 0.49 | 84.86 | 1.99 | 87.35 | 0 | 3070 | no |
| kill_during_save | software | 7.87 | 70.49 | 2.47 | 80.83 | 29 | 3201 | no |
| kill_during_restore | software | 0.25 | 78.35 | 92.04 | 170.64 | 8 | 3332 | no |
| spot_notice | host | 0.96 | 84.68 | 2.16 | 87.8 | 0 | 3385 | no |
| health_fail | gpu | 24.72 | 63.17 | 1.9 | 89.79 | 0 | 3829 | no |
| two_nodes_down | gpu | 21.9 | 90.16 | 1.68 | 113.74 | 0 | 4233 | no |
| hang | software | 20.81 | 66.0 | 1.88 | 88.69 | 17 | 4233 | no |
| node_loss | host | 38.85 | 77.46 | 1.98 | 118.29 | 27 | 4233 | no |
| split_brain | software | 9.67 | None | None | None | None | None | no |
| resume_after_fence | operator | 0.88 | 0.0 | 1.82 | 2.7 | 126 | 6594 | no |
| nan | software | 1.13 | None | None | None | None | None | yes |
| rollback | operator | 1.33 | 0.0 | 1.61 | 2.94 | 66 | 6594 | yes |
| loss_spike | software | 7.35 | None | None | None | None | None | yes |
| rollback | operator | 6.94 | 0.0 | 1.52 | 3.04 | 19 | 6721 | yes |

## Checkpoint

- committed: 57, failed: 1, restores: 16
- stall per save: mean 0.0234 s, p95 0.0274 s; commit latency mean 0.112 s; size 1152156 bytes (tiny model)

## All computed metrics

| ID | Metric | Value | Unit | Severity |
|---|---|---|---|---|
| CK-01 | Time since last committed checkpoint | 1.1266 | x interval | OK |
| CK-02 | Compute dollars at risk (since last commit) | 0.5705 | USD | OK |
| CK-03 | Consecutive failed saves | 1 | count | MEDIUM |
| CK-04 | Checksum mismatches (commit or restore) | 0 | count | OK |
| CK-05 | Restore check failures (digest, probe, next batch, frozen) | 0 | count | OK |
| CK-06 | Strict-load or environment refusals | 0 | count | OK |
| CK-07 | NaN guard trips | 1 | count | HIGH |
| CK-08 | Save stall as share of interval | 0.292 | % | OK |
| CK-09 | Commit latency (stall end to committed) | 0.2246 | s | OK |
| CK-10 | Save backpressure (previous save still running) | 0 | count/day | OK |
| CK-13 | Restore time (job start to first productive step) | 92.04 | s | OK |
| CK-14 | Work lost per incident | 0.935 | x interval | LOW |
| CK-15 | Protected checkpoint missing or deleted | 0 | count | OK |
| CK-16 | Orphan uncommitted objects in S3 | 0.0 | GB | OK |
| CK-18 | Export parity failures (safetensors) | 0 | count | OK |
| CK-19 | Resharded resumes | 1 | count | LOW |
| CK-20 | Probe value missing for this device class | 0 | count | OK |
| Q-01 | Main run not running (pending time) | 90.16 | s | OK |
| Q-02 | Main run held (exit 78 / crash loop) | 3 | count | HIGH |
| Q-03 | Reschedule latency (job end to running again) | 90.16 | s | OK |
| Q-04 | Crash-loop holds | 1 | count | MEDIUM |
| Q-05 | Main-run interruptions per day | 936.5984 | count/day | HIGH |
| Q-07 | Allocated but not stepping | 92.041 | s | OK |
| Q-09 | Drained nodes | 2 | count | MEDIUM |
| Q-10 | Node replacement time (drain to spare running the job) | 423.229 | s | MEDIUM |
| Q-14 | Team GPU-hours lost to preemption | 0.0 | node-h/week | OK |
| Q-16 | Jobs rejected for no valid account | 1 | count | LOW |
| Q-19 | Goodput of the main run | 11.9911 | % | HIGH |
| Q-20 | MTTR (fault to first productive step) | 79.6406 | s | OK |
| Q-21 | Human interventions | 4 | count/week | MEDIUM |
| H-01 | Fatal Xid on a node | 1 | count | HIGH |
| H-14 | Step watchdog fired (hang) | 1.8583 | count/h | MEDIUM |
| H-15 | Straggler ratio (rank step time / median) | 1.0821 | ratio | LOW |
| H-17 | Health check runtime (prolog) | 0.19 | s | OK |
| H-18 | Required check skipped | 0 | count | OK |
| H-21 | Spot interruption notice | 1 | count | HIGH |
| H-24 | Loss-spike guard trips | 1 | count | HIGH |
| C-02 | Cost per productive GPU-hour vs list price | 8.3395 | ratio | HIGH |
| C-03 | Wasted share of main-run spend | 88.0089 | % | HIGH |
| C-04 | Cost of one incident | 21.39 | x interval cost | HIGH |
