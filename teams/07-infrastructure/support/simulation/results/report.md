# Simulation metrics report: run `v5main`

Cluster price used: 2 × p6-b200.48xlarge = **$227.86/h** (assumed price, see metrics/prices.yaml). Interval: 8 s (one interval = $0.51).

> In the simulation a fault is injected about once a minute, while a real run sees one every few hours or days. So goodput and cost-waste numbers here are far worse than production will be. The simulation proves the *mechanism* and measures each step's *duration*; `support/cost/cost_model.py --measured` projects those durations onto real failure rates.

Metrics in catalog: 74; computed from simulation events: 40.

## Alerts

| Severity | ID | Metric | Value | Caused by the test? | Action |
|---|---|---|---|---|---|
| HIGH | CK-07 | NaN guard trips | 1 count | yes | Rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | Q-02 | Main run held (exit 78 / crash loop) | 2 count | yes | Read the job.held cause; rollback or fix, then release. |
| HIGH | Q-05 | Main-run interruptions per day | 859.7126 count/day | fault rate | Group incidents by node and cause_class; drain repeat offenders. |
| HIGH | Q-19 | Goodput of the main run | 22.5836 % | fault rate | See the largest non-productive bucket in the report. |
| HIGH | H-14 | Step watchdog fired (hang) | 2.1071 count/h | fault rate | Read the stack dump in the job log and the NCCL flight-recorder trace; drain the rank that stopped. |
| HIGH | H-15 | Straggler ratio (rank step time / median) | 1.321 ratio | no | Medium for 3 windows: drain at the next checkpoint. |
| HIGH | H-21 | Spot interruption notice | 1 count | yes | Automatic: drain + save + requeue. Page only if no commit within 90 s. |
| HIGH | H-24 | Loss-spike guard trips | 1 count | yes | Look at the batches before the spike; rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | C-02 | Cost per productive GPU-hour vs list price | 4.428 ratio | fault rate | Fix the largest waste bucket (C-03). |
| HIGH | C-03 | Wasted share of main-run spend | 77.4164 % | fault rate | The report lists each bucket in USD. |
| HIGH | C-04 | Cost of one incident | 21.4125 x interval cost | yes | Large values: missing emergency save, slow restore, or no spare. |
| MEDIUM | Q-04 | Crash-loop holds | 1 count | yes | Read the logs of the 3 crashes; for the main run this is also Q-02 (high). |
| MEDIUM | Q-09 | Drained nodes | 2 count | yes | Re-test with --full; repair, replace or stop the instance. |
| MEDIUM | Q-21 | Human interventions | 4 count/week | yes | Automate the step that needed a person. |
| LOW | CK-10 | Save backpressure (previous save still running) | 1 count/day | no | Increase the interval or upload bandwidth. |
| LOW | CK-14 | Work lost per incident | 0.9163 x interval | no | Check that SIGUSR1/SIGTERM reached the trainer and the emergency save committed. |
| LOW | CK-19 | Resharded resumes | 1 count | yes | Note it in the run log and the paper's failure table. |
| LOW | Q-16 | Jobs rejected for no valid account | 1 count | yes | Check the submit system's account mapping. |

## Where the main run's wall-clock time (and money) went

| Bucket | Seconds | Share | USD |
|---|---|---|---|
| checkpoint_stall | 1.37 | 0.1% | $0.09 |
| lost_work | 16.02 | 1.0% | $1.01 |
| detection | 142.81 | 8.7% | $9.04 |
| reschedule | 995.79 | 60.5% | $63.03 |
| restore | 117.52 | 7.1% | $7.44 |
| productive | 371.5 | 22.6% | $23.51 |
| wall_clock | 1645.01 | 100.0% | $104.12 |

## Recovery time by fault type (MTTR, seconds)

| Fault | n | p50 | p95 | max |
|---|---|---|---|---|
| hang | 1 | 86.86 | 86.86 | 86.86 |
| health_fail | 1 | 92.45 | 92.45 | 92.45 |
| kill | 3 | 81.42 | 89.21 | 89.21 |
| kill_during_restore | 1 | 170.06 | 170.06 | 170.06 |
| kill_during_save | 1 | 79.52 | 79.52 | 79.52 |
| node_loss | 1 | 118.44 | 118.44 | 118.44 |
| rollback | 2 | 3.12 | 3.19 | 3.19 |
| sigusr1 | 3 | 88.49 | 88.93 | 88.93 |
| spot_notice | 1 | 88.91 | 88.91 | 88.91 |
| two_nodes_down | 1 | 85.38 | 85.38 | 85.38 |

## Incidents

| Fault | Cause class | Detect s | Reschedule s | Restore s | MTTR s | Lost steps | Restored step | Human |
|---|---|---|---|---|---|---|---|---|
| kill | software | 0.23 | 86.9 | 2.08 | 89.21 | 124 | 2926 | no |
| kill | software | 0.25 | 79.01 | 2.16 | 81.42 | 3 | 3057 | no |
| kill | software | 0.24 | 77.34 | 1.97 | 79.55 | 31 | 3189 | no |
| sigusr1 | preemption | 0.44 | 86.48 | 2.01 | 88.93 | 0 | 3213 | no |
| sigusr1 | preemption | 0.4 | 85.18 | 1.5 | 87.08 | 0 | 3254 | no |
| sigusr1 | preemption | 0.48 | 85.82 | 2.18 | 88.49 | 0 | 3297 | no |
| kill_during_save | software | 7.84 | 69.75 | 1.92 | 79.52 | 32 | 3428 | no |
| kill_during_restore | software | 0.25 | 78.42 | 91.39 | 170.06 | 21 | 3559 | no |
| spot_notice | host | 2.65 | 84.7 | 1.55 | 88.91 | 0 | 3620 | no |
| health_fail | gpu | 26.87 | 63.1 | 2.48 | 92.45 | 0 | 4084 | no |
| two_nodes_down | gpu | 22.13 | 60.99 | 2.26 | 85.38 | 0 | 4480 | no |
| hang | software | 21.34 | 63.86 | 1.66 | 86.86 | 20 | 4480 | no |
| node_loss | host | 42.6 | 74.24 | 1.6 | 118.44 | 20 | 4480 | no |
| nan | software | 0.91 | None | None | None | None | None | yes |
| rollback | operator | 1.78 | 0.0 | 1.41 | 3.19 | 4 | 6866 | yes |
| loss_spike | software | 7.42 | None | None | None | None | None | yes |
| rollback | operator | 6.98 | 0.0 | 1.35 | 3.06 | 16 | 6994 | yes |

## Checkpoint

- committed: 58, failed: 0, restores: 15
- stall per save: mean 0.0235 s, p95 0.0267 s; commit latency mean 0.1116 s; size 1152012 bytes (tiny model)

## All computed metrics

| ID | Metric | Value | Unit | Severity |
|---|---|---|---|---|
| CK-01 | Time since last committed checkpoint | 1.1386 | x interval | OK |
| CK-02 | Compute dollars at risk (since last commit) | 0.5765 | USD | OK |
| CK-03 | Consecutive failed saves | 0 | count | OK |
| CK-04 | Checksum mismatches (commit or restore) | 0 | count | OK |
| CK-05 | Restore check failures (digest, probe, next batch, frozen) | 0 | count | OK |
| CK-06 | Strict-load or environment refusals | 0 | count | OK |
| CK-07 | NaN guard trips | 1 | count | HIGH |
| CK-08 | Save stall as share of interval | 0.2943 | % | OK |
| CK-09 | Commit latency (stall end to committed) | 0.2115 | s | OK |
| CK-10 | Save backpressure (previous save still running) | 1 | count/day | LOW |
| CK-13 | Restore time (job start to first productive step) | 91.39 | s | OK |
| CK-14 | Work lost per incident | 0.9163 | x interval | LOW |
| CK-15 | Protected checkpoint missing or deleted | 0 | count | OK |
| CK-16 | Orphan uncommitted objects in S3 | 0.0 | GB | OK |
| CK-18 | Export parity failures (safetensors) | 0 | count | OK |
| CK-19 | Resharded resumes | 1 | count | LOW |
| CK-20 | Probe value missing for this device class | 0 | count | OK |
| Q-01 | Main run not running (pending time) | 86.9 | s | OK |
| Q-02 | Main run held (exit 78 / crash loop) | 2 | count | HIGH |
| Q-03 | Reschedule latency (job end to running again) | 86.9 | s | OK |
| Q-04 | Crash-loop holds | 1 | count | MEDIUM |
| Q-05 | Main-run interruptions per day | 859.7126 | count/day | HIGH |
| Q-07 | Allocated but not stepping | 91.395 | s | OK |
| Q-09 | Drained nodes | 2 | count | MEDIUM |
| Q-10 | Node replacement time (drain to spare running the job) | 86.507 | s | OK |
| Q-14 | Team GPU-hours lost to preemption | 0.0 | node-h/week | OK |
| Q-16 | Jobs rejected for no valid account | 1 | count | LOW |
| Q-19 | Goodput of the main run | 22.5836 | % | HIGH |
| Q-20 | MTTR (fault to first productive step) | 82.8367 | s | OK |
| Q-21 | Human interventions | 4 | count/week | MEDIUM |
| H-01 | Fatal Xid on a node | 0 | count | OK |
| H-14 | Step watchdog fired (hang) | 2.1071 | count/h | HIGH |
| H-15 | Straggler ratio (rank step time / median) | 1.321 | ratio | HIGH |
| H-17 | Health check runtime (prolog) | 0.003 | s | OK |
| H-18 | Required check skipped | 0 | count | OK |
| H-21 | Spot interruption notice | 1 | count | HIGH |
| H-24 | Loss-spike guard trips | 1 | count | HIGH |
| C-02 | Cost per productive GPU-hour vs list price | 4.428 | ratio | HIGH |
| C-03 | Wasted share of main-run spend | 77.4164 | % | HIGH |
| C-04 | Cost of one incident | 21.4125 | x interval cost | HIGH |
