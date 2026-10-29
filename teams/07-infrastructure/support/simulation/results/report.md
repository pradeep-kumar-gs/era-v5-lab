# Simulation metrics report: run `v5main`

Cluster price used: 2 × p6-b200.48xlarge = **$227.86/h** (assumed price, see metrics/prices.yaml). Interval: 8 s (one interval = $0.51).

> In the simulation a fault is injected about once a minute, while a real run sees one every few hours or days. So goodput and cost-waste numbers here are far worse than production will be. The simulation proves the *mechanism* and measures each step's *duration*; `support/cost/cost_model.py --measured` projects those durations onto real failure rates.

Metrics in catalog: 74; computed from simulation events: 40.

## Alerts

| Severity | ID | Metric | Value | Caused by the test? | Action |
|---|---|---|---|---|---|
| HIGH | CK-07 | NaN guard trips | 1 count | yes | Rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | Q-02 | Main run held (exit 78 / crash loop) | 2 count | yes | Read the job.held cause; rollback or fix, then release. |
| HIGH | Q-05 | Main-run interruptions per day | 843.6182 count/day | fault rate | Group incidents by node and cause_class; drain repeat offenders. |
| HIGH | Q-19 | Goodput of the main run | 22.0485 % | fault rate | See the largest non-productive bucket in the report. |
| HIGH | H-14 | Step watchdog fired (hang) | 2.0677 count/h | fault rate | Read the stack dump in the job log and the NCCL flight-recorder trace; drain the rank that stopped. |
| HIGH | H-21 | Spot interruption notice | 1 count | yes | Automatic: drain + save + requeue. Page only if no commit within 90 s. |
| HIGH | H-24 | Loss-spike guard trips | 1 count | yes | Look at the batches before the spike; rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | C-02 | Cost per productive GPU-hour vs list price | 4.5355 ratio | fault rate | Fix the largest waste bucket (C-03). |
| HIGH | C-03 | Wasted share of main-run spend | 77.9515 % | fault rate | The report lists each bucket in USD. |
| HIGH | C-04 | Cost of one incident | 21.4988 x interval cost | yes | Large values: missing emergency save, slow restore, or no spare. |
| MEDIUM | Q-04 | Crash-loop holds | 1 count | yes | Read the logs of the 3 crashes; for the main run this is also Q-02 (high). |
| MEDIUM | Q-09 | Drained nodes | 2 count | yes | Re-test with --full; repair, replace or stop the instance. |
| MEDIUM | Q-21 | Human interventions | 4 count/week | yes | Automate the step that needed a person. |
| LOW | CK-01 | Time since last committed checkpoint | 1.2508 x interval | no | Check checkpoint.failed events; if saves fail, fix storage before the next failure costs a full interval. |
| LOW | CK-02 | Compute dollars at risk (since last commit) | 0.6333 USD | no | Same as CK-01. Thresholds are multiples of the cost of one interval. |
| LOW | CK-14 | Work lost per incident | 0.6875 x interval | no | Check that SIGUSR1/SIGTERM reached the trainer and the emergency save committed. |
| LOW | CK-19 | Resharded resumes | 1 count | yes | Note it in the run log and the paper's failure table. |
| LOW | Q-03 | Reschedule latency (job end to running again) | 90.27 s | yes | Check the spare is up and idle (Q-08), and slurmctld load. |
| LOW | Q-16 | Jobs rejected for no valid account | 1 count | yes | Check the submit system's account mapping. |
| LOW | H-15 | Straggler ratio (rank step time / median) | 1.0841 ratio | no | Medium for 3 windows: drain at the next checkpoint. |

## Where the main run's wall-clock time (and money) went

| Bucket | Seconds | Share | USD |
|---|---|---|---|
| checkpoint_stall | 1.25 | 0.1% | $0.08 |
| lost_work | 14.25 | 0.9% | $0.90 |
| detection | 144.82 | 8.6% | $9.17 |
| reschedule | 1028.5 | 61.4% | $65.10 |
| restore | 116.91 | 7.0% | $7.40 |
| productive | 369.33 | 22.1% | $23.38 |
| wall_clock | 1675.06 | 100.0% | $106.02 |

## Recovery time by fault type (MTTR, seconds)

| Fault | n | p50 | p95 | max |
|---|---|---|---|---|
| hang | 1 | 88.14 | 88.14 | 88.14 |
| health_fail | 1 | 91.51 | 91.51 | 91.51 |
| kill | 3 | 81.55 | 86.92 | 86.92 |
| kill_during_restore | 1 | 170.46 | 170.46 | 170.46 |
| kill_during_save | 1 | 79.49 | 79.49 | 79.49 |
| node_loss | 1 | 118.24 | 118.24 | 118.24 |
| rollback | 2 | 3.94 | 5.05 | 5.05 |
| sigusr1 | 3 | 88.52 | 89.75 | 89.75 |
| spot_notice | 1 | 89.23 | 89.23 | 89.23 |
| two_nodes_down | 1 | 116.02 | 116.02 | 116.02 |

## Incidents

| Fault | Cause class | Detect s | Reschedule s | Restore s | MTTR s | Lost steps | Restored step | Human |
|---|---|---|---|---|---|---|---|---|
| kill | software | 0.25 | 85.13 | 1.54 | 86.92 | 93 | 2907 | no |
| kill | software | 0.24 | 79.74 | 1.57 | 81.55 | 11 | 3039 | no |
| kill | software | 0.25 | 79.66 | 1.52 | 81.43 | 0 | 3170 | no |
| sigusr1 | preemption | 0.47 | 87.07 | 2.21 | 89.75 | 0 | 3194 | no |
| sigusr1 | preemption | 0.47 | 85.2 | 1.54 | 87.21 | 0 | 3229 | no |
| sigusr1 | preemption | 0.42 | 86.49 | 1.62 | 88.52 | 0 | 3262 | no |
| kill_during_save | software | 7.85 | 69.54 | 2.1 | 79.49 | 37 | 3393 | no |
| kill_during_restore | software | 0.24 | 78.35 | 91.86 | 170.46 | 26 | 3524 | no |
| spot_notice | host | 3.01 | 84.65 | 1.57 | 89.23 | 0 | 3580 | no |
| health_fail | gpu | 26.1 | 63.22 | 2.2 | 91.51 | 0 | 4025 | no |
| two_nodes_down | gpu | 23.35 | 90.27 | 2.4 | 116.02 | 0 | 4425 | no |
| hang | software | 21.2 | 65.32 | 1.61 | 88.14 | 15 | 4425 | no |
| node_loss | host | 42.11 | 73.86 | 2.27 | 118.24 | 35 | 4425 | no |
| nan | software | 1.21 | None | None | None | None | None | yes |
| rollback | operator | 3.59 | 0.0 | 1.46 | 5.05 | 6 | 6804 | yes |
| loss_spike | software | 7.29 | None | None | None | None | None | yes |
| rollback | operator | 6.77 | 0.0 | 1.44 | 2.83 | 18 | 6932 | yes |

## Checkpoint

- committed: 58, failed: 0, restores: 15
- stall per save: mean 0.0216 s, p95 0.0251 s; commit latency mean 0.1227 s; size 1151980 bytes (tiny model)

## All computed metrics

| ID | Metric | Value | Unit | Severity |
|---|---|---|---|---|
| CK-01 | Time since last committed checkpoint | 1.2508 | x interval | LOW |
| CK-02 | Compute dollars at risk (since last commit) | 0.6333 | USD | LOW |
| CK-03 | Consecutive failed saves | 0 | count | OK |
| CK-04 | Checksum mismatches (commit or restore) | 0 | count | OK |
| CK-05 | Restore check failures (digest, probe, next batch, frozen) | 0 | count | OK |
| CK-06 | Strict-load or environment refusals | 0 | count | OK |
| CK-07 | NaN guard trips | 1 | count | HIGH |
| CK-08 | Save stall as share of interval | 0.2703 | % | OK |
| CK-09 | Commit latency (stall end to committed) | 0.2094 | s | OK |
| CK-10 | Save backpressure (previous save still running) | 0 | count/day | OK |
| CK-13 | Restore time (job start to first productive step) | 91.86 | s | OK |
| CK-14 | Work lost per incident | 0.6875 | x interval | LOW |
| CK-15 | Protected checkpoint missing or deleted | 0 | count | OK |
| CK-16 | Orphan uncommitted objects in S3 | 0.0 | GB | OK |
| CK-18 | Export parity failures (safetensors) | 0 | count | OK |
| CK-19 | Resharded resumes | 1 | count | LOW |
| CK-20 | Probe value missing for this device class | 0 | count | OK |
| Q-01 | Main run not running (pending time) | 90.27 | s | OK |
| Q-02 | Main run held (exit 78 / crash loop) | 2 | count | HIGH |
| Q-03 | Reschedule latency (job end to running again) | 90.27 | s | LOW |
| Q-04 | Crash-loop holds | 1 | count | MEDIUM |
| Q-05 | Main-run interruptions per day | 843.6182 | count/day | HIGH |
| Q-07 | Allocated but not stepping | 91.863 | s | OK |
| Q-09 | Drained nodes | 2 | count | MEDIUM |
| Q-10 | Node replacement time (drain to spare running the job) | 92.962 | s | OK |
| Q-14 | Team GPU-hours lost to preemption | 0.0 | node-h/week | OK |
| Q-16 | Jobs rejected for no valid account | 1 | count | LOW |
| Q-19 | Goodput of the main run | 22.0485 | % | HIGH |
| Q-20 | MTTR (fault to first productive step) | 85.09 | s | OK |
| Q-21 | Human interventions | 4 | count/week | MEDIUM |
| H-01 | Fatal Xid on a node | 0 | count | OK |
| H-14 | Step watchdog fired (hang) | 2.0677 | count/h | HIGH |
| H-15 | Straggler ratio (rank step time / median) | 1.0841 | ratio | LOW |
| H-17 | Health check runtime (prolog) | 0.01 | s | OK |
| H-18 | Required check skipped | 0 | count | OK |
| H-21 | Spot interruption notice | 1 | count | HIGH |
| H-24 | Loss-spike guard trips | 1 | count | HIGH |
| C-02 | Cost per productive GPU-hour vs list price | 4.5355 | ratio | HIGH |
| C-03 | Wasted share of main-run spend | 77.9515 | % | HIGH |
| C-04 | Cost of one incident | 21.4988 | x interval cost | HIGH |
