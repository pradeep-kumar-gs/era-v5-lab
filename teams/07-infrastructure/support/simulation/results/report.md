# Simulation metrics report: run `v5main`

Cluster price used: 2 × p6-b200.48xlarge = **$227.86/h** (assumed price, see metrics/prices.yaml). Interval: 8 s (one interval = $0.51).

> In the simulation a fault is injected about once a minute, while a real run sees one every few hours or days. So goodput and cost-waste numbers here are far worse than production will be. The simulation proves the *mechanism* and measures each step's *duration*; `support/cost/cost_model.py --measured` projects those durations onto real failure rates.

Metrics in catalog: 74; computed from simulation events: 40.

## Alerts

| Severity | ID | Metric | Value | Caused by the test? | Action |
|---|---|---|---|---|---|
| HIGH | CK-07 | NaN guard trips | 1 count | yes | Rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | Q-02 | Main run held (exit 78 / crash loop) | 2 count | yes | Read the job.held cause; rollback or fix, then release. |
| HIGH | Q-05 | Main-run interruptions per day | 1191.6246 count/day | fault rate | Group incidents by node and cause_class; drain repeat offenders. |
| HIGH | Q-19 | Goodput of the main run | 26.9177 % | fault rate | See the largest non-productive bucket in the report. |
| HIGH | H-14 | Step watchdog fired (hang) | 2.9206 count/h | fault rate | Read the stack dump in the job log and the NCCL flight-recorder trace; drain the rank that stopped. |
| HIGH | H-21 | Spot interruption notice | 1 count | yes | Automatic: drain + save + requeue. Page only if no commit within 90 s. |
| HIGH | H-24 | Loss-spike guard trips | 1 count | yes | Look at the batches before the spike; rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | C-02 | Cost per productive GPU-hour vs list price | 3.715 ratio | fault rate | Fix the largest waste bucket (C-03). |
| HIGH | C-03 | Wasted share of main-run spend | 73.0823 % | fault rate | The report lists each bucket in USD. |
| HIGH | C-04 | Cost of one incident | 14.0375 x interval cost | yes | Large values: missing emergency save, slow restore, or no spare. |
| MEDIUM | Q-04 | Crash-loop holds | 1 count | yes | Read the logs of the 3 crashes; for the main run this is also Q-02 (high). |
| MEDIUM | Q-09 | Drained nodes | 2 count | yes | Re-test with --full; repair, replace or stop the instance. |
| MEDIUM | Q-21 | Human interventions | 4 count/week | yes | Automate the step that needed a person. |
| MEDIUM | H-15 | Straggler ratio (rank step time / median) | 1.1593 ratio | no | Medium for 3 windows: drain at the next checkpoint. |
| LOW | CK-01 | Time since last committed checkpoint | 1.2285 x interval | no | Check checkpoint.failed events; if saves fail, fix storage before the next failure costs a full interval. |
| LOW | CK-02 | Compute dollars at risk (since last commit) | 0.6221 USD | no | Same as CK-01. Thresholds are multiples of the cost of one interval. |
| LOW | CK-14 | Work lost per incident | 0.8688 x interval | no | Check that SIGUSR1/SIGTERM reached the trainer and the emergency save committed. |
| LOW | CK-19 | Resharded resumes | 1 count | yes | Note it in the run log and the paper's failure table. |
| LOW | Q-16 | Jobs rejected for no valid account | 1 count | yes | Check the submit system's account mapping. |

## Where the main run's wall-clock time (and money) went

| Bucket | Seconds | Share | USD |
|---|---|---|---|
| checkpoint_stall | 1.86 | 0.2% | $0.12 |
| lost_work | 19.25 | 1.7% | $1.22 |
| detection | 142.66 | 12.2% | $9.03 |
| reschedule | 601.58 | 51.4% | $38.08 |
| restore | 89.11 | 7.6% | $5.64 |
| productive | 314.72 | 26.9% | $19.92 |
| wall_clock | 1169.17 | 100.0% | $74.00 |

## Recovery time by fault type (MTTR, seconds)

| Fault | n | p50 | p95 | max |
|---|---|---|---|---|
| hang | 1 | 55.85 | 55.85 | 55.85 |
| health_fail | 1 | 58.69 | 58.69 | 58.69 |
| kill | 3 | 50.45 | 59.28 | 59.28 |
| kill_during_restore | 1 | 109.99 | 109.99 | 109.99 |
| kill_during_save | 1 | 49.83 | 49.83 | 49.83 |
| node_loss | 1 | 88.76 | 88.76 | 88.76 |
| rollback | 2 | 2.76 | 2.91 | 2.91 |
| sigusr1 | 3 | 58.43 | 59.25 | 59.25 |
| spot_notice | 1 | 57.59 | 57.59 | 57.59 |
| two_nodes_down | 1 | 59.01 | 59.01 | 59.01 |

## Incidents

| Fault | Cause class | Detect s | Reschedule s | Restore s | MTTR s | Lost steps | Restored step | Human |
|---|---|---|---|---|---|---|---|---|
| kill | software | 0.11 | 56.95 | 2.22 | 59.28 | 28 | 1942 | no |
| kill | software | 0.11 | 47.46 | 2.09 | 49.66 | 28 | 2072 | no |
| kill | software | 0.11 | 48.49 | 1.85 | 50.45 | 18 | 2202 | no |
| sigusr1 | preemption | 0.48 | 55.74 | 2.21 | 58.43 | 0 | 2235 | no |
| sigusr1 | preemption | 0.43 | 56.65 | 2.16 | 59.25 | 0 | 2252 | no |
| sigusr1 | preemption | 0.45 | 55.66 | 1.48 | 57.59 | 0 | 2281 | no |
| kill_during_save | software | 7.68 | 40.31 | 1.84 | 49.83 | 39 | 2411 | no |
| kill_during_restore | software | 0.1 | 47.41 | 62.47 | 109.99 | 39 | 2541 | no |
| spot_notice | host | 1.52 | 54.51 | 1.55 | 57.59 | 0 | 2590 | no |
| health_fail | gpu | 25.48 | 31.1 | 2.12 | 58.69 | 0 | 3037 | no |
| two_nodes_down | gpu | 23.66 | 32.95 | 2.4 | 59.01 | 0 | 3472 | no |
| hang | software | 21.4 | 32.89 | 1.55 | 55.85 | 18 | 3472 | no |
| node_loss | host | 45.01 | 41.46 | 2.29 | 88.76 | 18 | 3472 | no |
| nan | software | 0.89 | None | None | None | None | None | yes |
| rollback | operator | 1.2 | 0.0 | 1.41 | 2.61 | 117 | 5843 | yes |
| loss_spike | software | 7.17 | None | None | None | None | None | yes |
| rollback | operator | 6.86 | 0.0 | 1.47 | 2.91 | 19 | 5971 | yes |

## Checkpoint

- committed: 51, failed: 0, restores: 15
- stall per save: mean 0.0364 s, p95 0.082 s; commit latency mean 0.0911 s; size 1151992 bytes (tiny model)

## All computed metrics

| ID | Metric | Value | Unit | Severity |
|---|---|---|---|---|
| CK-01 | Time since last committed checkpoint | 1.2285 | x interval | LOW |
| CK-02 | Compute dollars at risk (since last commit) | 0.6221 | USD | LOW |
| CK-03 | Consecutive failed saves | 0 | count | OK |
| CK-04 | Checksum mismatches (commit or restore) | 0 | count | OK |
| CK-05 | Restore check failures (digest, probe, next batch, frozen) | 0 | count | OK |
| CK-06 | Strict-load or environment refusals | 0 | count | OK |
| CK-07 | NaN guard trips | 1 | count | HIGH |
| CK-08 | Save stall as share of interval | 0.4556 | % | OK |
| CK-09 | Commit latency (stall end to committed) | 0.1518 | s | OK |
| CK-10 | Save backpressure (previous save still running) | 0 | count/day | OK |
| CK-13 | Restore time (job start to first productive step) | 62.47 | s | OK |
| CK-14 | Work lost per incident | 0.8688 | x interval | LOW |
| CK-15 | Protected checkpoint missing or deleted | 0 | count | OK |
| CK-16 | Orphan uncommitted objects in S3 | 0.0 | GB | OK |
| CK-18 | Export parity failures (safetensors) | 0 | count | OK |
| CK-19 | Resharded resumes | 1 | count | LOW |
| CK-20 | Probe value missing for this device class | 0 | count | OK |
| Q-01 | Main run not running (pending time) | 56.95 | s | OK |
| Q-02 | Main run held (exit 78 / crash loop) | 2 | count | HIGH |
| Q-03 | Reschedule latency (job end to running again) | 56.95 | s | OK |
| Q-04 | Crash-loop holds | 1 | count | MEDIUM |
| Q-05 | Main-run interruptions per day | 1191.6246 | count/day | HIGH |
| Q-07 | Allocated but not stepping | 62.47 | s | OK |
| Q-09 | Drained nodes | 2 | count | MEDIUM |
| Q-10 | Node replacement time (drain to spare running the job) | 56.367 | s | OK |
| Q-14 | Team GPU-hours lost to preemption | 0.0 | node-h/week | OK |
| Q-16 | Jobs rejected for no valid account | 1 | count | LOW |
| Q-19 | Goodput of the main run | 26.9177 | % | HIGH |
| Q-20 | MTTR (fault to first productive step) | 54.66 | s | OK |
| Q-21 | Human interventions | 4 | count/week | MEDIUM |
| H-01 | Fatal Xid on a node | 0 | count | OK |
| H-14 | Step watchdog fired (hang) | 2.9206 | count/h | HIGH |
| H-15 | Straggler ratio (rank step time / median) | 1.1593 | ratio | MEDIUM |
| H-17 | Health check runtime (prolog) | 0.011 | s | OK |
| H-18 | Required check skipped | 0 | count | OK |
| H-21 | Spot interruption notice | 1 | count | HIGH |
| H-24 | Loss-spike guard trips | 1 | count | HIGH |
| C-02 | Cost per productive GPU-hour vs list price | 3.715 | ratio | HIGH |
| C-03 | Wasted share of main-run spend | 73.0823 | % | HIGH |
| C-04 | Cost of one incident | 14.0375 | x interval cost | HIGH |
