# Simulation metrics report: run `v5main`

Cluster price used: 2 × p6-b200.48xlarge = **$227.86/h** (assumed price, see cost/prices.yaml). Interval: 8 s (one interval = $0.51).

> In the simulation a fault is injected about once a minute, while a real run sees one every few hours or days. So goodput and cost-waste numbers here are far worse than production will be. The simulation proves the *mechanism* and measures each step's *duration*; `cost/cost_model.py --measured` projects those durations onto real failure rates.

Metrics in catalog: 74; computed from simulation events: 40.

## Alerts

| Severity | ID | Metric | Value | Caused by the test? | Action |
|---|---|---|---|---|---|
| HIGH | CK-07 | NaN guard trips | 1 count | yes | Rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | Q-02 | Main run held (exit 78 / crash loop) | 2 count | yes | Read the job.held cause; rollback or fix, then release. |
| HIGH | Q-05 | Main-run interruptions per day | 1643.3317 count/day | fault rate | Group incidents by node and cause_class; drain repeat offenders. |
| HIGH | Q-19 | Goodput of the main run | 58.1872 % | fault rate | See the largest non-productive bucket in the report. |
| HIGH | H-14 | Step watchdog fired (hang) | 4.0278 count/h | fault rate | Read the flight-recorder and py-spy dumps; drain the rank that stopped. |
| HIGH | H-15 | Straggler ratio (rank step time / median) | 1.2775 ratio | no | Medium for 3 windows: drain at the next checkpoint. |
| HIGH | H-21 | Spot interruption notice | 1 count | yes | Automatic: drain + save + requeue. Page only if no commit within 90 s. |
| HIGH | H-24 | Loss-spike guard trips | 1 count | yes | Look at the batches before the spike; rollback: --from-step <last good> --skip N --branch <new>. |
| HIGH | C-02 | Cost per productive GPU-hour vs list price | 1.7186 ratio | fault rate | Fix the largest waste bucket (C-03). |
| HIGH | C-03 | Wasted share of main-run spend | 41.8128 % | fault rate | The report lists each bucket in USD. |
| HIGH | C-04 | Cost of one incident | 7.98 x interval cost | yes | Large values: missing emergency save, slow restore, or no spare. |
| MEDIUM | Q-04 | Crash-loop holds | 1 count | yes | Read the logs of the 3 crashes; for the main run this is also Q-02 (high). |
| MEDIUM | Q-09 | Drained nodes | 2 count | yes | Re-test with --full; repair, replace or stop the instance. |
| MEDIUM | Q-21 | Human interventions | 4 count/week | yes | Automate the step that needed a person. |
| LOW | CK-01 | Time since last committed checkpoint | 1.3845 x interval | no | Check checkpoint.failed events; if saves fail, fix storage before the next failure costs a full interval. |
| LOW | CK-02 | Compute dollars at risk (since last commit) | 0.701 USD | no | Same as CK-01. Thresholds are multiples of the cost of one interval. |
| LOW | CK-14 | Work lost per incident | 0.8163 x interval | no | Check that SIGUSR1/SIGTERM reached the trainer and the emergency save committed. |
| LOW | CK-19 | Resharded resumes | 1 count | yes | Note it in the run log and the paper's failure table. |
| LOW | Q-16 | Jobs rejected for no valid account | 1 count | yes | Check the submit system's account mapping. |

## Where the main run's wall-clock time (and money) went

| Bucket | Seconds | Share | USD |
|---|---|---|---|
| checkpoint_stall | 2.61 | 0.3% | $0.17 |
| lost_work | 20.56 | 2.5% | $1.30 |
| detection | 184.65 | 22.3% | $11.69 |
| reschedule | 98.9 | 11.9% | $6.26 |
| restore | 39.88 | 4.8% | $2.52 |
| productive | 482.33 | 58.2% | $30.53 |
| wall_clock | 828.93 | 100.0% | $52.47 |

## Recovery time by fault type (MTTR, seconds)

| Fault | n | p50 | p95 | max |
|---|---|---|---|---|
| hang | 1 | 24.8 | 24.8 | 24.8 |
| health_fail | 1 | 59.53 | 59.53 | 59.53 |
| kill | 3 | 7.1 | 8.4 | 8.4 |
| kill_during_restore | 1 | 14.77 | 14.77 | 14.77 |
| kill_during_save | 1 | 16.22 | 16.22 | 16.22 |
| node_loss | 1 | 46.81 | 46.81 | 46.81 |
| rollback | 2 | 20.53 | 37.54 | 37.54 |
| sigusr1 | 3 | 3.18 | 4.16 | 4.16 |
| spot_notice | 1 | 9.32 | 9.32 | 9.32 |
| two_nodes_down | 1 | 63.84 | 63.84 | 63.84 |

## Incidents

| Fault | Cause class | Detect s | Reschedule s | Restore s | MTTR s | Lost steps | Restored step | Human |
|---|---|---|---|---|---|---|---|---|
| kill | software | 0.11 | 4.72 | 2.27 | 7.1 | 109 | 4831 | no |
| kill | software | 0.11 | 4.44 | 1.99 | 6.55 | 9 | 4961 | no |
| kill | software | 0.11 | 6.24 | 2.05 | 8.4 | 19 | 5091 | no |
| sigusr1 | preemption | 0.44 | 1.44 | 2.28 | 4.16 | 0 | 5112 | no |
| sigusr1 | preemption | 0.46 | 1.17 | 1.55 | 3.18 | 0 | 5133 | no |
| sigusr1 | preemption | 0.52 | 0.95 | 1.57 | 3.04 | 0 | 5169 | no |
| kill_during_save | software | 9.88 | 4.15 | 2.18 | 16.22 | 21 | 5299 | no |
| kill_during_restore | software | 0.11 | 3.58 | 11.08 | 14.77 | 30 | 5430 | no |
| spot_notice | host | 4.97 | 2.19 | 2.16 | 9.32 | 0 | 5538 | no |
| health_fail | gpu | 26.76 | 30.61 | 2.15 | 59.53 | 0 | 5989 | no |
| two_nodes_down | gpu | 24.62 | 36.98 | 2.24 | 63.84 | 0 | 6400 | no |
| hang | software | 21.39 | 1.08 | 2.33 | 24.8 | 20 | 6400 | no |
| node_loss | host | 43.19 | 1.35 | 2.27 | 46.81 | 30 | 6400 | no |
| nan | software | 1.26 | None | None | None | None | None | yes |
| rollback | operator | 35.98 | 0.0 | 1.56 | 37.54 | 86 | 8524 | yes |
| loss_spike | software | 7.24 | None | None | None | None | None | yes |
| rollback | operator | 7.5 | 0.0 | 2.2 | 3.52 | 19 | 8651 | yes |

## Checkpoint

- committed: 72, failed: 0, restores: 15
- stall per save: mean 0.0362 s, p95 0.0847 s; commit latency mean 0.0965 s; size 1151908 bytes (tiny model)

## All computed metrics

| ID | Metric | Value | Unit | Severity |
|---|---|---|---|---|
| CK-01 | Time since last committed checkpoint | 1.3845 | x interval | LOW |
| CK-02 | Compute dollars at risk (since last commit) | 0.701 | USD | LOW |
| CK-03 | Consecutive failed saves | 0 | count | OK |
| CK-04 | Checksum mismatches (commit or restore) | 0 | count | OK |
| CK-05 | Restore check failures (digest | 0 | count | OK |
| CK-06 | Strict-load or environment refusals | 0 | count | OK |
| CK-07 | NaN guard trips | 1 | count | HIGH |
| CK-08 | Save stall as share of interval | 0.4531 | % | OK |
| CK-09 | Commit latency (stall end to committed) | 0.2437 | s | OK |
| CK-10 | Save backpressure (previous save still running) | 0 | count/day | OK |
| CK-13 | Restore time (job start to first productive step) | 11.08 | s | OK |
| CK-14 | Work lost per incident | 0.8163 | x interval | LOW |
| CK-15 | Protected checkpoint missing or deleted | 0 | count | OK |
| CK-16 | Orphan uncommitted objects in S3 | 0.0 | GB | OK |
| CK-18 | Export parity failures (safetensors) | 0 | count | OK |
| CK-19 | Resharded resumes | 1 | count | LOW |
| CK-20 | Probe value missing for this device class | 0 | count | OK |
| Q-01 | Main run not running (pending time) | 36.98 | s | OK |
| Q-02 | Main run held (exit 78 / crash loop) | 2 | count | HIGH |
| Q-03 | Reschedule latency (job end to running again) | 36.98 | s | OK |
| Q-04 | Crash-loop holds | 1 | count | MEDIUM |
| Q-05 | Main-run interruptions per day | 1643.3317 | count/day | HIGH |
| Q-07 | Allocated but not stepping | 20.6 | s | OK |
| Q-09 | Drained nodes | 2 | count | MEDIUM |
| Q-10 | Node replacement time (drain to spare running the job) | 39.473 | s | OK |
| Q-14 | Team GPU-hours lost to preemption | 0.0 | node-h/week | OK |
| Q-16 | Jobs rejected for no valid account | 1 | count | LOW |
| Q-19 | Goodput of the main run | 58.1872 | % | HIGH |
| Q-20 | MTTR (fault to first productive step) | 20.5853 | s | OK |
| Q-21 | Human interventions | 4 | count/week | MEDIUM |
| H-01 | Fatal Xid on a node | 0 | count | OK |
| H-14 | Step watchdog fired (hang) | 4.0278 | count/h | HIGH |
| H-15 | Straggler ratio (rank step time / median) | 1.2775 | ratio | HIGH |
| H-17 | Health check runtime (prolog) | 0.008 | s | OK |
| H-18 | Required check skipped | 0 | count | OK |
| H-21 | Spot interruption notice | 1 | count | HIGH |
| H-24 | Loss-spike guard trips | 1 | count | HIGH |
| C-02 | Cost per productive GPU-hour vs list price | 1.7186 | ratio | HIGH |
| C-03 | Wasted share of main-run spend | 41.8128 | % | HIGH |
| C-04 | Cost of one incident | 7.98 | x interval cost | HIGH |
