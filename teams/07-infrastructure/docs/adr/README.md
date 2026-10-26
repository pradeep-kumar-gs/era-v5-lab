# Team 07 architecture decision records

One ADR per decision area. Each records the context, the options we weighed, the choice, and what it costs us.
Status moves **Proposed → Accepted** once the lead (Suresh) approves the PR. Later changes go in a new ADR
that supersedes the old one; accepted ADRs are not rewritten.

| ADR | Title | Status | Design doc |
|---|---|---|---|
| [0001](0001-checkpoint-format-and-commit.md) | Checkpoint format, commit protocol and restore checks | Proposed | [checkpoint_design.md](../checkpoint_design.md) |
| [0002](0002-job-queue-slurm.md) | Job queue: self-managed Slurm on EC2 with two account tiers | Proposed | [queue_design.md](../queue_design.md) |
| [0003](0003-node-health-checks.md) | Node health checks: JSON report, skip-not-fail, drain on hard faults | Proposed | `../../health/` |

Template: Context · Decision drivers · Options considered · Decision · Consequences · Validation · Open points.
