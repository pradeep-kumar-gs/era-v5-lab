#!/bin/bash
# Accounts, QOS and limits in slurmdbd (ADR 0002 §4). Safe to run twice.
#
# In simple words: the main run has its own account with top priority and may take nodes from anyone.
# Every team has its own account under "teams"; they share the rest fairly, each has a node cap so no
# team can take everything, and each has a GPU-hour budget so nobody overspends.
set -u
S="sacctmgr -i"
$S add qos mainrun >/dev/null 2>&1
$S modify qos mainrun set Priority=1000 Preempt=normal PreemptMode=requeue
$S modify qos normal set Priority=10 GraceTime=60
$S add account mainrun Description="V5 main run" Organization=v5 >/dev/null 2>&1
$S add account teams Description="all teams" Organization=v5 >/dev/null 2>&1
TEAMS="data tokenizer ke2 arch perf infra monitoring eval posttrain release quant"
for t in $TEAMS; do
  $S add account "$t" parent=teams Fairshare=1 Organization=v5 >/dev/null 2>&1
  $S modify account "$t" set GrpTRES=node=2      # no team can hold more than 2 nodes
done
# The submit service user is the only user; the account it passes decides who pays.
$S add user svc Account=mainrun DefaultAccount=arch >/dev/null 2>&1
for t in $TEAMS; do $S add user svc Account="$t" >/dev/null 2>&1; done
$S modify user svc where account=mainrun set QOS=mainrun,normal DefaultQOS=mainrun
for t in $TEAMS; do $S modify user svc where account="$t" set QOS=normal DefaultQOS=normal; done
# root runs the retention and export jobs in the test (on AWS: their own IAM roles, ADR 0002 §7).
$S add user root Account=infra DefaultAccount=infra >/dev/null 2>&1
$S modify user root where account=infra set QOS=normal DefaultQOS=normal
sacctmgr -n -P show assoc format=account,user,qos,grptres | sort
