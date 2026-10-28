"""Scan the kernel log for NVIDIA Xid errors and classify them (ADR 0003 §1).

In simple words: when a GPU breaks, the driver writes an "Xid" line with a number into the kernel log.
Some numbers mean "this GPU is broken, take the node out" (fatal), some mean "watch it" (warn).
We remember which lines we already saw, so an old error does not take a repaired node out again.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

XID_RE = re.compile(r"NVRM: Xid \(PCI:([0-9A-Fa-f:.]+)\): (\d+)(?:, (.*))?")


def read_kernel_log(path: str = None) -> tuple:
    """Returns (lines, source). source is None when no log can be read (laptop, Colab, no root)."""
    if path:
        with open(path) as f:
            return f.read().splitlines(), path
    for cmd in (["journalctl", "-k", "-o", "short-iso", "--no-pager", "-q"], ["dmesg", "-T"]):
        if shutil.which(cmd[0]):
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                if p.returncode == 0 and p.stdout:
                    return p.stdout.splitlines(), " ".join(cmd[:2])
            except (OSError, subprocess.TimeoutExpired):
                pass
    return [], None


def classify(code: int, table: dict) -> str:
    if code in table.get("fatal", []):
        return "fatal"
    if code in table.get("warn", []):
        return "warn"
    return "unknown"


def scan(lines, table: dict, state_path: str = None) -> dict:
    seen = set()
    if state_path and os.path.exists(state_path):
        with open(state_path) as f:
            seen = set(json.load(f))
    new, old = [], []
    for line in lines:
        m = XID_RE.search(line)
        if not m:
            continue
        h = hashlib.sha256(line.encode()).hexdigest()[:16]
        code = int(m.group(2))
        rec = {"pci": m.group(1), "xid": code, "class": classify(code, table), "text": (m.group(3) or "")[:120]}
        (old if h in seen else new).append(rec)
        seen.add(h)
    if state_path:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(sorted(seen)[-5000:], f)
    worst = "pass"
    if any(r["class"] in ("warn", "unknown") for r in new):
        worst = "warn"     # an unknown code is reported as warn, never ignored
    if any(r["class"] == "fatal" for r in new):
        worst = "fail"
    return {"status": worst, "new": new, "already_seen": len(old)}


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="a saved kernel log; default: journalctl -k or dmesg")
    ap.add_argument("--state", help="file that remembers lines already seen")
    ap.add_argument("--thresholds", default=os.path.join(os.path.dirname(__file__), "thresholds.yaml"))
    a = ap.parse_args()
    table = yaml.safe_load(open(a.thresholds))["xid"]
    lines, source = read_kernel_log(a.log)
    out = scan(lines, table, a.state) if source else {"status": "skipped", "reason": "no kernel log readable"}
    print(json.dumps(out, indent=1))
    return {"pass": 0, "warn": 3, "fail": 4, "skipped": 0}[out["status"]]


if __name__ == "__main__":
    sys.exit(main())
