"""Scan a diagnosis sample pack for manual-review candidates.

This tool does not modify the Parquet file and does not assign final verdicts.
It creates a separate triage CSV; every document must still be read manually.
The detection rules live in detectors.py and are shared with view_pack_web.py.

Usage:
    python diagnosis/tools/scan_pack.py PACK.parquet --output scan_candidates.csv
    python diagnosis/tools/scan_pack.py PACK.parquet --min-confidence medium
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from detectors import CONFIDENCE_RANK, Finding, scan_records  # noqa: F401  (Finding is re-exported)


def scan_pack(pack: Path, min_confidence: str = "low") -> list[dict[str, object]]:
    documents = pq.read_table(pack).to_pylist()
    limit = CONFIDENCE_RANK[min_confidence]
    rows: list[dict[str, object]] = []
    for document, result in zip(documents, scan_records(documents)):
        rows.append({
            "record": result["index"] + 1,
            "doc_id": document["doc_id"],
            "source": document["source"],
            "lang_label": document["lang_label"],
            "chars": result["chars"],
            "suggested_verdict": result["suggested_verdict"],
            "findings": [f for f in result["findings"] if CONFIDENCE_RANK[f.confidence] <= limit],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path, default=Path("scan_candidates.csv"))
    parser.add_argument("--all", action="store_true", help="include records with no candidates")
    parser.add_argument("--min-confidence", choices=("high", "medium", "low"), default="low",
                        help="drop findings below this confidence")
    args = parser.parse_args()

    rows = scan_pack(args.pack, args.min_confidence)
    selected = rows if args.all else [row for row in rows if row["findings"]]
    fieldnames = ("record", "doc_id", "source", "lang_label", "chars", "suggested_verdict",
                  "candidates", "confidence", "evidence", "reason", "hex")
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            findings: list[Finding] = row["findings"]  # type: ignore[assignment]
            writer.writerow({
                "record": row["record"],
                "doc_id": row["doc_id"],
                "source": row["source"],
                "lang_label": row["lang_label"],
                "chars": row["chars"],
                "suggested_verdict": row["suggested_verdict"],
                "candidates": ";".join(dict.fromkeys(f.code for f in findings)),
                "confidence": ";".join(f"{f.code}:{f.confidence}" for f in findings),
                "evidence": " | ".join(f"{f.code}: {f.evidence}" for f in findings),
                "reason": " | ".join(f"{f.code}: {f.reason}" for f in findings),
                "hex": ";".join(dict.fromkeys(f.hex_value for f in findings if f.hex_value and not f.sensitive)),
            })

    counts = Counter(f"{f.code}/{f.confidence}" for row in rows for f in row["findings"])
    summary = ", ".join(f"{code}={count}" for code, count in sorted(counts.items())) or "none"
    print(f"Scanned {len(rows)} documents; wrote {len(selected)} rows to {args.output}")
    print(f"Finding counts (code/confidence): {summary}")
    print("These are triage hints only. Read every document and decide the final labels manually.")


if __name__ == "__main__":
    main()
