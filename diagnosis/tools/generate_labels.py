"""Generate a manual-review labels.csv scaffold from a diagnosis pack.

By default, only doc_id is populated. Use --prefill-candidates to copy the
conservative findings from scan_pack.py into the disease/evidence/hex columns.
Prefilled values are not final labels and must be manually confirmed.

Usage:
    python diagnosis/tools/generate_labels.py PACK.parquet --output labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scan_pack import Finding, scan_pack


FIELDNAMES = ("doc_id", "read", "verdict", "diseases", "evidence", "hex", "note")


def label_row(row: dict[str, object], prefill_candidates: bool) -> dict[str, object]:
    output: dict[str, object] = {
        "doc_id": row["doc_id"],
        "read": "",
        "verdict": "",
        "diseases": "",
        "evidence": "",
        "hex": "",
        "note": "",
    }
    if not prefill_candidates:
        return output

    findings: list[Finding] = row["findings"]  # type: ignore[assignment]
    if not findings:
        return output

    output.update({
        "diseases": ";".join(dict.fromkeys(finding.code for finding in findings)),
        "evidence": " | ".join(finding.evidence for finding in findings),
        "hex": ";".join(
            dict.fromkeys(finding.hex_value for finding in findings if finding.hex_value)
        ),
        "note": "AUTO-CANDIDATES ONLY - verify manually",
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path, default=Path("labels.csv"))
    parser.add_argument(
        "--prefill-candidates",
        action="store_true",
        help="prefill scanner hints; still leaves read and verdict blank",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file (can destroy manual label work)",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        parser.error(f"output already exists: {args.output} (use --force only if intentional)")
    if not args.pack.is_file():
        parser.error(f"pack does not exist: {args.pack}")

    rows = scan_pack(args.pack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.force else "x"
    with args.output.open(mode, newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(label_row(row, args.prefill_candidates) for row in rows)

    prefill_message = " with unverified scanner hints" if args.prefill_candidates else ""
    print(f"Created {args.output} with {len(rows)} rows{prefill_message}.")
    print("Manually fill read and verdict, verify every candidate, and read every document.")


if __name__ == "__main__":
    main()
