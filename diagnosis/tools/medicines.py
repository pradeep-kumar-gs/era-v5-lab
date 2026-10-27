"""Medicines: one fix or drop rule per disease code, and the medicine-sheet writer.

A medicine is either
  * a text fix  -- fix(text) -> new text, applied to every document, or
  * a drop rule -- drop(finding, record_no) -> bool, decided from the detectors.py findings.

The sheet (medicine.md, from diagnosis/templates/medicine_TEMPLATE.md) is built
from the HUMAN labels in labels.csv, not from the scanner. For every disease
labelled in more than five documents it gives the count, three examples, the
rule in words and code, the action, the harm test on the gold set, and the side
effects (known ones, plus the documents where the rule disagrees with your
labels, plus the "medicine wrong here" notes recorded in the viewer).

Usage:
    python diagnosis/tools/medicines.py PACK.parquet --labels labels.csv \
        [--gold GOLD.parquet] [--handle you] [--output medicine.md]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import inspect
import json
import os
import re
import textwrap
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from detectors import (HIGH, LICENCE_RE, MOJIBAKE_RE, SECRET_PATTERNS, Finding, _leading_comment, parse_doc,
                       scan_records)

HARM_LIMIT = 0.01        # a medicine that changes or drops more than 1% of the gold set fails
MIN_COUNT = 5            # write a section for each disease found MORE than five times


@dataclass
class Medicine:
    code: str
    action: str                                   # "fix" or "drop"
    words: str
    fix: Callable[[str], str] | None = None
    drop: Callable[[Finding, int], bool] | None = None
    side_effects: tuple[str, ...] = ()

    def source(self) -> str:
        return textwrap.dedent(inspect.getsource(self.fix or self.drop)).strip()


# ---------------------------------------------------------------------------
# Commit packs: apply a fix to the commit message / the code sections only.
# ---------------------------------------------------------------------------
SECTION_SPLIT_RE = re.compile(r"(^### (?:commit message|file path|old contents|new contents)[ \t]*\n)", re.M)


def _per_section(text: str, fn: Callable[[str], str], sections: tuple[str, ...]) -> str:
    """Apply fn to the bodies of the named sections; documents without sections count as 'text'."""
    parts = SECTION_SPLIT_RE.split(text)
    if len(parts) == 1:
        return fn(text) if "text" in sections else text
    out, current = [parts[0]], ""
    for part in parts[1:]:
        if SECTION_SPLIT_RE.fullmatch(part):
            current = part[4:].strip()
            out.append(part)
        else:
            out.append(fn(part) if current in sections else part)
    return "".join(out)


PROSE = ("commit message", "text")
CODE = ("old contents", "new contents", "text")


# ---------------------------------------------------------------------------
# Text fixes
# ---------------------------------------------------------------------------
def fix_invisible(text: str) -> str:
    """A1: strip invisible junk, keep ZWJ/ZWNJ next to Indic letters, turn look-alike spaces into ' '."""
    text = re.sub("[​⁠᠎­‎‏‪-‮⁦-⁩﻿]", "", text)
    text = re.sub("(?<![ऀ-෿])[‌‍](?![ऀ-෿])", "", text)
    text = re.sub("[  ]", "\n", text)
    return re.sub("[   -   　]", " ", text)


def fix_controls(text: str) -> str:
    """A2: CRLF / lone CR -> LF, then delete the other C0/C1 control characters (tab and newline stay)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)


def fix_mojibake(text: str) -> str:
    """A3/E5: re-decode runs of UTF-8 that were read as cp1252/latin-1 ('Ã©' -> 'é')."""
    def redecode(m: re.Match) -> str:
        for enc in ("cp1252", "latin-1"):
            try:
                return m.group(0).encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        return m.group(0)
    return MOJIBAKE_RE.sub(redecode, text)


def fix_nfc(text: str) -> str:
    """A5: normalise to NFC."""
    return unicodedata.normalize("NFC", text)


def fix_entities(text: str) -> str:
    """B1: decode HTML entities in prose (commit message), never inside code."""
    return _per_section(text, html.unescape, PROSE)


def fix_tags(text: str) -> str:
    """B2: remove HTML tags from prose (commit message), keep their inner text."""
    tag = re.compile(r"</?(?:div|span|p|br|a|b|i|u|em|strong|ul|ol|li|table|tr|td|th|img|h[1-6]|pre|code|font|"
                     r"center|hr)\b[^>]{0,200}/?>", re.I)
    return _per_section(text, lambda s: tag.sub("", s), PROSE)


def fix_trailers(text: str) -> str:
    """C1: delete Signed-off-by / Change-Id / git-svn-id ... trailer lines from the commit message."""
    trailer = re.compile(r"^(?:Signed-off-by|Change-Id|git-svn-id|Reviewed-on|Reviewed-by|Tested-by|Acked-by|Cc|"
                         r"Former-commit-id):.*\n?", re.I | re.M)
    return _per_section(text, lambda s: trailer.sub("", s), PROSE)


def fix_repeated_lines(text: str) -> str:
    """C2: collapse 3+ identical consecutive non-blank lines (8+ chars) into one, inside each section."""
    def collapse(s: str) -> str:
        return re.sub(r"^([^\n]{8,})\n(?:\1\n){2,}", r"\1\n", s, flags=re.M)
    return _per_section(text, collapse, PROSE + CODE)


def fix_secrets(text: str) -> str:
    """E3: replace secret values with a placeholder (key patterns from detectors.SECRET_PATTERNS)."""
    for kind, _, rx in SECRET_PATTERNS:
        text = rx.sub(f"<REDACTED_{kind.upper().replace(' ', '_')}>", text)
    return text


def fix_licence_header(text: str) -> str:
    """E4: remove a leading comment block that contains licence text, from every code section."""
    def strip(s: str) -> str:
        off, block = _leading_comment(s)
        if off < 0 or not LICENCE_RE.search(block):
            return s
        return s[:off] + s[off + len(block):].lstrip("\n")
    return _per_section(text, strip, CODE)


def fix_emails(text: str) -> str:
    """F1: replace real-looking email addresses with a placeholder (example.com and test addresses stay)."""
    email = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
    keep = re.compile(r"(?i)@(?:[\w-]+\.)*(?:example\.(?:com|org|net)|test|invalid|localhost)$")
    return email.sub(lambda m: m.group(0) if keep.search(m.group(0)) else "<EMAIL>", text)


def fix_ghost_tokens(text: str) -> str:
    """B3: break chat/special markers so they are no longer our special tokens ('<|im_start|>' -> '< |im_start| >')."""
    text = re.sub(r"<\|([a-z_]{2,30})\|>", r"< |\1| >", text, flags=re.I)
    return re.sub(r"\[(/?)INST\]", r"[\1 INST ]", text)


# ---------------------------------------------------------------------------
# Drop rules (decided from the detector finding)
# ---------------------------------------------------------------------------
def drop_high(f: Finding, record_no: int) -> bool:
    """Drop the document when the detector finding for this code is high confidence."""
    return f.confidence == HIGH


def drop_later_copy(f: Finding, record_no: int) -> bool:
    """H1/H2: keep the first copy, drop each later one (the finding names the other record numbers)."""
    m = re.search(r"record(?:\(s\))? ([\d,]+)", f.evidence)
    others = [int(n) for n in m.group(1).split(",") if n] if m else []
    return f.confidence == HIGH and bool(others) and min(others) < record_no


def drop_replacement(f: Finding, record_no: int) -> bool:
    """A4: drop when 3 or more U+FFFD replacement characters (lost bytes cannot be repaired)."""
    return int(f.evidence.split(" x ")[0]) >= 3


MEDICINES: dict[str, Medicine] = {m.code: m for m in (
    Medicine("A1", "fix", "Delete zero-width spaces, BOMs, soft hyphens, LRM/RLM and bidi controls; delete ZWJ/ZWNJ "
             "only when neither neighbour is an Indic letter; replace look-alike spaces (NBSP, U+2000-U+200A, "
             "U+202F, U+3000) with a normal space.", fix=fix_invisible,
             side_effects=("Emoji ZWJ sequences (family/flag emoji) are split into separate emoji.",
                           "Tests or string literals that check NBSP/ZWSP handling lose the character they test.",
                           "RLM/LRM are needed in real Arabic/Hebrew text.")),
    Medicine("A2", "fix", "Convert CRLF and lone CR to LF, then delete the other control characters "
             "(U+0000-U+001F except tab/newline, U+007F-U+009F).", fix=fix_controls,
             side_effects=("Form feed (U+000C) used as a page break in GNU-style C/Python sources is removed.",
                           "C1 characters may be mojibake; deleting them loses the letter that A3 could repair.")),
    Medicine("A3", "fix", "Re-decode mojibake runs (UTF-8 read as cp1252/latin-1) back to the right characters.",
             fix=fix_mojibake, side_effects=("Encoding tests and ftfy-style fixtures that contain mojibake on "
                                             "purpose are 'repaired'.",)),
    Medicine("E5", "fix", "Same as A3, inside code strings and comments.", fix=fix_mojibake,
             side_effects=("Encoding test fixtures that contain mojibake on purpose are 'repaired'.",)),
    Medicine("A4", "drop", "Drop the document when it holds 3 or more U+FFFD replacement characters.",
             drop=drop_replacement, side_effects=("Unicode/decoder code that uses U+FFFD literally is dropped.",)),
    Medicine("A5", "fix", "Normalise the whole text to NFC.", fix=fix_nfc,
             side_effects=("Normalisation tests that keep NFD input on purpose are changed.",)),
    Medicine("B1", "fix", "html.unescape() the commit message (prose only, never the code).", fix=fix_entities,
             side_effects=("A message that talks about escaping ('use &amp; not &') loses its meaning.",)),
    Medicine("B2", "fix", "Remove HTML tags from the commit message, keep their inner text.", fix=fix_tags,
             side_effects=("A message that names a tag ('wrap it in <pre>') loses the tag name.",)),
    Medicine("B3", "fix", "Break ghost chat markers so they no longer match our special tokens "
             "('<|im_start|>' -> '< |im_start| >', '[INST]' -> '[ INST ]').", fix=fix_ghost_tokens,
             side_effects=("Tokenizer and LLM code that needs the exact marker string no longer works as code; "
                           "dropping the document is the stricter alternative.",)),
    Medicine("C1", "fix", "Delete Signed-off-by / Change-Id / git-svn-id / Reviewed-by ... trailer lines from the "
             "commit message.", fix=fix_trailers, side_effects=("Credit lines for reviewers are lost.",)),
    Medicine("C2", "fix", "Collapse 3 or more identical consecutive lines (8+ chars) into one, inside each section.",
             fix=fix_repeated_lines, side_effects=("Test vectors and data tables with deliberately repeated rows "
                                                   "are shortened, which can change program behaviour.",)),
    Medicine("C3", "drop", "Drop documents whose new contents stop mid-block (high-confidence C3: unclosed comment "
             "or string).", drop=drop_high,
             side_effects=("Files that are legitimately partial (snippets, templates) are dropped.",)),
    Medicine("C4", "drop", "Drop documents with two joined records (high-confidence C4).", drop=drop_high,
             side_effects=("A real full rewrite of a file is not two documents; only medium C4 catches those, "
                           "and medium is not dropped.",)),
    Medicine("E1", "drop", "Drop minified or generated files (high-confidence E1: 1000+ char lines, "
             "'auto-generated'/'do not edit' markers, .min.js/_pb2.py names).", drop=drop_high,
             side_effects=("Hand-written files that keep a 'generated by' template comment are kept (medium only), "
                           "but a hand-edited generated file is dropped.",)),
    Medicine("E2", "drop", "Drop documents where data blobs are more than 30% of a code section.", drop=drop_high,
             side_effects=("Small embedded fixtures below 30% stay in the data.",)),
    Medicine("E3", "fix", "Replace each secret value with <REDACTED_KIND> (AWS/GitHub/Slack/Google/Stripe keys, private "
             "key headers, JWTs, passwords in URLs).", fix=fix_secrets,
             side_effects=("Fake keys in tests are redacted too (harmless).",
                           "High-entropy assignments are only flagged, not redacted: review them by hand.")),
    Medicine("E4", "fix", "Remove the leading comment block when it contains licence text, in old and new contents.",
             fix=fix_licence_header,
             side_effects=("A module docstring that mentions a licence is removed with its other text.",
                           "Licence attribution must be kept in metadata (meta.license), not lost.")),
    Medicine("E6", "drop", "Drop documents whose file is not source code (README, JSON, YAML, lock files...).",
             drop=drop_high, side_effects=("Code-like config files the pack wants (e.g. .json schemas) are lost.",)),
    Medicine("F1", "fix", "Replace real-looking email addresses with <EMAIL>; example.com/test/localhost stay.",
             fix=fix_emails, side_effects=("Email-parsing tests lose their sample addresses.",
                                           "Organisation addresses (a project's security or support mailbox) are replaced too.")),
    Medicine("F2", "drop", "Drop documents that contain a benchmark problem or solution (confirmed by hand).",
             drop=drop_high, side_effects=("The detector only finds markers; unmarked leaks are not caught.",)),
    Medicine("H1", "drop", "Keep the first copy of each exact duplicate, drop the later ones.", drop=drop_later_copy),
    Medicine("H2", "drop", "Keep the first of each near-duplicate group (90%+ shingle overlap), drop the later ones.",
             drop=drop_later_copy, side_effects=("Two real versions of one file (forks, backports) lose one.",)),
)}


# ---------------------------------------------------------------------------
# Applying medicines
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    status: str                    # "unchanged", "changed", "dropped"
    diff: list[str] = field(default_factory=list)


def _short_diff(old: str, new: str, limit: int = 24) -> list[str]:
    lines = []
    for line in difflib.unified_diff(old.split("\n"), new.split("\n"), lineterm="", n=0):
        if line.startswith(("---", "+++")):
            continue
        lines.append(line if len(line) <= 160 else line[:159] + "…")
        if len(lines) >= limit:
            lines.append("…")
            break
    return lines


def apply(med: Medicine, text: str, findings: list[Finding], record_no: int) -> Outcome:
    if med.fix:
        new = med.fix(text)
        return Outcome("changed", _short_diff(text, new)) if new != text else Outcome("unchanged")
    for f in findings:
        if f.code == med.code and med.drop(f, record_no):
            return Outcome("dropped")
    return Outcome("unchanged")


def run_on(records: list[dict], med: Medicine, scan: list[dict] | None = None) -> list[Outcome]:
    scan = scan if scan is not None else (scan_records(records) if med.drop else [{"findings": []}] * len(records))
    return [apply(med, r.get("text") or "", s["findings"], i + 1) for i, (r, s) in enumerate(zip(records, scan))]


# ---------------------------------------------------------------------------
# Medicine sheet
# ---------------------------------------------------------------------------
def read_labels(path: str) -> dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f) if r.get("doc_id")}


def notes_path(labels_path: str) -> str:
    return os.path.splitext(labels_path)[0] + ".medicine_notes.json"


def read_notes(labels_path: str) -> list[dict]:
    path = notes_path(labels_path)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _codes(row: dict) -> list[str]:
    return [c.strip().upper() for c in (row.get("diseases") or "").split(";") if c.strip()]


def _evidence_for(row: dict, code: str) -> str:
    for line in (row.get("evidence") or "").split("\n"):
        if line.startswith(code + ":"):
            return line[len(code) + 1:].strip()
    return (row.get("evidence") or "").split("\n")[0].strip()


def build_sheet(records: list[dict], labels: dict[str, dict], gold: list[dict] | None,
                notes: list[dict], handle: str = "<your GitHub handle>") -> tuple[str, list[dict]]:
    pack_id = (records[0].get("pack_id") if records else "") or "<pack_id>"
    index = {r["doc_id"]: i for i, r in enumerate(records)}
    labels = {d: r for d, r in labels.items()
              if any((r.get(k) or "").strip() for k in ("read", "verdict", "diseases", "evidence", "hex", "note"))}
    by_code: defaultdict[str, list[str]] = defaultdict(list)
    for did, row in labels.items():
        if did in index:
            for code in dict.fromkeys(_codes(row)):
                by_code[code].append(did)
    pack_scan = scan_records(records)
    gold_scan = scan_records(gold) if gold else None
    summary, parts = [], [f"# Medicine sheet: {pack_id} ({handle})", "",
                          f"Built from {len(labels)} labelled rows of {len(records)} documents. One section for each "
                          f"disease labelled in more than {MIN_COUNT} documents.", ""]
    number = 0
    for code, dids in sorted(by_code.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(dids) <= MIN_COUNT:
            summary.append({"code": code, "count": len(dids), "status": f"<= {MIN_COUNT}, no section"})
            continue
        number += 1
        med = MEDICINES.get(code)
        dids_sorted = sorted(dids, key=lambda d: index[d])
        parts += [f"## Medicine {number}: {code}", ""]
        ex = "; ".join(f"`{d}` ({_evidence_for(labels[d], code)})" for d in dids_sorted[:3])
        parts += [f"1. **Disease:** {code}, {len(dids)} documents in this pack. Examples: {ex}."]
        if not med:
            parts += ["2. **Medicine:** _no automatic rule for this code: write the rule by hand._",
                      "3. **Action:** _fix or drop_", "4. **Harm test:** _not run_", "5. **Side effects:** _to do_", ""]
            summary.append({"code": code, "count": len(dids), "status": "no rule (write by hand)"})
            continue
        parts += [f"2. **Medicine:** {med.words}", "", "   ```python",
                  textwrap.indent(med.source(), "   "), "   ```", "",
                  f"3. **Action:** {'fix the text' if med.action == 'fix' else 'drop the document'}."]

        if gold:
            outs = run_on(gold, med, gold_scan)
            changed = sum(o.status == "changed" for o in outs)
            dropped = sum(o.status == "dropped" for o in outs)
            share = (changed + dropped) / max(1, len(gold))
            verdict = "PASS" if share <= HARM_LIMIT else "FAIL"
            parts.append(f"4. **Harm test:** gold set of {len(gold)} documents: {changed} changed, {dropped} dropped "
                         f"({share:.1%}). **{verdict}** (limit {HARM_LIMIT:.0%}).")
        else:
            verdict = "not run"
            outs_clean = [run_on([records[index[d]]], med, [pack_scan[index[d]]])[0]
                          for d, row in labels.items() if d in index and (row.get("verdict") or "") == "clean"]
            hit = sum(o.status != "unchanged" for o in outs_clean)
            parts.append(f"4. **Harm test:** _not run: no gold set given (`--gold`)._ Proxy on this pack: it changes "
                         f"or drops {hit} of the {len(outs_clean)} documents you labelled clean.")

        pack_outs = run_on(records, med, pack_scan)
        labelled = set(dids)
        missed = [d for d in dids_sorted if pack_outs[index[d]].status == "unchanged"]
        extra = [r["doc_id"] for r, o in zip(records, pack_outs)
                 if o.status != "unchanged" and r["doc_id"] in labels and r["doc_id"] not in labelled]
        effects = list(med.side_effects)
        if missed:
            effects.append(f"Misses {len(missed)} of the {len(dids)} documents you labelled {code} "
                           f"(e.g. {', '.join('`'+d+'`' for d in missed[:3])}): those need a hand fix or a better rule.")
        if extra:
            effects.append(f"Also {'changes' if med.action == 'fix' else 'drops'} {len(extra)} labelled document(s) "
                           f"you did not mark {code} (e.g. {', '.join('`'+d+'`' for d in extra[:3])}): check them.")
        for n in notes:
            if n.get("code") == code:
                effects.append(f"`{n['doc_id']}`: {n['note']}")
        parts.append("5. **Side effects:**")
        parts += [f"   - {e}" for e in effects] or ["   - none found"]
        parts.append("")
        summary.append({"code": code, "count": len(dids), "status": f"harm test {verdict}",
                        "missed": len(missed), "extra": len(extra)})
    if not number:
        parts.append(f"_No disease is labelled in more than {MIN_COUNT} documents yet._")
    return "\n".join(parts) + "\n", summary


def main() -> None:
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--gold", help="gold-set Parquet for the harm test")
    ap.add_argument("--handle", default="<your GitHub handle>")
    ap.add_argument("--output", help="default: medicine.md next to the labels file")
    a = ap.parse_args()
    records = pq.read_table(a.pack).to_pylist()
    gold = pq.read_table(a.gold).to_pylist() if a.gold else None
    text, summary = build_sheet(records, read_labels(a.labels), gold, read_notes(a.labels), a.handle)
    out = a.output or os.path.join(os.path.dirname(os.path.abspath(a.labels)), "medicine.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {out}")
    for s in summary:
        print(f"  {s['code']}: {s['count']} docs, {s['status']}")


if __name__ == "__main__":
    main()
