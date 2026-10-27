"""Web viewer for a sample pack. Same logic as view_pack.py, in the browser.

Replaces the terminal viewer with a single-page web app (Python standard
library only -- http.server, no web framework). It reuses the exact
invisible-character and hex-dump logic from view_pack.py so the diagnostic
view is byte-for-byte faithful to the script.

Features
  * Diagnostic pane: each document rendered with hidden characters made
    visible (same VISIBLE map and control-char rule as view_pack.py).
  * Character offset on cursor: click or move the cursor in the diagnostic
    pane and the status bar shows the CODE-POINT offset into the raw text
    (identical to Python text[offset]) plus the code point and Unicode name.
  * Selection -> disease + evidence: select a span, pick a disease, and that
    code is added to diseases. Evidence keeps one line per code; further hits
    for the same code are separated with "; ". Per-code rules (EVIDENCE_RULES)
    can change that text — C2 counts how often the line repeats. Hex still
    fills from the selection (U+XXXX, labels format).
  * Markdown preview: a rendered preview of the document (toggle).
  * Hex dump pane: same offset / code-point / Unicode-name list as the script.
  * Detected panel: detectors.py scans the whole pack at start-up. Each
    finding shows code, confidence, section, line and the reason the rule
    fired; its span is highlighted in the diagnostic pane. Accept copies the
    code, evidence and (A/G only) hex into the row; Reject hides it. The
    header filter jumps between documents with a given code, suggestion or
    review state. F1/E3 evidence is type + offset only.
  * Commit context: for commit packs, old vs new contents are diffed up
    front (green = added, red = removed lines) so the text repeated between
    the two copies is not mistaken for a duplicate.
  * Checks panel: every disease code with what was checked and its result
    (hit / low hint / passed / n/a / read by eye), plus why the document
    is suggested clean, fix or drop.
  * Medicine: for every code in the diseases field, the medicine from
    medicines.py is run on this document and its effect is shown (diff or
    "would drop"). "Medicine wrong here" records a side effect. The
    "medicine sheet" button writes medicine.md (from medicine_TEMPLATE.md)
    next to the labels file, with the harm test when --gold is given.
  * Per-row annotation: textboxes for every CSV column (read, verdict,
    diseases, evidence, hex, note). Save keeps the row; Export downloads a CSV
    matching labels_TEMPLATE.csv exactly.

Usage
    python view_pack_web.py <pack.parquet> [--start N] [--port 8000]
    python view_pack_web.py <pack.parquet> --out labels.csv   # server-side save
    python view_pack_web.py <pack.parquet> --out labels.csv --gold gold.parquet --handle you

Then open http://127.0.0.1:8000 in a browser.
"""
import argparse
import csv
import html
import io
import json
import os
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow.parquet as pq

import medicines
from detectors import scan_records

# ---------------------------------------------------------------------------
# Shared logic (identical to view_pack.py).
# ---------------------------------------------------------------------------
VISIBLE = {
    "\u200b": "\u27e6ZWSP\u27e7", "\u200c": "\u27e6ZWNJ\u27e7",
    "\u200d": "\u27e6ZWJ\u27e7", "\ufeff": "\u27e6BOM\u27e7",
    "\u00ad": "\u27e6SHY\u27e7", "\u202a": "\u27e6LRE\u27e7",
    "\u202b": "\u27e6RLE\u27e7", "\u202c": "\u27e6PDF\u27e7",
    "\u202d": "\u27e6LRO\u27e7", "\u202e": "\u27e6RLO\u27e7",
    "\u2066": "\u27e6LRI\u27e7", "\u2067": "\u27e6RLI\u27e7",
    "\u2068": "\u27e6FSI\u27e7", "\u2069": "\u27e6PDI\u27e7",
    "\ufffd": "\u27e6U+FFFD\u27e7", "\t": "\u27e6TAB\u27e7",
}


def token_for(ch: str):
    """Return the visible token for ch, or None if ch is shown as-is.

    Mirrors view_pack.py show(): mapped specials, then non-newline control
    chars as U+XXXX, else the character itself (including newline).
    """
    if ch in VISIBLE:
        return VISIBLE[ch]
    if ch != "\n" and unicodedata.category(ch) == "Cc":
        return f"\u27e6U+{ord(ch):04X}\u27e7"
    # Beyond view_pack.py: other invisible code points and look-alike spaces,
    # which would otherwise render as nothing or as a plain space.
    if unicodedata.category(ch) in ("Cf", "Co", "Cn", "Cs", "Zl", "Zp") or (
            unicodedata.category(ch) == "Zs" and ch != " "):
        return f"\u27e6U+{ord(ch):04X}\u27e7"
    return None


def show(text: str) -> str:
    """Identical to view_pack.py show()."""
    out = []
    for ch in text:
        tok = token_for(ch)
        out.append(tok if tok is not None else ch)
    return "".join(out)


def hexdump_rows(text: str, limit: int | None = None):
    """Same filter/format as view_pack.py hexdump(), as structured rows.

    Each row: {offset, cp (U+XXXX), name}. Offsets are code-point indices.
    """
    rows = []
    for i, ch in enumerate(text[:limit]):
        if ord(ch) > 127 or unicodedata.category(ch)[0] in "CZ":
            rows.append({
                "offset": i,
                "cp": f"U+{ord(ch):04X}",
                "name": unicodedata.name(ch, "?"),
            })
    return rows


def segments(text: str, breaks: frozenset = frozenset()):
    """Split text into display segments carrying raw code-point offsets.

    Each segment is one of:
      {"t": "text", "s": "...", "off": start_offset}            normal run
      {"t": "tok",  "s": "token", "raw": ch, "cp": "U+XXXX",
       "name": "...", "off": offset}                            invisible/ctrl

    A newline is emitted as its own text segment so the frontend can break
    lines while keeping exact offsets. Offsets are code-point indices into the
    raw text (what Python text[off] returns).
    """
    segs = []
    buf = []
    buf_start = 0
    for i, ch in enumerate(text):
        tok = token_for(ch)
        if buf and i in breaks:
            segs.append({"t": "text", "s": "".join(buf), "off": buf_start})
            buf = []
        if tok is None and ch != "\n":
            if not buf:
                buf_start = i
            buf.append(ch)
            continue
        # flush any pending normal run
        if buf:
            segs.append({"t": "text", "s": "".join(buf), "off": buf_start})
            buf = []
        if ch == "\n":
            segs.append({"t": "text", "s": "\n", "off": i})
        else:
            segs.append({
                "t": "tok", "s": tok, "raw": ch, "off": i,
                "cp": f"U+{ord(ch):04X}",
                "name": unicodedata.name(ch, "?"),
            })
    if buf:
        segs.append({"t": "text", "s": "".join(buf), "off": buf_start})
    return segs


# ---------------------------------------------------------------------------
# Server.
# ---------------------------------------------------------------------------
class State:
    docs = []          # list of row dicts
    scan = []           # detectors.scan_records() result, one per doc
    gold = None         # gold-set records for the harm test (--gold)
    handle = "<your GitHub handle>"
    out_path = None    # CSV save path (default labels.csv in CWD)
    doc_index = {}     # doc_id -> pack position, for stable CSV ordering


def doc_payload(i: int):
    d = State.docs[i]
    text = d.get("text", "") or ""
    scan = State.scan[i]
    findings = [dict(f.to_dict(), id=f"{f.code}-{n}") for n, f in enumerate(scan["findings"])]
    context = scan["context"]
    spans = [(f["start"], f["end"]) for f in findings if f["start"] >= 0]
    spans += [(r["start"], r["end"]) for r in context.get("ranges", [])]
    breaks = frozenset(x for a, b in spans for x in (a, b))
    return {
        "i": i,
        "count": len(State.docs),
        "doc_id": d.get("doc_id", ""),
        "source": d.get("source", ""),
        "lang_label": d.get("lang_label", ""),
        "url": d.get("url", "") or "",
        "chars": len(text),
        "raw_text": text,
        "segments": segments(text, breaks),
        "findings": findings,
        "suggested_verdict": scan["suggested_verdict"],
        "context": context,
        "checks": scan["checks"],
        "why": scan["why"],
        "hexdump": hexdump_rows(text),
    }


FIELDNAMES = ["doc_id", "read", "verdict", "diseases", "evidence", "hex", "note"]


def rows_to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDNAMES, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (r.get(k, "") or "") for k in FIELDNAMES})
    return buf.getvalue()


def read_csv_rows(path):
    """Read labels.csv into {doc_id: rowdict}. Missing/empty file -> {}."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            did = (row.get("doc_id") or "").strip()
            if did:
                out[did] = {k: (row.get(k, "") or "") for k in FIELDNAMES}
    return out


def ordered_rows(rowmap):
    """Rows sorted by pack position; unknown doc_ids go last, stable."""
    def key(did):
        return (0, State.doc_index[did]) if did in State.doc_index else (1, 0)
    return [rowmap[d] for d in sorted(rowmap, key=lambda d: key(d))]


def merge_save(incoming):
    """Upsert incoming rows into the on-disk CSV by doc_id; never delete.

    Re-reads the current file so rows the browser never touched are preserved.
    Returns (abs_path, total_rows).
    """
    disk = read_csv_rows(State.out_path)
    for r in incoming:
        did = (r.get("doc_id") or "").strip()
        if did:
            disk[did] = {k: (r.get(k, "") or "") for k in FIELDNAMES}
    csv_text = rows_to_csv(ordered_rows(disk))
    with open(State.out_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    return os.path.abspath(State.out_path), len(disk)


def medicine_preview(i: int, codes: str):
    """Run the medicine of each code on document i; return action, rule, effect and recorded notes."""
    text = State.docs[i].get("text", "") or ""
    findings = State.scan[i]["findings"]
    doc_id = State.docs[i].get("doc_id", "")
    notes = [n for n in medicines.read_notes(State.out_path) if n.get("doc_id") == doc_id]
    out = []
    for code in dict.fromkeys(c.strip().upper() for c in codes.split(";") if c.strip()):
        med = medicines.MEDICINES.get(code)
        if not med:
            out.append({"code": code, "action": "", "words": "no automatic medicine: write the rule by hand",
                        "status": "none", "diff": []})
            continue
        res = medicines.apply(med, text, findings, i + 1)
        out.append({"code": code, "action": med.action, "words": med.words, "status": res.status,
                    "diff": res.diff, "side_effects": list(med.side_effects),
                    "notes": [n["note"] for n in notes if n.get("code") == code]})
    return {"doc_id": doc_id, "medicines": out, "gold": bool(State.gold)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter console
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = dict(
            (k, v) for k, _, v in (p.partition("=") for p in query.split("&") if p)
        )
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/count":
            self._json({"count": len(State.docs)})
        elif path == "/api/rows":
            # current on-disk labels.csv, keyed by doc_id (source of truth)
            self._json({"rows": read_csv_rows(State.out_path)})
        elif path == "/api/summary":
            self._json({"docs": [{
                "i": r["index"], "doc_id": r["doc_id"], "suggested": r["suggested_verdict"],
                "codes": [f"{f.code}/{f.confidence}" for f in r["findings"]],
            } for r in State.scan]})
        elif path == "/api/medicine":
            try:
                i = int(params.get("i", "0"))
            except ValueError:
                return self._json({"error": "bad index"}, 400)
            if not (0 <= i < len(State.docs)):
                return self._json({"error": "out of range"}, 404)
            self._json(medicine_preview(i, params.get("codes", "")))
        elif path == "/api/doc":
            try:
                i = int(params.get("i", "0"))
            except ValueError:
                return self._json({"error": "bad index"}, 400)
            if not (0 <= i < len(State.docs)):
                return self._json({"error": "out of range"}, 404)
            self._json(doc_payload(i))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path, _, _ = self.path.partition("?")
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        if path in ("/api/save", "/api/export"):
            rows = data.get("rows", [])
            saved_path, total = merge_save(rows)
            self._json({"saved": saved_path, "rows": total})
        elif path == "/api/medicine_note":
            note = {k: str(data.get(k, "")).strip() for k in ("doc_id", "code", "note")}
            if not all(note.values()):
                return self._json({"error": "doc_id, code and note are required"}, 400)
            notes = medicines.read_notes(State.out_path)
            notes.append(note)
            with open(medicines.notes_path(State.out_path), "w", encoding="utf-8") as f:
                json.dump(notes, f, ensure_ascii=False, indent=1)
            self._json({"notes": len(notes)})
        elif path == "/api/medicine_sheet":
            text, summary = medicines.build_sheet(
                State.docs, medicines.read_labels(State.out_path), State.gold,
                medicines.read_notes(State.out_path), data.get("handle") or State.handle)
            out = os.path.join(os.path.dirname(os.path.abspath(State.out_path)), "medicine.md")
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            self._json({"saved": out, "summary": summary})
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--out", default="labels.csv",
                    help="CSV save path (default: labels.csv in the current directory)")
    ap.add_argument("--gold", help="gold-set Parquet; enables the harm test in the medicine sheet")
    ap.add_argument("--handle", default="<your GitHub handle>", help="your GitHub handle for the medicine sheet")
    a = ap.parse_args()
    State.gold = pq.read_table(a.gold).to_pylist() if a.gold else None
    State.handle = a.handle

    State.docs = pq.read_table(a.pack).to_pylist()
    State.out_path = a.out
    print(f"Scanning {len(State.docs)} documents with detectors.py ...")
    State.scan = scan_records(State.docs)
    State.doc_index = {d.get("doc_id", ""): i for i, d in enumerate(State.docs)}
    start = max(0, min(a.start, len(State.docs) - 1)) if State.docs else 0

    global PAGE
    PAGE = PAGE.replace("__START__", str(start))

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    shown_host = "127.0.0.1" if a.host == "0.0.0.0" else a.host
    url = f"http://{shown_host}:{a.port}"
    print(f"Loaded {len(State.docs)} documents from {a.pack}")
    print(f"Serving on {a.host}:{a.port}  ->  open {url}  (Ctrl-C to stop)")
    if State.out_path:
        print(f"Save button writes: {os.path.abspath(State.out_path)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


# ---------------------------------------------------------------------------
# Frontend (single inlined page, offline, no external assets).
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pack viewer</title>
<style>
  :root { --bg:#1e1e1e; --panel:#252526; --fg:#d4d4d4; --muted:#888;
          --accent:#4ec9b0; --tok:#f0a500; --tokbg:#4a3a00; --border:#3a3a3a;
          --right-w:520px; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg);
         color:var(--fg); font-size:14px; }
  header { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
           padding:8px 12px; background:var(--panel);
           border-bottom:1px solid var(--border); position:sticky; top:0; z-index:5; }
  header .meta { color:var(--muted); font-size:12px; }
  button, input, select, textarea { font-family:inherit; font-size:13px; }
  button { background:#333; color:var(--fg); border:1px solid var(--border);
           border-radius:4px; padding:4px 10px; cursor:pointer; }
  button:hover { background:#3d3d3d; }
  button.active { background:var(--accent); color:#000; }
  .wrap { display:flex; height:calc(100vh - 100px); min-height:0; }
  .left { flex:1 1 auto; min-width:200px; overflow:auto; padding:12px; }
  .splitter { flex:0 0 8px; cursor:col-resize; position:relative;
              background:var(--panel); touch-action:none; }
  .splitter::before { content:""; position:absolute; top:0; bottom:0; left:3px;
                      width:2px; background:var(--border); }
  .splitter:hover::before, .splitter.dragging::before { left:2px; width:4px;
                      background:var(--accent); }
  .right { flex:0 0 var(--right-w); width:var(--right-w); min-width:280px;
           overflow:auto; padding:12px; background:var(--panel); }
  #diag { white-space:pre-wrap; word-break:break-word; line-height:1.55;
          font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
          font-size:13px; }
  .tok { color:var(--tok); background:var(--tokbg); border-radius:3px;
         padding:0 2px; cursor:help; }
  #md { line-height:1.6; max-width:52em; }
  #md pre { background:#111; padding:10px; overflow:auto; border-radius:4px; }
  #md code { background:#111; padding:1px 4px; border-radius:3px; }
  #md table { border-collapse:collapse; } #md td,#md th { border:1px solid var(--border); padding:4px 8px; }
  #md blockquote { border-left:3px solid var(--border); margin:0; padding-left:10px; color:var(--muted); }
  footer { position:sticky; bottom:0; background:var(--panel);
           border-top:1px solid var(--border); padding:6px 12px; font-size:12px;
           display:flex; gap:16px; align-items:center; z-index:5; }
  .status b { color:var(--accent); }
  .hexrow { font-family:monospace; font-size:12px; white-space:pre; }
  .field { margin-bottom:8px; }
  .field label { display:block; color:var(--muted); font-size:11px;
                 text-transform:uppercase; letter-spacing:.04em; margin-bottom:2px; }
  .field input, .field textarea, .field select { width:100%; background:#1b1b1b;
     color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:5px; }
  .field textarea { resize:vertical; min-height:38px; }
  .row-actions { display:flex; gap:8px; margin:8px 0; }
  #dxPicked { width:100%; background:#1b1b1b; color:var(--accent); border:1px solid var(--border);
              border-radius:4px; padding:5px; font-family:monospace; }
  .dxsections { border:1px solid var(--border); border-radius:4px; margin-top:4px; }
  details.dxsec { border-bottom:1px solid var(--border); }
  details.dxsec:last-child { border-bottom:none; }
  details.dxsec > summary { cursor:pointer; padding:5px 8px; font-size:12px; color:var(--fg);
                            list-style:none; user-select:none; background:#1b1b1b; }
  details.dxsec > summary::-webkit-details-marker { display:none; }
  details.dxsec > summary:hover { background:#262626; }
  details.dxsec[open] > summary { color:var(--accent); }
  .dxitems { padding:4px 8px 6px; }
  .dxitem { display:flex; align-items:flex-start; gap:6px; padding:2px 0; font-size:12px; }
  .dxitem input { width:auto; margin-top:2px; }
  .dxitem .code { color:var(--tok); font-family:monospace; min-width:26px; }
  .dxitem .desc { color:var(--fg); }
  .dxitem .desc small { color:var(--muted); }
  h3 { margin:14px 0 6px; font-size:12px; color:var(--muted);
       text-transform:uppercase; letter-spacing:.05em; }
  .hint { color:var(--muted); font-size:11px; margin-top:2px; }
  .warn { color:#e0a0a0; font-size:11px; }
  #exportBox { width:100%; height:120px; background:#111; color:var(--fg);
               border:1px solid var(--border); font-family:monospace; font-size:11px; }
  .saved { color:var(--accent); }
  #dxPop { position:fixed; z-index:30; width:320px; max-height:360px;
           display:flex; flex-direction:column; gap:6px; padding:8px;
           background:var(--panel); border:1px solid var(--accent); border-radius:6px;
           box-shadow:0 8px 24px rgba(0,0,0,.45); }
  #dxPop[hidden] { display:none; }
  #dxPop .dxpop-head { font-size:12px; color:var(--fg); }
  #dxPop .dxpop-where { color:var(--accent); font-family:monospace; }
  #dxPopSearch { width:100%; background:#1b1b1b; color:var(--fg);
                 border:1px solid var(--border); border-radius:4px; padding:5px; }
  #dxPopList { overflow:auto; min-height:0; }
  .dxpop-sec { color:var(--muted); font-size:11px; text-transform:uppercase;
               letter-spacing:.04em; margin:6px 2px 2px; }
  .dxpop-item { display:block; width:100%; text-align:left; background:transparent;
                color:var(--fg); border:none; border-radius:4px; padding:4px 6px; cursor:pointer; }
  .dxpop-item:hover { background:#333; }
  .dxpop-item .code { color:var(--tok); font-family:monospace; margin-right:6px; }
/* ---- detected findings ---- */
.hl-high { background:rgba(241,76,76,.35); border-bottom:2px solid #f14c4c; cursor:pointer; }
.hl-medium { background:rgba(240,165,0,.28); border-bottom:2px solid #f0a500; cursor:pointer; }
.hl-low { background:rgba(136,136,136,.25); border-bottom:1px dashed #888; cursor:pointer; }
.hl-active { outline:2px solid var(--accent); }
#fxList { margin-bottom:10px; }
.fx { border:1px solid var(--border); border-left:4px solid #888; border-radius:4px; padding:6px 8px;
      margin-bottom:6px; background:#1b1b1b; font-size:12px; }
.fx.c-high { border-left-color:#f14c4c; } .fx.c-medium { border-left-color:#f0a500; }
.fx.st-a { background:#14281f; } .fx.st-r { opacity:.45; }
.fx.flash { outline:2px solid var(--accent); }
.fx-head { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.fx-head .code { color:var(--tok); font-family:monospace; font-weight:bold; }
.conf { font-size:10px; padding:0 5px; border-radius:8px; text-transform:uppercase; color:#000; }
.conf-high { background:#f14c4c; } .conf-medium { background:#f0a500; } .conf-low { background:#888; }
.fx-where { color:var(--muted); font-family:monospace; }
.fx-btns { margin-left:auto; display:flex; gap:4px; }
.fx-btns button { padding:1px 7px; font-size:11px; }
.fx-reason { margin-top:4px; color:var(--fg); line-height:1.4; }
.fx-ev { margin-top:3px; color:var(--muted); font-family:monospace; word-break:break-word; }
.fx-state { font-size:10px; color:var(--accent); }
.sugg { font-size:12px; padding:1px 8px; border-radius:8px; background:#333; color:var(--fg); font-weight:normal; }
.df-add { background:rgba(78,201,120,.18); }
.df-del { background:rgba(241,76,76,.14); text-decoration:line-through rgba(241,76,76,.5); }
.ctx { border:1px solid var(--border); border-radius:4px; padding:6px 8px; background:#1b1b1b; font-size:12px;
       line-height:1.45; margin-bottom:8px; }
.ctx b { color:var(--accent); }
.ctx .stat { font-family:monospace; } .ctx .plus { color:#4ec978; } .ctx .minus { color:#f14c4c; }
.why { font-size:12px; line-height:1.45; margin:4px 0 8px; color:var(--fg); }
details.checks > summary { cursor:pointer; color:var(--muted); font-size:12px; margin-bottom:4px; }
table.ck { width:100%; border-collapse:collapse; font-size:11px; }
table.ck td { border-top:1px solid var(--border); padding:3px 4px; vertical-align:top; }
table.ck td.code { font-family:monospace; color:var(--tok); white-space:nowrap; }
.ck-st { white-space:nowrap; font-weight:bold; }
.ck-hit { color:#f14c4c; } .ck-weak { color:#f0a500; } .ck-pass { color:#4ec978; }
.ck-na { color:var(--muted); } .ck-manual { color:#6cb6ff; }
.ck-how { color:var(--muted); }
tr.ck-row-pass .ck-how, tr.ck-row-na .ck-how { opacity:.8; }
.med { border:1px solid var(--border); border-radius:4px; padding:6px 8px; margin-bottom:6px; background:#1b1b1b;
       font-size:12px; }
.med-head { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.med-head .code { color:var(--tok); font-family:monospace; font-weight:bold; }
.med-st { font-size:10px; padding:0 6px; border-radius:8px; text-transform:uppercase; }
.med-changed { background:#4a3a00; } .med-dropped { background:#5a1d1d; } .med-unchanged { background:#333; }
.med-none { background:#333; color:var(--muted); }
.med-words { margin-top:3px; color:var(--muted); line-height:1.4; }
.med pre { margin:4px 0 0; padding:4px; background:#111; border-radius:3px; font-size:11px; max-height:180px;
           overflow:auto; white-space:pre-wrap; word-break:break-all; }
.med pre .a { color:#4ec978; } .med pre .d { color:#f14c4c; } .med pre .h { color:var(--muted); }
.med-note { margin-top:3px; color:#f0a500; }
.med-head button { margin-left:auto; padding:1px 7px; font-size:11px; }
.sugg.drop { background:#5a1d1d; } .sugg.fix { background:#4a3a00; } .sugg.clean { background:#14281f; }
</style>
</head>
<body>
<header>
  <button id="prev">\u25c0 back</button>
  <button id="next">next \u25b6</button>
  <span>#<input id="jump" type="number" min="1" style="width:70px"> </span>
  <input id="jumpId" placeholder="jump to doc_id" style="width:160px">
  <button id="goId">go</button>
  <button id="toggleMd">markdown preview</button>
  <button id="toggleHex" class="active">hex dump</button>
  <select id="filter" title="which documents next/prev match visits"></select>
  <button id="fPrev" title="previous matching document (p)">\u25c0 match</button>
  <button id="fNext" title="next matching document (n)">match \u25b6</button>
  <span class="meta" id="fInfo"></span>
  <span class="meta" id="meta"></span>
</header>

<div class="wrap">
  <div class="left">
    <div id="diag"></div>
    <div id="md" style="display:none"></div>
  </div>
  <div class="splitter" id="splitter" title="Drag to resize the annotation pane. Double-click to reset."></div>
  <div class="right">
    <h3 id="ctxHead">Commit context</h3>
    <div id="ctx" class="ctx"></div>

    <h3>Detected <span id="fxCount"></span> <span id="sugg" class="sugg"></span></h3>
    <div id="why" class="why"></div>
    <div class="row-actions">
      <button id="fxAcceptHigh" title="accept every high-confidence finding">accept all high</button>
      <button id="fxUseVerdict" title="copy the suggested verdict into the verdict field">use suggested verdict</button>
      <label class="hint"><input type="checkbox" id="fxShowLow" checked> show low</label>
    </div>
    <div id="fxList"></div>
    <div class="hint">Hints only. Click a highlight on the left to find its card. Read the whole document before
      you mark it; "clean" is your claim.</div>
    <details class="checks" id="ckBox"><summary id="ckSum">what was checked</summary>
      <div class="row-actions"><label class="hint"><input type="checkbox" id="ckOnlyOpen"> hide passed / n/a</label></div>
      <table class="ck" id="ckTable"></table></details>

    <h3>Annotation (this document)</h3>
    <div class="field"><label>doc_id</label><input id="f_doc_id" readonly></div>
    <div class="field"><label>read</label>
      <select id="f_read"><option value=""></option><option>full</option><option>partial</option></select></div>
    <div class="field"><label>verdict</label>
      <select id="f_verdict"><option value=""></option><option>clean</option><option>fix</option><option>drop</option></select></div>
    <div class="field"><label>diseases (tick below &rarr; e.g. A1;B3)</label>
      <input id="f_diseases" placeholder="codes appear here; you can also type">
      <div class="dxsections" id="dxSections"></div></div>
    <div class="field"><label>evidence</label><textarea id="f_evidence"></textarea>
      <div class="hint">Select text on the left, then pick a disease. The code is added above and the text is appended here with its line number.</div></div>
    <div class="field"><label>hex (U+XXXX ...)</label><input id="f_hex">
      <div class="warn" id="hexWarn" style="display:none">F1/E3: leave hex empty. Never paste the value.</div></div>
    <div class="field"><label>note</label><input id="f_note"></div>
    <div class="field"><label>medicine (what the rule for each disease above does to this document)</label>
      <div id="medList"><div class="hint">add a disease code to see its medicine</div></div>
      <div class="hint">"unchanged" on a document you marked fix = the medicine misses it (a side effect to note
        or a sign the rule needs work). Notes go to the side-effects part of the medicine sheet.</div></div>
    <div class="row-actions">
      <button id="clearRow">clear this row (UI only)</button>
      <span id="rowState" class="hint"></span>
    </div>

    <h3>Save (<span id="savedCount">0</span> rows in file)</h3>
    <div class="row-actions">
      <button id="exportBtn">save labels.csv</button>
      <button id="medSheetBtn" title="save labels, then write medicine.md next to the labels file">write medicine sheet</button>
    </div>
    <div class="hint" id="medSheetMsg"></div>
    <textarea id="exportBox" readonly placeholder="current row (CSV) appears here"></textarea>
    <div class="hint" id="exportMsg"></div>

    <h3 id="hexHead">Hex dump (non-ASCII / control / separator)</h3>
    <div id="hexList"></div>
  </div>
</div>

<footer>
  <span class="status" id="status">offset: \u2013</span>
  <span class="status" id="selstatus"></span>
</footer>
<div id="dxPop" hidden>
  <div class="dxpop-head">Tag this selection as a disease <span class="dxpop-where" id="dxPopWhere"></span></div>
  <input id="dxPopSearch" placeholder="filter diseases" autocomplete="off">
  <div id="dxPopList"></div>
</div>

<script>
// ---- tiny markdown renderer (bundled JS, no deps) ----
function escapeHtml(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function mdInline(s){
  s = escapeHtml(s);
  s = s.replace(/`([^`]+)`/g,'<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*]+)\*/g,'$1<em>$2</em>');
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
function renderMarkdown(text){
  const lines = text.split('\n');
  let out = [], i = 0;
  while(i < lines.length){
    let line = lines[i];
    if(/^```/.test(line)){
      let code=[]; i++;
      while(i<lines.length && !/^```/.test(lines[i])){ code.push(lines[i]); i++; }
      i++;
      out.push('<pre><code>'+escapeHtml(code.join('\n'))+'</code></pre>'); continue;
    }
    let h = line.match(/^(#{1,6})\s+(.*)$/);
    if(h){ const n=h[1].length; out.push('<h'+n+'>'+mdInline(h[2])+'</h'+n+'>'); i++; continue; }
    if(/^\s*>/.test(line)){
      let q=[]; while(i<lines.length && /^\s*>/.test(lines[i])){ q.push(lines[i].replace(/^\s*>\s?/,'')); i++; }
      out.push('<blockquote>'+mdInline(q.join('<br>'))+'</blockquote>'); continue;
    }
    if(/^\s*([-*+]|\d+\.)\s+/.test(line)){
      const ordered=/^\s*\d+\./.test(line); let items=[];
      while(i<lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])){
        items.push('<li>'+mdInline(lines[i].replace(/^\s*([-*+]|\d+\.)\s+/,''))+'</li>'); i++; }
      out.push((ordered?'<ol>':'<ul>')+items.join('')+(ordered?'</ol>':'</ul>')); continue;
    }
    if(/^\s*\|.*\|\s*$/.test(line) && i+1<lines.length && /^\s*\|[-:\s|]+\|\s*$/.test(lines[i+1])){
      const parse=r=>r.trim().replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
      const head=parse(line); i+=2; let body=[];
      while(i<lines.length && /^\s*\|.*\|\s*$/.test(lines[i])){ body.push(parse(lines[i])); i++; }
      let t='<table><thead><tr>'+head.map(c=>'<th>'+mdInline(c)+'</th>').join('')+'</tr></thead><tbody>';
      t+=body.map(r=>'<tr>'+r.map(c=>'<td>'+mdInline(c)+'</td>').join('')+'</tr>').join('');
      out.push(t+'</tbody></table>'); continue;
    }
    if(line.trim()===''){ i++; continue; }
    let para=[]; while(i<lines.length && lines[i].trim()!=='' && !/^(#{1,6}\s|```|\s*>|\s*([-*+]|\d+\.)\s)/.test(lines[i])){ para.push(lines[i]); i++; }
    out.push('<p>'+mdInline(para.join(' '))+'</p>');
  }
  return out.join('\n');
}

// ---- state ----
let cur = null;           // current doc payload
let idx = parseInt("__START__",10) || 0;
const saved = {};         // doc_id -> row object (persisted in localStorage)
const LS_KEY = 'view_pack_web_rows';
// NOTE: seeding happens in bootstrap(): CSV (source of truth) first,
// then localStorage overlaid on top as the unsaved-edit buffer.

const $ = id => document.getElementById(id);

// ---- disease catalog (verbatim from DISEASES.md) ----
const DISEASES = [
 ["A","Characters",[
   ["A1","invisible junk characters","ZWSP U+200B, BOM U+FEFF, bidi marks, soft hyphen (ZWJ/ZWNJ are NOT diseases in Indic)"],
   ["A2","control characters","U+0000-U+001F except tab/newline, U+007F-U+009F"],
   ["A3","mojibake","UTF-8 decoded twice, e.g. Ã¤ à¤¹ â€™"],
   ["A4","replacement characters","U+FFFD \ufffd ; count them"],
   ["A5","wrong normal form","same word in two byte forms; nukta two ways (we use NFC)"],
   ["A6","wrong or mixed script","Hindi with Latin look-alikes; Bengali word in Assamese letters"]]],
 ["B","Markup and ghost tokens",[
   ["B1","HTML entities","&amp; &nbsp; &#2361;"],
   ["B2","HTML or XML tags","<div> <br/> </p>"],
   ["B3","ghost chat markers","<|im_start|> <|endoftext|> [INST] <s> ### Assistant: <think> (most dangerous)"],
   ["B4","template/code junk in prose","{{ variable }} ${...} undefined NaN"],
   ["B5","broken markdown or LaTeX","unclosed code fence, \\frac{ with no end"]]],
 ["C","Document structure",[
   ["C1","boilerplate","menus, cookie banners, Share on WhatsApp, footer links"],
   ["C2","repeated lines","same line many times in one document"],
   ["C3","truncated document","text stops mid-sentence or mid-code-block"],
   ["C4","two documents joined","two unrelated texts, no boundary"],
   ["C5","lists of keywords or links","SEO spam, tag clouds"]]],
 ["D","Language",[
   ["D1","wrong language label","label says Marathi, text is Hindi"],
   ["D2","romanised Indic","Hindi written in Latin letters"],
   ["D3","mixed language","Hindi + English in one sentence (can be natural; label, do not judge)"],
   ["D4","bad machine translation","wrong meaning, broken grammar, untranslated words"],
   ["D5","translationese","grammatical, but no native speaker writes like this"]]],
 ["E","Code",[
   ["E1","minified or generated code","one very long line; auto-generated do not edit"],
   ["E2","data blobs","base64, long number arrays, embedded images"],
   ["E3","secrets","API keys, passwords, tokens, private keys (hex empty, position only)"],
   ["E4","repeated licence headers","same 30-line licence at top of many files"],
   ["E5","broken encoding in code","mojibake inside strings or comments"],
   ["E6","not code","README, log file or CSV labelled as code"]]],
 ["F","Safety, privacy and contamination",[
   ["F1","personal data","emails, phones, addresses, IDs of private people (hex empty, position only)"],
   ["F2","benchmark leak","a question/solution from a benchmark test set"],
   ["F3","harmful content","content we must not train on"]]],
 ["G","Damage from earlier cleaning",[
   ["G1","cut at the vowel sign","word split before a matra or virama, e.g. \u0915 \u093f"],
   ["G2","lost ZWJ or ZWNJ","a conjunct that renders wrong"],
   ["G3","Indic dropped by an English rule","good doc with low quality score; sections missing"],
   ["G4","over-normalised","NFKC changed chars; Indic digits changed to Latin"]]],
 ["H","Duplicates",[
   ["H1","exact duplicate in the pack",""],
   ["H2","near duplicate","same text with small changes"],
   ["H3","same content in another source or language","translation of a doc seen in English"]]],
];
// canonical order index for sorting the picked codes
const DX_ORDER = {}; let _k=0;
DISEASES.forEach(([,,items])=>items.forEach(([code])=>{ DX_ORDER[code]=_k++; }));

function buildDiseasePicker(){
  const box = $('dxSections'); box.innerHTML='';
  for(const [sec, title, items] of DISEASES){
    const det=document.createElement('details'); det.className='dxsec';
    const sum=document.createElement('summary'); sum.textContent=sec+'. '+title;
    det.appendChild(sum);
    const wrap=document.createElement('div'); wrap.className='dxitems';
    for(const [code, name, how] of items){
      const row=document.createElement('label'); row.className='dxitem';
      const cb=document.createElement('input'); cb.type='checkbox'; cb.value=code;
      cb.dataset.dx='1'; cb.addEventListener('change', onDxCheck);
      const desc=document.createElement('span'); desc.className='desc';
      desc.innerHTML='<span class="code">'+code+'</span> '+name + (how?' <small>\u2014 '+how.replace(/</g,'&lt;')+'</small>':'');
      row.appendChild(cb); row.appendChild(desc);
      if(how) row.title=how;
      wrap.appendChild(row);
    }
    det.appendChild(wrap); box.appendChild(det);
  }
}
function dxCheckboxes(){ return Array.from($('dxSections').querySelectorAll('input[data-dx]')); }
// checkboxes -> diseases string (canonical order), preserving any typed-in unknown codes
function onDxCheck(ev){
  const ticked = dxCheckboxes().filter(c=>c.checked).map(c=>c.value);
  const known = new Set(Object.keys(DX_ORDER));
  const existing = ($('f_diseases').value||'').split(';').map(s=>s.trim()).filter(Boolean);
  const unknown = existing.filter(c=>!known.has(c.toUpperCase()));
  const all = [...new Set([...ticked, ...unknown])];
  all.sort((a,b)=>(DX_ORDER[a]??999)-(DX_ORDER[b]??999));
  $('f_diseases').value = all.join(';');
  if(ev && ev.target && ev.target.dataset && ev.target.dataset.dx && ev.target.checked && pendingSel){
    appendTaggedEvidence(ev.target.value);
  }
  updateHexWarn();
  renderRowPreview();
}
// diseases string -> checkbox states (for load / manual typing)
function syncDiseaseChecks(){
  const codes = new Set(($('f_diseases').value||'').toUpperCase().split(';').map(s=>s.trim()).filter(Boolean));
  dxCheckboxes().forEach(c=>{ c.checked = codes.has(c.value); });
}

function cpLen(s){ return Array.from(s).length; }           // code-point length
function cpAt(s, n){ return Array.from(s)[n]; }

// Build diagnostic DOM; each node carries data-off (code-point offset of its
// first char) and data-len (code-point length of its raw content).
function renderDiag(payload){
  const diag = $('diag'); diag.innerHTML='';
  for(const seg of payload.segments){
    if(seg.t==='text'){
      if(seg.s==='\n'){ diag.appendChild(document.createTextNode('\n')); continue; }
      const span=document.createElement('span');
      span.textContent=seg.s; span.dataset.off=seg.off; span.dataset.len=cpLen(seg.s);
      span.dataset.kind='text'; diag.appendChild(span);
    } else {
      const span=document.createElement('span');
      span.className='tok'; span.textContent=seg.s;
      span.dataset.off=seg.off; span.dataset.len=1; span.dataset.kind='tok';
      span.dataset.raw=seg.raw; span.dataset.cp=seg.cp; span.dataset.name=seg.name;
      span.title=seg.cp+' '+seg.name+'  (offset '+seg.off+')';
      diag.appendChild(span);
    }
  }
}

// Map a DOM (node, domOffset) to a raw code-point offset in the text.
function rawOffsetOf(node, domOffset){
  let span = node.nodeType===3 ? node.parentElement : node;
  if(!span || span.id==='diag'){
    // cursor between spans: approximate via previous element
    const el = node.childNodes && node.childNodes[domOffset-1];
    if(el && el.dataset) return parseInt(el.dataset.off)+parseInt(el.dataset.len);
    return null;
  }
  if(span.dataset.off===undefined) return null;
  const base = parseInt(span.dataset.off);
  if(span.dataset.kind==='tok') return base + (domOffset>0?1:0);
  // text span: domOffset counts UTF-16 units; convert to code points
  const txt = span.textContent.slice(0, domOffset);
  return base + cpLen(txt);
}

function describeOffset(off){
  if(off===null || off===undefined || !cur) return 'offset: \u2013';
  const ch = cpAt(cur.raw_text, off);
  if(ch===undefined) return 'offset: '+off+' (end)';
  const cp = ch.codePointAt(0);
  const hex = 'U+'+cp.toString(16).toUpperCase().padStart(4,'0');
  let name=''; // name not shipped per-char; show cp. Tokens carry names via title.
  return 'offset: <b>'+off+'</b> \u00b7 '+hex;
}

const VISIBLE_JS = {
  '\u200b':'\u27e6ZWSP\u27e7','\u200c':'\u27e6ZWNJ\u27e7','\u200d':'\u27e6ZWJ\u27e7','\ufeff':'\u27e6BOM\u27e7',
  '\u00ad':'\u27e6SHY\u27e7','\u202a':'\u27e6LRE\u27e7','\u202b':'\u27e6RLE\u27e7','\u202c':'\u27e6PDF\u27e7',
  '\u202d':'\u27e6LRO\u27e7','\u202e':'\u27e6RLO\u27e7','\u2066':'\u27e6LRI\u27e7','\u2067':'\u27e6RLI\u27e7',
  '\u2068':'\u27e6FSI\u27e7','\u2069':'\u27e6PDI\u27e7','\ufffd':'\u27e6U+FFFD\u27e7','\t':'\u27e6TAB\u27e7'
};
let pendingSel = null;   // last text selection in the diagnostic pane
let lastPointer = {x:80, y:120};

function visibleChar(c){
  if(VISIBLE_JS[c]) return VISIBLE_JS[c];
  const n = c.codePointAt(0);
  if(c!=='\n' && (n<32 || n===127)) return '\u27e6U+'+n.toString(16).toUpperCase().padStart(4,'0')+'\u27e7';
  return c;
}
function lineAt(off){
  if(!cur) return 1;
  const arr = Array.from(cur.raw_text);
  let line = 1;
  const n = Math.max(0, Math.min(off, arr.length));
  for(let i=0;i<n;i++) if(arr[i]==='\n') line++;
  return line;
}
function lineLabel(sel){
  return sel.startLine===sel.endLine ? ('line '+sel.startLine) : ('lines '+sel.startLine+'-'+sel.endLine);
}
function plainEvidence(sel){
  return (sel.vis || '(empty selection)')+' at '+lineLabel(sel);
}
// Text after "C2: " for codes that need more than the selected span.
// Add a function and a line in EVIDENCE_RULES to customise another code.
function positionOnlyEvidence(sel){
  return 'at '+lineLabel(sel);
}
function repeatedLineEvidence(sel){
  const lines = ((cur && cur.raw_text) || '').split('\n');
  const bits = [];
  const seen = new Set();
  for(let n=sel.startLine; n<=sel.endLine; n++){
    const text = (lines[n-1] || '').trim();
    if(!text || seen.has(text)) continue;
    seen.add(text);
    let count = 0;
    for(let i=0;i<lines.length;i++) if(lines[i].trim()===text) count++;
    bits.push(text+' at line '+n+' ('+count+(count===1?' time':' times')+')');
  }
  return bits.length ? bits.join('; ') : plainEvidence(sel);
}
const EVIDENCE_RULES = {
  C2: repeatedLineEvidence,   // same line many times in one document
  F1: positionOnlyEvidence,   // personal data: line only, never the value
  E3: positionOnlyEvidence,   // secrets: line only, never the value
};
function evidenceSnippet(code, sel){
  const rule = EVIDENCE_RULES[code];
  return rule ? rule(sel) : plainEvidence(sel);
}
function selectionInsideDiag(sel){
  if(!sel || !sel.rangeCount) return false;
  const node = sel.anchorNode;
  const diag = $('diag');
  return !!(diag && node && diag.contains(node));
}
function hideDxPop(){
  $('dxPop').hidden = true;
}
function showDxPop(){
  if(!pendingSel) return;
  const pop = $('dxPop');
  pop.hidden = false;
  $('dxPopWhere').textContent = '\u00b7 '+lineLabel(pendingSel);
  renderDxPopList();
  const sel = window.getSelection();
  let rect = null;
  if(sel && sel.rangeCount){
    const r = sel.getRangeAt(0).getBoundingClientRect();
    if(r && (r.width || r.height)) rect = r;
  }
  const margin = 8, width = 320;
  const height = Math.min(360, pop.offsetHeight || 360);
  let left = rect ? rect.left : lastPointer.x;
  let top = rect ? rect.bottom + 6 : lastPointer.y + 12;
  if(left + width > window.innerWidth - margin) left = window.innerWidth - width - margin;
  if(top + height > window.innerHeight - margin) top = Math.max(margin, (rect ? rect.top : lastPointer.y) - height - 6);
  if(left < margin) left = margin;
  if(top < margin) top = margin;
  pop.style.left = left+'px';
  pop.style.top = top+'px';
}
function renderDxPopList(){
  const q = ($('dxPopSearch').value||'').trim().toLowerCase();
  const list = $('dxPopList');
  list.innerHTML = '';
  const picked = new Set(($('f_diseases').value||'').toUpperCase().split(';').map(s=>s.trim()).filter(Boolean));
  for(const [sec, title, items] of DISEASES){
    const matched = items.filter(([code, name, how])=>{
      if(!q) return true;
      return (code+' '+name+' '+(how||'')).toLowerCase().includes(q);
    });
    if(!matched.length) continue;
    const h = document.createElement('div');
    h.className = 'dxpop-sec';
    h.textContent = sec+'. '+title;
    list.appendChild(h);
    for(const [code, name] of matched){
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'dxpop-item';
      const codeEl = document.createElement('span');
      codeEl.className = 'code';
      codeEl.textContent = code;
      b.appendChild(codeEl);
      b.appendChild(document.createTextNode(name));
      if(picked.has(code)){
        const mark = document.createElement('span');
        mark.className = 'saved';
        mark.textContent = ' \u2713';
        b.appendChild(mark);
      }
      b.addEventListener('click', ()=> tagSelection(code));
      list.appendChild(b);
    }
  }
  if(!list.childNodes.length){
    const empty = document.createElement('div');
    empty.className = 'hint';
    empty.textContent = 'No diseases match.';
    list.appendChild(empty);
  }
}
function checkDisease(code){
  const cb = dxCheckboxes().find(c=>c.value===code);
  if(!cb) return;
  cb.checked = true;
  const det = cb.closest('details');
  if(det) det.open = true;
}
function appendTaggedEvidence(code){
  if(!pendingSel) return;
  const incoming = evidenceSnippet(code, pendingSel).split('; ').map(s=>s.trim()).filter(Boolean);
  if(!incoming.length) return;
  const box = $('f_evidence');
  const curVal = box.value || '';
  const plain = plainEvidence(pendingSel);
  const rows = (!curVal.trim() || curVal.trim()===plain) ? [] : curVal.replace(/\s+$/,'').split('\n');
  const prefix = code+': ';
  const idx = rows.findIndex(r=>r.startsWith(prefix));
  const entries = idx===-1 ? [] : rows[idx].slice(prefix.length).split('; ').map(s=>s.trim()).filter(Boolean);
  let changed = false;
  for(const bit of incoming){
    if(!entries.includes(bit)){ entries.push(bit); changed = true; }
  }
  if(!changed) return;
  const line = prefix+entries.join('; ');
  if(idx===-1) rows.push(line); else rows[idx] = line;
  box.value = rows.join('\n');
}
function tagSelection(code){
  if(!pendingSel) return;
  checkDisease(code);
  onDxCheck();
  appendTaggedEvidence(code);
  updateHexWarn();
  renderRowPreview();
  renderDxPopList();
  $('selstatus').textContent = 'added '+code+' \u00b7 '+lineLabel(pendingSel);
}

function onCursor(){
  const sel = window.getSelection();
  if(!sel.rangeCount){ return; }
  const r = sel.getRangeAt(0);
  const off = rawOffsetOf(r.startContainer, r.startOffset);
  $('status').innerHTML = describeOffset(off);

  if(!sel.isCollapsed && selectionInsideDiag(sel)){
    const startOff = rawOffsetOf(r.startContainer, r.startOffset);
    const endOff = rawOffsetOf(r.endContainer, r.endOffset);
    if(startOff!==null && endOff!==null && endOff>startOff && cur){
      const arr = Array.from(cur.raw_text).slice(startOff, endOff);
      const hex = arr.map(c=>'U+'+c.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')).join(' ');
      const vis = arr.map(visibleChar).join('').replace(/\s*\n\s*/g,' ').replace(/[ \t]{2,}/g,' ').trim();
      const startLine = lineAt(startOff);
      const endLine = lineAt(endOff-1);
      pendingSel = {vis, hex, startOff, endOff, startLine, endLine};
      window._selFill = { evidence: plainEvidence(pendingSel), hex: hex };
      $('selstatus').textContent = 'selected '+lineLabel(pendingSel)+' \u2014 pick a disease';
      return;
    }
  }
  if(sel.isCollapsed){
    if(pendingSel) $('selstatus').textContent = 'selected '+lineLabel(pendingSel)+' \u2014 pick a disease';
    else $('selstatus').textContent = '';
    const popOpen = !$('dxPop').hidden;
    if(!popOpen && !pendingSel && off!==null && off!==undefined){
      navigator.clipboard && navigator.clipboard.writeText(String(off)).then(()=>{
        $('status').innerHTML = describeOffset(off) + ' <span class="saved">\u2713 copied</span>';
      }).catch(()=>{});
    }
  }
}

function fillFields(docId){
  const row = saved[docId] || {};
  $('f_doc_id').value = docId;
  $('f_read').value = row.read||'';
  $('f_verdict').value = row.verdict||'';
  $('f_diseases').value = row.diseases||'';
  $('f_evidence').value = row.evidence||'';
  $('f_hex').value = row.hex||'';
  $('f_note').value = row.note||'';
  $('rowState').textContent = saved[docId] ? 'saved' : '';
  syncDiseaseChecks();
  updateHexWarn();
  renderRowPreview();
}

function updateHexWarn(){
  const dis=($('f_diseases').value||'').toUpperCase();
  $('hexWarn').style.display = (dis.includes('F1')||dis.includes('E3')) ? 'block':'none';
}

// CSV-quote one field to match Python csv (quote if it has , " or newline; double quotes)
function csvField(s){
  s = (s==null)?'':String(s);
  return /[",\n\r]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s;
}
const CSV_HEADER = 'doc_id,read,verdict,diseases,evidence,hex,note';
function currentRowValues(){
  return {
    doc_id: $('f_doc_id').value, read: $('f_read').value, verdict: $('f_verdict').value,
    diseases: $('f_diseases').value, evidence: $('f_evidence').value,
    hex: $('f_hex').value, note: $('f_note').value,
  };
}
// Live preview of the row being built, in the CSV box below the save button.
function renderRowPreview(){
  const r = currentRowValues();
  const line = ['doc_id','read','verdict','diseases','evidence','hex','note']
                 .map(k=>csvField(r[k])).join(',');
  $('exportBox').value = CSV_HEADER + '\n' + line;
  scheduleMedicine();
}

async function load(i){
  commitCurrent();            // keep in-progress edits when traversing
  const res = await fetch('/api/doc?i='+i);
  if(!res.ok){ return; }
  cur = await res.json(); idx = cur.i;
  $('meta').textContent = '['+(cur.i+1)+'/'+cur.count+']  doc_id='+cur.doc_id
     +'  source='+cur.source+'  lang='+cur.lang_label+'  chars='+cur.chars
     + (cur.url? '  '+cur.url : '');
  $('jump').value = cur.i+1;
  renderDiag(cur);
  // markdown
  $('md').innerHTML = renderMarkdown(cur.raw_text);
  // hex dump
  const hl=$('hexList'); hl.innerHTML = cur.hexdump.length
    ? cur.hexdump.map(h=>'<div class="hexrow">'+String(h.offset).padStart(6)+'  '+h.cp+'  '+escapeHtml(h.name)+'</div>').join('')
    : '<div class="hint">(no non-ASCII characters in this document)</div>';
  fillFields(cur.doc_id);
  highlightDiff(); highlightFindings(); renderContext(); renderFindings(); renderChecks(); updateFilterInfo();
  pendingSel = null; window._selFill = null; hideDxPop();
  $('status').innerHTML='offset: \u2013'; $('selstatus').textContent='';
}

// ---- events ----
$('prev').onclick = ()=> load(Math.max(0, idx-1));
$('next').onclick = ()=> load(Math.min((cur?cur.count:1)-1, idx+1));
$('jump').onchange = e => { const n=parseInt(e.target.value,10)-1; if(!isNaN(n)) load(n); };
$('goId').onclick = async ()=>{
  const id=$('jumpId').value.trim(); if(!id) return;
  const res=await fetch('/api/count'); const {count}=await res.json();
  for(let k=0;k<count;k++){ const r=await (await fetch('/api/doc?i='+k)).json(); if(r.doc_id===id){ load(k); return; } }
  alert('doc_id not found');
};
$('toggleMd').onclick = e => {
  const md=$('md'), dg=$('diag');
  const on = md.style.display==='none';
  md.style.display = on?'block':'none'; dg.style.display = on?'none':'block';
  e.target.classList.toggle('active', on);
};
$('toggleHex').onclick = e => {
  const on = $('hexList').style.display==='none';
  $('hexList').style.display = on?'block':'none';
  $('hexHead').style.display = on?'block':'none';
  e.target.classList.toggle('active', on);
};
function onDiagMouseUp(e){
  lastPointer = {x:e.clientX, y:e.clientY};
  onCursor();
  const sel = window.getSelection();
  if(pendingSel && sel && !sel.isCollapsed && selectionInsideDiag(sel)) showDxPop();
  else { pendingSel = null; window._selFill = null; hideDxPop(); $('selstatus').textContent=''; }
}
document.getElementById('diag').addEventListener('mouseup', onDiagMouseUp);
document.getElementById('diag').addEventListener('keyup', ()=>{
  onCursor();
  const sel = window.getSelection();
  if(sel && !sel.isCollapsed && selectionInsideDiag(sel) && pendingSel) showDxPop();
  else if(sel && sel.isCollapsed && sel.anchorNode && $('diag').contains(sel.anchorNode)){
    pendingSel = null; window._selFill = null; hideDxPop();
    $('selstatus').textContent = '';
  }
});
document.addEventListener('mouseup', e=>{
  if($('dxPop').contains(e.target) || $('diag').contains(e.target)) return;
  const sel = window.getSelection();
  if(selectionInsideDiag(sel) && !sel.isCollapsed){
    lastPointer = {x:e.clientX, y:e.clientY};
    onCursor();
    if(pendingSel) showDxPop();
  }
});
document.addEventListener('selectionchange', ()=>{
  const ae = document.activeElement;
  if(ae && (ae.tagName==='TEXTAREA' || ae.id==='dxPopSearch')) return;
  onCursor();
});
$('dxPop').addEventListener('mousedown', e=>{
  if(e.target.id!=='dxPopSearch') e.preventDefault();
});
$('dxPopSearch').addEventListener('input', renderDxPopList);
document.addEventListener('mousedown', e=>{
  const pop = $('dxPop');
  if(pop.hidden) return;
  if(pop.contains(e.target) || $('diag').contains(e.target)) return;
  if(e.target && e.target.dataset && e.target.dataset.dx) return;
  hideDxPop();
});

// use current selection to fill evidence+hex
$('f_evidence').addEventListener('focus', ()=>{ if(window._selFill && !$('f_evidence').value){ $('f_evidence').value=window._selFill.evidence; } });
$('f_hex').addEventListener('focus', ()=>{ if(window._selFill && !$('f_hex').value){ $('f_hex').value=window._selFill.hex; } });
$('f_diseases').addEventListener('input', ()=>{ syncDiseaseChecks(); updateHexWarn(); renderRowPreview(); });
['f_read','f_verdict','f_evidence','f_hex','f_note'].forEach(id=>{
  $(id).addEventListener('input', renderRowPreview);
  $(id).addEventListener('change', renderRowPreview);
});

// Commit the current doc's field values into the in-memory store + localStorage
// buffer (does NOT touch disk). Called on navigation and before saving.
function commitCurrent(){
  if(!cur) return;
  const r = currentRowValues();
  // keep a row only if something was entered
  const nonEmpty = ['read','verdict','diseases','evidence','hex','note'].some(k=>r[k]);
  if(nonEmpty){ saved[cur.doc_id] = r; }
  localStorage.setItem(LS_KEY, JSON.stringify(saved));
  $('savedCount').textContent = Object.keys(saved).length;
}

$('clearRow').onclick = ()=>{
  if(!cur) return; delete saved[cur.doc_id];
  localStorage.setItem(LS_KEY, JSON.stringify(saved));
  fillFields(cur.doc_id); $('savedCount').textContent = Object.keys(saved).length;
  $('exportMsg').textContent = 'cleared in UI (labels.csv unchanged; delete rows there manually)';
};

$('exportBtn').onclick = async ()=>{
  commitCurrent();
  const rows = Object.values(saved);
  const res = await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify({rows})});
  const data = await res.json();
  $('exportMsg').innerHTML = data.saved
     ? '<span class="saved">saved \u2192 '+data.saved+' ('+data.rows+' rows in file)</span>'
     : 'save failed';
  $('rowState').textContent = 'saved';
  $('savedCount').textContent = data.rows;
  renderRowPreview();
};

document.addEventListener('keydown', e=>{
  if(e.key==='Escape' && !$('dxPop').hidden){ hideDxPop(); return; }
  if(!$('dxPop').hidden) return;
  if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
  if(e.key==='ArrowRight' || e.key==='Enter') $('next').click();
  else if(e.key==='ArrowLeft' || e.key==='b') $('prev').click();
  else if(e.key==='m') $('toggleMd').click();
  else if(e.key==='n') $('fNext').click();
  else if(e.key==='c') jumpChange();
  else if(e.key==='p') $('fPrev').click();
  else if(e.key==='h') $('toggleHex').click();
});

(function setupSplitter(){
  const split = $('splitter');
  const right = document.querySelector('.right');
  const KEY = 'view_pack_web_right_w';
  const savedW = parseInt(localStorage.getItem(KEY)||'', 10);
  function applyWidth(px){
    document.documentElement.style.setProperty('--right-w', px+'px');
  }
  if(savedW) applyWidth(savedW);
  let dragging = false;
  split.addEventListener('mousedown', e => {
    dragging = true;
    split.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if(!dragging) return;
    const rect = document.querySelector('.wrap').getBoundingClientRect();
    const min = 280;
    const max = Math.max(min, rect.width - 200);
    const w = Math.max(min, Math.min(max, rect.right - e.clientX));
    applyWidth(w);
  });
  window.addEventListener('mouseup', () => {
    if(!dragging) return;
    dragging = false;
    split.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    const w = parseInt(getComputedStyle(right).width, 10);
    if(w) localStorage.setItem(KEY, String(w));
  });
  split.addEventListener('dblclick', () => {
    document.documentElement.style.removeProperty('--right-w');
    localStorage.removeItem(KEY);
  });
})();

async function bootstrap(){
  // 1) CSV is the source of truth
  try {
    const data = await (await fetch('/api/rows')).json();
    Object.assign(saved, data.rows || {});
  } catch(e){}
  // 2) overlay localStorage (unsaved edits win)
  try {
    const ls = JSON.parse(localStorage.getItem(LS_KEY)||'{}');
    Object.assign(saved, ls);
  } catch(e){}
  $('savedCount').textContent = Object.keys(saved).length;
  buildDiseasePicker();
  try { Object.assign(fxState, JSON.parse(localStorage.getItem(FX_KEY)||'{}')); } catch(e){}
  try { summary = (await (await fetch('/api/summary')).json()).docs || []; } catch(e){}
  buildFilter();
  load(idx);
}

// ---- detected findings (from detectors.py) ----
const FX_KEY = 'view_pack_web_fx';
const fxState = {};          // doc_id -> {finding id: 'a' accepted | 'r' rejected}
let summary = [];            // [{i, doc_id, suggested, codes:['C2/medium',...]}]
let activeFid = null;

function fxOf(fid){ return (cur.findings||[]).find(f=>f.id===fid); }
function fxGet(fid){ return ((fxState[cur.doc_id]||{})[fid]) || ''; }
function fxSet(fid, v){
  const m = fxState[cur.doc_id] = fxState[cur.doc_id] || {};
  if(v) m[fid] = v; else delete m[fid];
  localStorage.setItem(FX_KEY, JSON.stringify(fxState));
}

function highlightFindings(){
  const spans = Array.from($('diag').querySelectorAll('span[data-off]'));
  for(const f of (cur.findings||[])){
    if(f.start < 0) continue;
    for(const sp of spans){
      const off = parseInt(sp.dataset.off, 10);
      if(off >= f.start && off < f.end){
        sp.classList.add('hl-'+f.confidence);
        sp.dataset.fids = (sp.dataset.fids ? sp.dataset.fids+' ' : '') + f.id;
      }
    }
  }
}

function setActive(fid, scroll){
  activeFid = fid;
  $('diag').querySelectorAll('.hl-active').forEach(e=>e.classList.remove('hl-active'));
  const marks = Array.from($('diag').querySelectorAll('span[data-fids]'))
                     .filter(e=>e.dataset.fids.split(' ').includes(fid));
  marks.forEach(e=>e.classList.add('hl-active'));
  if(scroll && marks.length) marks[0].scrollIntoView({block:'center', behavior:'smooth'});
  document.querySelectorAll('.fx.flash').forEach(e=>e.classList.remove('flash'));
  const card = document.querySelector('.fx[data-fid="'+fid+'"]');
  if(card){ card.classList.add('flash'); if(!scroll) card.scrollIntoView({block:'nearest', behavior:'smooth'}); }
}

function renderFindings(){
  const list = $('fxList');
  const showLow = $('fxShowLow').checked;
  const all = cur.findings || [];
  const shown = all.filter(f=>showLow || f.confidence!=='low');
  const accepted = all.filter(f=>fxGet(f.id)==='a').length;
  $('fxCount').textContent = '('+all.length+(accepted? ', '+accepted+' accepted':'')+')';
  const sv = (cur.suggested_verdict||'').replace('?','');
  $('sugg').textContent = 'suggested: '+(cur.suggested_verdict||'-');
  $('sugg').className = 'sugg '+sv;
  if(!shown.length){
    list.innerHTML = '<div class="hint">'+(all.length? 'only low-confidence findings (tick "show low")'
                     : 'no detector fired. Still read the whole document: D4/D5, F3, H3 and subtle C3/C4 are not automated.')+'</div>';
    return;
  }
  list.innerHTML = shown.map(f=>{
    const st = fxGet(f.id);
    const where = (f.section||'whole doc') + (f.line? ' \u00b7 line '+f.line : '') + (f.start>=0? ' \u00b7 @'+f.start : '');
    return '<div class="fx c-'+f.confidence+(st? ' st-'+st:'')+'" data-fid="'+f.id+'">'
      + '<div class="fx-head"><span class="code">'+f.code+'</span>'
      + '<span class="conf conf-'+f.confidence+'">'+f.confidence+'</span>'
      + '<span class="fx-where">'+escapeHtml(where)+'</span>'
      + (st? '<span class="fx-state">'+(st==='a'?'accepted':'rejected')+'</span>' : '')
      + '<span class="fx-btns">'
      + (f.start>=0? '<button data-act="show">show</button>' : '')
      + '<button data-act="a">'+(st==='a'?'undo':'accept')+'</button>'
      + '<button data-act="r">'+(st==='r'?'undo':'reject')+'</button></span></div>'
      + '<div class="fx-reason">'+escapeHtml(f.reason)+'</div>'
      + '<div class="fx-ev">evidence: '+escapeHtml(f.evidence)+(f.hex_value? '  \u00b7 hex '+escapeHtml(f.hex_value):'')
      + (f.sensitive? '  \u00b7 <b>value never copied</b>':'')+'</div></div>';
  }).join('');
}

function evidenceBits(code){
  const prefix = code+': ';
  const rows = ($('f_evidence').value||'').split('\n');
  const k = rows.findIndex(r=>r.startsWith(prefix));
  return {rows, k, prefix, bits: k===-1 ? [] : rows[k].slice(prefix.length).split('; ').map(s=>s.trim()).filter(Boolean)};
}
function addEvidence(code, bit){
  const e = evidenceBits(code);
  if(e.bits.includes(bit)) return;
  e.bits.push(bit);
  const line = e.prefix + e.bits.join('; ');
  const rows = e.rows.filter(r=>r.trim());
  const k = rows.findIndex(r=>r.startsWith(e.prefix));
  if(k===-1) rows.push(line); else rows[k] = line;
  $('f_evidence').value = rows.join('\n');
}
function removeEvidence(code, bit){
  const e = evidenceBits(code);
  if(e.k===-1) return;
  const bits = e.bits.filter(b=>b!==bit);
  if(bits.length) e.rows[e.k] = e.prefix + bits.join('; '); else e.rows.splice(e.k, 1);
  $('f_evidence').value = e.rows.filter(r=>r.trim()).join('\n');
}
function hexTokens(){ return ($('f_hex').value||'').split(/\s+/).filter(Boolean); }
function usesHex(f){ return f.hex_value && !f.sensitive && /^[AG]/.test(f.code); }

function acceptFinding(f){
  checkDisease(f.code); onDxCheck();
  addEvidence(f.code, f.evidence);
  if(usesHex(f)){
    const have = hexTokens();
    f.hex_value.split(/\s+/).forEach(t=>{ if(!have.includes(t)) have.push(t); });
    $('f_hex').value = have.join(' ');
  }
  fxSet(f.id, 'a');
}
function unacceptFinding(f){
  removeEvidence(f.code, f.evidence);
  fxSet(f.id, '');
  const stillUsed = (cur.findings||[]).some(g=>g.code===f.code && fxGet(g.id)==='a');
  if(!stillUsed && !evidenceBits(f.code).bits.length){
    const cb = dxCheckboxes().find(c=>c.value===f.code);
    if(cb){ cb.checked = false; onDxCheck(); }
  }
  if(usesHex(f)){
    const keep = new Set((cur.findings||[]).filter(g=>fxGet(g.id)==='a' && usesHex(g))
                           .flatMap(g=>g.hex_value.split(/\s+/)));
    const drop = new Set(f.hex_value.split(/\s+/).filter(t=>!keep.has(t)));
    $('f_hex').value = hexTokens().filter(t=>!drop.has(t)).join(' ');
  }
}
function afterFindingChange(){
  updateHexWarn(); renderRowPreview(); commitCurrent(); renderFindings();
}

$('fxList').addEventListener('click', e=>{
  const btn = e.target.closest('button'); const card = e.target.closest('.fx');
  if(!card) return;
  const f = fxOf(card.dataset.fid); if(!f) return;
  if(!btn){ setActive(f.id, true); return; }
  const act = btn.dataset.act, st = fxGet(f.id);
  if(act==='show'){ setActive(f.id, true); return; }
  if(act==='a'){ if(st==='a') unacceptFinding(f); else acceptFinding(f); }
  if(act==='r'){ if(st==='a') unacceptFinding(f); fxSet(f.id, st==='r' ? '' : 'r'); }
  afterFindingChange(); setActive(f.id, false);
});
$('diag').addEventListener('click', e=>{
  const sp = e.target.closest('span[data-fids]');
  if(!sp || !window.getSelection().isCollapsed) return;
  const ids = sp.dataset.fids.split(' ');
  const k = ids.indexOf(activeFid);
  setActive(ids[(k+1) % ids.length], false);   // repeated clicks cycle overlapping findings
});
$('fxShowLow').onchange = renderFindings;
$('fxAcceptHigh').onclick = ()=>{
  (cur.findings||[]).filter(f=>f.confidence==='high' && !fxGet(f.id)).forEach(acceptFinding);
  afterFindingChange();
};
$('fxUseVerdict').onclick = ()=>{
  const v = (cur.suggested_verdict||'').replace('?','');
  if(v){ $('f_verdict').value = v; renderRowPreview(); commitCurrent(); }
};

// ---- commit context (old vs new diff) ----
function highlightDiff(){
  const ranges = (cur.context && cur.context.ranges) || [];
  if(!ranges.length) return;
  $('diag').querySelectorAll('span[data-off]').forEach(sp=>{
    const off = parseInt(sp.dataset.off, 10);
    const r = ranges.find(r=>off >= r.start && off < r.end);
    if(r) sp.classList.add(r.kind==='add' ? 'df-add' : 'df-del');
  });
}
let changeIdx = -1;
function renderContext(){
  const c = cur.context || {};
  $('ctxHead').style.display = $('ctx').style.display = c.kind ? '' : 'none';
  if(!c.kind) return;
  changeIdx = -1;
  const n = (c.ranges||[]).length;
  $('ctx').innerHTML = '<b>'+escapeHtml(c.kind)+'</b> &nbsp;<span class="stat">old '+c.old_lines+' lines \u2192 new '
    + c.new_lines+' lines \u00b7 <span class="plus">+'+c.added+'</span> / <span class="minus">-'+c.removed+'</span> \u00b7 '
    + Math.round((c.similarity||0)*100)+'% same</span>'
    + (n ? ' &nbsp;<button id="nextChange" title="jump to next changed block (c)">next change \u25bc</button>' : '')
    + '<div style="margin-top:4px">'+escapeHtml(c.summary||'')+'</div>';
  if(n) $('nextChange').onclick = jumpChange;
}
function jumpChange(){
  const rs = (cur.context && cur.context.ranges) || [];
  if(!rs.length) return;
  const order = rs.slice().sort((a,b)=>a.start-b.start);
  changeIdx = (changeIdx+1) % order.length;
  const r = order[changeIdx];
  const sp = Array.from($('diag').querySelectorAll('span[data-off]'))
                  .find(e=>parseInt(e.dataset.off,10) >= r.start);
  if(sp) sp.scrollIntoView({block:'center', behavior:'smooth'});
  $('selstatus').textContent = (r.kind==='add'?'added':'removed')+' block '+(changeIdx+1)+'/'+order.length+' at line '+r.line;
}

// ---- checklist: what was checked and why it is clean ----
const CK_LABEL = {hit:'\u2716 found', weak:'\u26a0 low hint', pass:'\u2714 passed', na:'\u2013 n/a', manual:'\u25cb read by eye'};
const CK_ORDER = {hit:0, weak:1, manual:2, pass:3, na:4};
function renderChecks(){
  $('why').textContent = cur.why || '';
  const checks = cur.checks || [];
  const n = {}; checks.forEach(c=>{ n[c.status]=(n[c.status]||0)+1; });
  $('ckSum').textContent = 'what was checked: '+(n.pass||0)+' passed \u00b7 '+(n.hit||0)+' found \u00b7 '
    + (n.weak||0)+' low \u00b7 '+(n.manual||0)+' by eye \u00b7 '+(n.na||0)+' n/a';
  const onlyOpen = $('ckOnlyOpen').checked;
  const rows = checks.filter(c=>!onlyOpen || !['pass','na'].includes(c.status))
    .sort((a,b)=>(CK_ORDER[a.status]-CK_ORDER[b.status]) || ((DX_ORDER[a.code]??0)-(DX_ORDER[b.code]??0)));
  $('ckTable').innerHTML = rows.map(c=>'<tr class="ck-row-'+c.status+'"><td class="code">'+c.code+'</td>'
    + '<td class="ck-st ck-'+c.status+'">'+CK_LABEL[c.status]+'</td>'
    + '<td><div>'+escapeHtml(c.result)+'</div><div class="ck-how">'+escapeHtml(c.checked)+'</div></td></tr>').join('');
}
$('ckOnlyOpen').onchange = renderChecks;
try { $('ckBox').open = localStorage.getItem(FX_KEY+'_ck') !== '0'; } catch(e){ $('ckBox').open = true; }
$('ckBox').addEventListener('toggle', ()=>{ try { localStorage.setItem(FX_KEY+'_ck', $('ckBox').open?'1':'0'); } catch(e){} });

// ---- medicine preview + sheet ----
var medTimer = null, medKey = '';
function scheduleMedicine(){
  if(!cur) return;
  const key = cur.i+'|'+($('f_diseases').value||'').toUpperCase();
  if(key===medKey) return;
  clearTimeout(medTimer);
  medTimer = setTimeout(()=>renderMedicine(key), 250);
}
async function renderMedicine(key){
  medKey = key;
  const codes = ($('f_diseases').value||'').trim();
  if(!codes){ $('medList').innerHTML = '<div class="hint">add a disease code to see its medicine</div>'; return; }
  let data;
  try { data = await (await fetch('/api/medicine?i='+cur.i+'&codes='+encodeURIComponent(codes))).json(); }
  catch(e){ return; }
  if(!cur || data.doc_id!==cur.doc_id) return;
  const label = {changed:'fixes text', dropped:'would drop', unchanged:'no effect', none:'no rule'};
  $('medList').innerHTML = data.medicines.map(m=>{
    const diff = (m.diff||[]).map(l=>'<span class="'+(l[0]==='+'?'a':l[0]==='-'?'d':'h')+'">'+escapeHtml(l)+'</span>').join('\n');
    return '<div class="med" data-code="'+m.code+'"><div class="med-head"><span class="code">'+m.code+'</span>'
      + (m.action? '<span class="hint">'+m.action+'</span>' : '')
      + '<span class="med-st med-'+m.status+'">'+label[m.status]+'</span>'
      + (m.action? '<button data-act="note" title="record a case where this medicine is wrong">medicine wrong here</button>' : '')
      + '</div><div class="med-words">'+escapeHtml(m.words)+'</div>'
      + (diff? '<pre>'+diff+'</pre>' : '')
      + (m.notes||[]).map(n=>'<div class="med-note">side effect noted: '+escapeHtml(n)+'</div>').join('')
      + '</div>';
  }).join('');
}
$('medList').addEventListener('click', async e=>{
  const btn = e.target.closest('button[data-act="note"]'); if(!btn || !cur) return;
  const code = btn.closest('.med').dataset.code;
  const note = prompt('Why is the '+code+' medicine wrong on this document? (one line, no personal data or secrets)');
  if(!note || !note.trim()) return;
  await fetch('/api/medicine_note', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({doc_id: cur.doc_id, code, note: note.trim()})});
  medKey = ''; scheduleMedicine();
});
$('medSheetBtn').onclick = async ()=>{
  await $('exportBtn').onclick();
  const data = await (await fetch('/api/medicine_sheet', {method:'POST', headers:{'Content-Type':'application/json'},
    body: '{}'})).json();
  $('medSheetMsg').innerHTML = data.saved
    ? '<span class="saved">wrote '+escapeHtml(data.saved)+'</span><br>'
      + data.summary.map(s=>escapeHtml(s.code+': '+s.count+' docs, '+s.status
          + (s.missed? ', misses '+s.missed : '') + (s.extra? ', also hits '+s.extra+' unlabelled' : ''))).join('<br>')
    : 'failed';
};

// ---- filter navigation ----
function buildFilter(){
  const sel = $('filter');
  const count = pred => summary.filter(pred).length;
  const codes = {};
  summary.forEach(d=>new Set(d.codes.map(c=>c.split('/')[0])).forEach(c=>{ codes[c]=(codes[c]||0)+1; }));
  const opts = [
    ['all', 'all docs ('+summary.length+')'],
    ['unlabelled', 'no verdict yet'],
    ['flagged', 'medium/high findings ('+count(d=>d.codes.some(c=>!c.endsWith('/low')))+')'],
    ['s:drop?', 'suggested drop ('+count(d=>d.suggested==='drop?')+')'],
    ['s:fix?', 'suggested fix ('+count(d=>d.suggested==='fix?')+')'],
    ['s:clean?', 'suggested clean ('+count(d=>d.suggested==='clean?')+')'],
    ...Object.keys(codes).sort((a,b)=>(DX_ORDER[a]??999)-(DX_ORDER[b]??999))
       .map(c=>['c:'+c, c+' ('+codes[c]+' docs)']),
  ];
  sel.innerHTML = opts.map(([v,t])=>'<option value="'+v+'">'+escapeHtml(t)+'</option>').join('');
  try { sel.value = localStorage.getItem(FX_KEY+'_filter') || 'all'; } catch(e){}
  if(!sel.value) sel.value = 'all';
  sel.onchange = ()=>{ localStorage.setItem(FX_KEY+'_filter', sel.value); updateFilterInfo(); };
}
function matches(d){
  const f = $('filter').value || 'all';
  if(f==='all') return true;
  if(f==='unlabelled') return !((saved[d.doc_id]||{}).verdict);
  if(f==='flagged') return d.codes.some(c=>!c.endsWith('/low'));
  if(f.startsWith('s:')) return d.suggested===f.slice(2);
  if(f.startsWith('c:')) return d.codes.some(c=>c.split('/')[0]===f.slice(2));
  return true;
}
function stepFilter(dir){
  commitCurrent();
  for(let k=idx+dir; k>=0 && k<summary.length; k+=dir){
    if(matches(summary[k])){ load(k); return; }
  }
  $('fInfo').textContent = 'no more matches '+(dir>0?'after':'before')+' this doc';
}
function updateFilterInfo(){
  if(!summary.length || !cur) return;
  const hits = summary.filter(matches);
  const pos = hits.findIndex(d=>d.i===idx);
  $('fInfo').textContent = hits.length+' match'+(hits.length===1?'':'es')+(pos>=0? ' \u00b7 this is '+(pos+1) : '');
}
$('fNext').onclick = ()=>stepFilter(1);
$('fPrev').onclick = ()=>stepFilter(-1);

bootstrap();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
