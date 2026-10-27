"""Section-aware disease detectors for diagnosis sample packs.

Every finding carries a confidence level, the exact span (code-point offsets
into the raw text, the same as Python text[offset]) and a reason that says why
the rule fired and when it is a false positive. Findings are review hints for
the viewer; they are never final labels.

Commit packs (bigcode/commitpackft) have a fixed layout:

    ### commit message / ### file path / ### old contents / ### new contents

The old and new contents are near-copies of each other by design, so rules
such as repeated lines and licence headers are computed per section, never
across the old/new pair.

For F1 (personal data) and E3 (secrets) the evidence gives only the type and
the position. The matched value is never copied into evidence or reason.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

HIGH, MEDIUM, LOW = "high", "medium", "low"
CONFIDENCE_RANK = {HIGH: 0, MEDIUM: 1, LOW: 2}


@dataclass
class Finding:
    code: str
    confidence: str
    reason: str
    evidence: str
    start: int = -1          # code-point offset, -1 = whole document
    end: int = -1
    section: str = ""
    line: int = 0
    hex_value: str = ""
    sensitive: bool = False  # F1/E3: never copy the value anywhere

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Section:
    name: str
    start: int   # offset of the first body character
    end: int
    body: str


@dataclass
class Doc:
    index: int
    doc_id: str
    lang_label: str
    text: str
    meta: dict
    sections: list[Section]
    line_starts: list[int] = field(default_factory=list)

    @property
    def is_commit(self) -> bool:
        return any(s.name == "new contents" for s in self.sections)

    def section(self, name: str) -> Section | None:
        return next((s for s in self.sections if s.name == name), None)

    def code_sections(self) -> list[Section]:
        if not self.is_commit:
            return self.sections
        return [s for s in self.sections if s.name in ("old contents", "new contents")]

    def line_of(self, offset: int) -> int:
        lo, hi = 0, len(self.line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def section_at(self, offset: int) -> str:
        for s in self.sections:
            if s.start <= offset <= s.end:
                return s.name
        return ""


SECTION_RE = re.compile(r"^### (commit message|file path|old contents|new contents)[ \t]*$", re.M)


def parse_doc(index: int, record: dict) -> Doc:
    text = record.get("text") or ""
    meta_raw = record.get("meta") or {}
    if isinstance(meta_raw, str):
        try:
            meta_raw = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta_raw = {}
    sections: list[Section] = []
    heads = list(SECTION_RE.finditer(text))
    for i, head in enumerate(heads):
        start = head.end() + 1 if head.end() < len(text) else head.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        sections.append(Section(head.group(1), start, end, text[start:end]))
    if not sections:
        sections = [Section("text", 0, len(text), text)]
    line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
    return Doc(index, record.get("doc_id", ""), record.get("lang_label", ""), text,
               meta_raw if isinstance(meta_raw, dict) else {}, sections, line_starts)


def _hex(chars: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in chars)


def _quote(text: str, limit: int = 60) -> str:
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _make(doc: Doc, code: str, confidence: str, reason: str, start: int = -1, end: int = -1,
          evidence: str = "", hex_value: str = "", sensitive: bool = False) -> Finding:
    line = doc.line_of(start) if start >= 0 else 0
    section = doc.section_at(start) if start >= 0 else ""
    if not evidence:
        evidence = _quote(doc.text[start:end]) if start >= 0 else ""
    if start >= 0:
        evidence = f"{evidence} at offset {start} (line {line})"
    return Finding(code, confidence, reason, evidence, start, end, section, line, hex_value, sensitive)


# ---------------------------------------------------------------------------
# A. Characters
# ---------------------------------------------------------------------------
A1_CHARS = {
    "\u200b": "ZERO WIDTH SPACE", "\ufeff": "BOM / ZERO WIDTH NO-BREAK SPACE",
    "\u00ad": "SOFT HYPHEN", "\u2060": "WORD JOINER", "\u180e": "MONGOLIAN VOWEL SEPARATOR",
    "\u202a": "LRE", "\u202b": "RLE", "\u202c": "PDF", "\u202d": "LRO", "\u202e": "RLO",
    "\u2066": "LRI", "\u2067": "RLI", "\u2068": "FSI", "\u2069": "PDI",
    "\u200e": "LEFT-TO-RIGHT MARK", "\u200f": "RIGHT-TO-LEFT MARK",
}
ODD_SPACES = {"\u00a0", "\u1680", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", "\u0085"} | {
    chr(c) for c in range(0x2000, 0x200B)}
INDIC_RANGE = (0x0900, 0x0DFF)


def _is_indic(ch: str) -> bool:
    return bool(ch) and INDIC_RANGE[0] <= ord(ch) <= INDIC_RANGE[1]


def detect_characters(doc: Doc) -> list[Finding]:
    out: list[Finding] = []
    text = doc.text
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    spaces: defaultdict[str, list[int]] = defaultdict(list)
    for i, ch in enumerate(text):
        if ch in A1_CHARS:
            grouped[ch].append(i)
        elif ch in "\u200c\u200d":
            before = text[i - 1] if i else ""
            after = text[i + 1] if i + 1 < len(text) else ""
            if not (_is_indic(before) or _is_indic(after)):
                grouped[ch].append(i)
        elif unicodedata.category(ch) in ("Cf", "Co", "Cn", "Cs"):
            grouped[ch].append(i)           # any other format / private-use / unassigned / surrogate
        elif ch in ODD_SPACES or (unicodedata.category(ch) in ("Zs", "Zl", "Zp") and ch != " "):
            spaces[ch].append(i)
    for ch, offsets in spaces.items():
        name = unicodedata.name(ch, f"U+{ord(ch):04X}")
        out.append(_make(doc, "A1", MEDIUM,
                         f"{len(offsets)} x {name}: looks like a normal space (or line break) but is a different "
                         "code point; usually left over from HTML &nbsp; or a word processor. Normalise to ' ' / "
                         "'\\n'. Legitimate only inside a string that tests it.",
                         offsets[0], offsets[0] + 1, evidence=f"{len(offsets)} x {name}, first",
                         hex_value=f"U+{ord(ch):04X}"))
    for ch, offsets in grouped.items():
        name = A1_CHARS.get(ch) or ("ZWJ/ZWNJ outside Indic text" if ch in "\u200c\u200d"
                                    else f"{unicodedata.name(ch, 'unnamed')} ({unicodedata.category(ch)})")
        first = offsets[0]
        bom_at_file_start = ch == "\ufeff" and any(first == s.start for s in doc.sections)
        reason = (f"{len(offsets)} x {name} (invisible). "
                  + ("BOM at the start of a file body: the file was saved with a UTF-8 BOM. "
                     if bom_at_file_start else "")
                  + "Invisible junk in training text; strip it.")
        out.append(_make(doc, "A1", HIGH, reason, first, first + 1,
                         evidence=f"{len(offsets)} x {name}, first", hex_value=f"U+{ord(ch):04X}"))

    cr = [i for i, ch in enumerate(text) if ch == "\r"]
    crlf = sum(1 for i in cr if text[i + 1: i + 2] == "\n")
    if cr:
        if crlf == len(cr):
            out.append(_make(doc, "A2", MEDIUM,
                             f"{len(cr)} CRLF line endings (Windows file). U+000D is a control character; "
                             "normalise to LF. Often harmless in code, so verdict is usually 'fix'.",
                             cr[0], cr[0] + 1, evidence=f"{len(cr)} x CR (CRLF), first", hex_value="U+000D"))
        else:
            lone = [i for i in cr if text[i + 1: i + 2] != "\n"]
            out.append(_make(doc, "A2", HIGH,
                             f"{len(lone)} lone carriage return(s) not followed by LF: broken line endings.",
                             lone[0], lone[0] + 1, evidence=f"{len(lone)} x lone CR, first", hex_value="U+000D"))

    controls: defaultdict[int, list[int]] = defaultdict(list)
    for i, ch in enumerate(text):
        value = ord(ch)
        if ch in "\t\n\r":
            continue
        if value < 32 or 0x7F <= value <= 0x9F:
            controls[value].append(i)
    for value, offsets in controls.items():
        if value == 0x0C:
            conf, why = MEDIUM, "form feed (U+000C). Used as a page break in some C/Python/Go sources (legitimate) but still a control char."
        elif 0x80 <= value <= 0x9F:
            conf, why = HIGH, "C1 control character; usually a sign of cp1252 text decoded as latin-1 (mojibake)."
        else:
            conf, why = HIGH, f"control character U+{value:04X} ({unicodedata.name(chr(value), 'control')})."
        out.append(_make(doc, "A2", conf, f"{len(offsets)} x {why}", offsets[0], offsets[0] + 1,
                         evidence=f"{len(offsets)} x U+{value:04X}, first", hex_value=f"U+{value:04X}"))

    out += _detect_mojibake(doc)

    fffd = [i for i, ch in enumerate(text) if ch == "\ufffd"]
    if fffd:
        out.append(_make(doc, "A4", HIGH,
                         f"{len(fffd)} replacement character(s) U+FFFD: bytes were lost before release. "
                         "Cannot be repaired; many of them usually means drop.",
                         fffd[0], fffd[0] + 1, evidence=f"{len(fffd)} x U+FFFD, first", hex_value="U+FFFD"))

    out += _detect_nfc(doc)
    out += _detect_mixed_script(doc)
    return out


MOJIBAKE_RE = re.compile(
    r"(?:Ã[\u0080-\u00bf]|Â[\u00a0-\u00bf]|â€[\u0080-\u00bf\u2018-\u203a\u0152\u0153\u0160\u0161\u017d\u017e\u0178\u02c6\u02dc\u2122]"
    r"|ðŸ[\u0080-\u00bf\u2018-\u203a\u0152-\u0178]|à[¤¥¦§¨©ª«¬­®¯°±²³´µ][\u0080-\u00bf\u2018-\u203a\u0152-\u0178])+"
)


def _detect_mojibake(doc: Doc) -> list[Finding]:
    out = []
    by_code: defaultdict[str, list[re.Match]] = defaultdict(list)
    for m in MOJIBAKE_RE.finditer(doc.text):
        in_code = doc.section_at(m.start()) in ("old contents", "new contents")
        by_code["E5" if in_code else "A3"].append(m)
    for code, matches in by_code.items():
        m = matches[0]
        where = "inside code (string or comment)" if code == "E5" else "in prose"
        out.append(_make(doc, code, HIGH,
                         f"{len(matches)} mojibake sequence(s) {where}: UTF-8 bytes decoded as latin-1/cp1252 "
                         "(e.g. 'Ã©' for 'é', 'â€™' for '’'). Fixable with ftfy-style re-decoding.",
                         m.start(), m.end(), hex_value=_hex(m.group(0)[:6])))
    return out


def _detect_nfc(doc: Doc) -> list[Finding]:
    if unicodedata.is_normalized("NFC", doc.text):
        return []
    hits = []
    for m in re.finditer(r"\S+", doc.text):
        word = m.group(0)
        if not unicodedata.is_normalized("NFC", word):
            hits.append(m)
    if not hits:
        return []
    m = hits[0]
    nfc = unicodedata.normalize("NFC", m.group(0))
    return [_make(doc, "A5", HIGH,
                  f"{len(hits)} word(s) are not in NFC (decomposed accents or two-codepoint nukta). "
                  f"NFC turns {_hex(m.group(0)[:8])} into {_hex(nfc[:8])}; check whether the look changes.",
                  m.start(), m.end(), hex_value=_hex(m.group(0)[:8]))]


def _script(ch: str) -> str:
    name = unicodedata.name(ch, "")
    for script in ("LATIN", "CYRILLIC", "GREEK", "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU",
                   "KANNADA", "MALAYALAM", "GUJARATI", "GURMUKHI", "ORIYA", "ARABIC"):
        if name.startswith(script):
            return script
    return ""


def _detect_mixed_script(doc: Doc) -> list[Finding]:
    hits = []
    for m in re.finditer(r"[^\W\d_]{2,}", doc.text):
        scripts = {_script(c) for c in m.group(0)} - {""}
        if len(scripts) > 1:
            hits.append((m, scripts))
    if not hits:
        return []
    m, scripts = hits[0]
    return [_make(doc, "A6", MEDIUM,
                  f"{len(hits)} word(s) mix scripts ({'+'.join(sorted(scripts))}): look-alike letters, "
                  "e.g. Cyrillic 'а' inside a Latin word. Breaks search and tokenisation.",
                  m.start(), m.end(), hex_value=_hex(m.group(0)[:8]))]


# ---------------------------------------------------------------------------
# B. Markup and ghost tokens
# ---------------------------------------------------------------------------
HTML_ENTITY_RE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#[xX][0-9A-Fa-f]{1,6});")
HTML_TAG_RE = re.compile(r"</?(?:div|span|p|br|a|b|i|u|em|strong|ul|ol|li|table|tr|td|th|img|h[1-6]|pre|code|font|center|hr)\b[^>]{0,200}/?>", re.I)
GHOST_RE = re.compile(
    r"<\|[a-z_]{2,30}\|>|\[/?INST\]|<<SYS>>|<</SYS>>|"
    r"^#{2,3} ?(?:Assistant|Human|User|System|Instruction|Response)\s*:", re.M | re.I)
GHOST_WEAK_RE = re.compile(r"</?s>|</?think>")  # case-sensitive: <S> is a Java/TS generic
TEMPLATE_RE = re.compile(r"\{\{[^{}\n]{1,80}\}\}|\$\{[^{}\n]{1,80}\}|\{%[^%\n]{1,80}%\}|\bundefined\b|\bNaN\b|\[object Object\]")
MARKUP_EXTS = {".html", ".htm", ".xml", ".jsx", ".tsx", ".vue", ".svg", ".md", ".erb", ".php"}


def detect_markup(doc: Doc) -> list[Finding]:
    out = []
    text = doc.text
    ext = _file_ext(doc)
    msg = doc.section("commit message") if doc.is_commit else None
    prose_sections = [msg] if msg else ([] if doc.is_commit else doc.sections)

    for s in prose_sections:
        ents = list(HTML_ENTITY_RE.finditer(s.body))
        if ents:
            m = ents[0]
            out.append(_make(doc, "B1", HIGH,
                             f"{len(ents)} HTML entit(y/ies) in prose: should be decoded (html.unescape).",
                             s.start + m.start(), s.start + m.end()))
        tags = list(HTML_TAG_RE.finditer(s.body))
        if tags:
            m = tags[0]
            out.append(_make(doc, "B2", HIGH if not doc.is_commit else MEDIUM,
                             f"{len(tags)} HTML tag(s) in prose. In a commit message this can be a pasted PR body.",
                             s.start + m.start(), s.start + m.end()))
        tmpl = [m for m in TEMPLATE_RE.finditer(s.body)
                if not (doc.is_commit and m.group(0) in ("undefined", "NaN", "[object Object]"))]
        if tmpl:
            m = tmpl[0]
            out.append(_make(doc, "B4", LOW if doc.is_commit else MEDIUM,
                             f"{len(tmpl)} template/code junk token(s) in prose ({{{{ }}}}, ${{}}, undefined, NaN). "
                             "False positive if the message is talking about the code itself.",
                             s.start + m.start(), s.start + m.end()))
        if s.body.count("```") % 2:
            pos = s.start + s.body.rfind("```")
            out.append(_make(doc, "B5", HIGH, "Odd number of ``` fences: an unclosed code block.", pos, pos + 3))

    if doc.is_commit and ext not in MARKUP_EXTS:
        for s in doc.code_sections():
            ents = list(HTML_ENTITY_RE.finditer(s.body))
            if ents:
                m = ents[0]
                out.append(_make(doc, "B1", LOW,
                                 f"{len(ents)} HTML entit(y/ies) inside {s.name}. Usually legitimate in code "
                                 "(Javadoc &lt;, test strings, HTML escaping helpers). Accept only if the entity "
                                 "is a scraping artefact (e.g. &amp;&amp; for && in an expression).",
                                 s.start + m.start(), s.start + m.end()))
                break

    ghosts = list(GHOST_RE.finditer(text))
    if ghosts:
        m = ghosts[0]
        out.append(_make(doc, "B3", HIGH,
                         f"{len(ghosts)} ghost chat/special marker(s) like '{_quote(m.group(0), 20)}'. Teaches the "
                         "model to emit our special tokens as plain text. Almost never legitimate.",
                         m.start(), m.end()))
    weak = list(GHOST_WEAK_RE.finditer(text))
    if weak:
        m = weak[0]
        in_code = doc.section_at(m.start()) in ("old contents", "new contents")
        out.append(_make(doc, "B3", LOW if in_code else MEDIUM,
                         f"{len(weak)} '<s>'/'<think>' marker(s). In code this is often real HTML (strikethrough) "
                         "or a parser test; in prose it is a ghost token.", m.start(), m.end()))
    return out


# ---------------------------------------------------------------------------
# C. Structure
# ---------------------------------------------------------------------------
TRIVIAL_LINE_RE = re.compile(r"^[\s{}()\[\];,.:'\"`<>/*#=+\-|\\]*$")
TRIVIAL_WORDS = {
    "end", "else", "else:", "try:", "finally:", "pass", "break;", "break", "continue;", "return;",
    "return", "default:", "});", "})", "fi", "done", "esac", "@override", "#endif", "endif",
    "*/", "/**", "*", "return nil", "return err", "return null;", "return true;", "return false;",
    "} else {", "if err != nil {", "</div>", "<div>", "\"\"\"", "'''", "@Override", "else {", "};",
    "return self", "return None", "Ok(())", "return Ok(())", "None", "null", "#[test]", "#[derive(Debug)]",
}
BOILERPLATE_RE = re.compile(
    r"^(?:Signed-off-by|Change-Id|git-svn-id|Reviewed-on|Reviewed-by|Tested-by|Acked-by|Cc|Co-authored-by|"
    r"Former-commit-id|Conflicts):|share on (?:whatsapp|facebook|twitter)|cookie policy|all rights reserved"
    r"|subscribe to our newsletter|\[skip ci\]|\[ci skip\]",
    re.I | re.M)
URL_RE = re.compile(r"https?://\S+")


def _is_trivial(line: str) -> bool:
    s = line.strip()
    return len(s) < 8 or bool(TRIVIAL_LINE_RE.match(s)) or s in TRIVIAL_WORDS


def detect_structure(doc: Doc) -> list[Finding]:
    out = []
    for s in doc.sections:
        lines = s.body.split("\n")
        offsets, pos = [], s.start
        for line in lines:
            offsets.append(pos)
            pos += len(line) + 1
        counts = Counter(line.strip() for line in lines if not _is_trivial(line))
        run_best, run_line = 1, -1
        run = 1
        for i in range(1, len(lines)):
            if lines[i].strip() and lines[i].strip() == lines[i - 1].strip() and not _is_trivial(lines[i]):
                run += 1
                if run > run_best:
                    run_best, run_line = run, i - run + 1
            else:
                run = 1
        if run_best >= 3:
            off = offsets[run_line]
            out.append(_make(doc, "C2", HIGH,
                             f"The same line repeats {run_best} times in a row inside '{s.name}'. "
                             "Consecutive identical non-trivial lines are a copy/scrape artefact, not normal code.",
                             off, off + len(lines[run_line])))
            continue
        rep = [(line, n) for line, n in counts.items() if n >= 5]
        if rep:
            line, n = max(rep, key=lambda x: x[1])
            idx = next(i for i, l in enumerate(lines) if l.strip() == line)
            off = offsets[idx]
            share = n / max(1, len([l for l in lines if l.strip()]))
            conf = MEDIUM if share >= 0.15 else LOW
            out.append(_make(doc, "C2", conf,
                             f"Line repeats {n} times inside '{s.name}' ({share:.0%} of its lines). "
                             "Counted per section, so old/new copies do not count. Can be normal code "
                             "(repeated asserts, case labels); accept only if it looks like junk.",
                             off, off + len(lines[idx])))

    if doc.is_commit:
        new = doc.section("new contents")
        old = doc.section("old contents")
        out += _detect_truncation(doc, new)
        if old and new and old.body.strip() and new.body.strip():
            a, b = _token_set(old.body), _token_set(new.body)
            if len(a) > 20 and len(b) > 20:
                jac = len(a & b) / len(a | b)
                if jac < 0.05:
                    out.append(_make(doc, "C4", MEDIUM,
                                     f"Old and new contents share only {jac:.0%} of their tokens: the 'diff' may be "
                                     "two unrelated files joined. False positive for full rewrites.",
                                     new.start, new.start + 1, evidence="old/new token Jaccard "
                                     f"{jac:.2f}"))
        if len(SECTION_RE.findall(doc.text)) > 4:
            out.append(_make(doc, "C4", HIGH, "More than one set of commit section headers: two records joined.",
                             -1, -1, evidence="extra ### section headers"))
        msg = doc.section("commit message")
        if msg:
            bp = list(BOILERPLATE_RE.finditer(msg.body))
            if bp:
                m = bp[0]
                out.append(_make(doc, "C1", LOW,
                                 f"{len(bp)} boilerplate trailer(s) in the commit message (Signed-off-by, Change-Id, "
                                 "git-svn-id...). Tool metadata, not natural text. Low priority; often left as is.",
                                 msg.start + m.start(), msg.start + m.end()))
            urls = URL_RE.findall(msg.body)
            if len(urls) >= 5:
                out.append(_make(doc, "C5", MEDIUM, f"Commit message holds {len(urls)} URLs: a list of links.",
                                 msg.start, msg.start + 1, evidence=f"{len(urls)} URLs"))
    else:
        out += _detect_truncation(doc, doc.sections[-1])
        bp = list(BOILERPLATE_RE.finditer(doc.text))
        if bp:
            m = bp[0]
            out.append(_make(doc, "C1", MEDIUM, f"{len(bp)} boilerplate phrase(s).", m.start(), m.end()))
    return out


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text))


BRACE_LANGS = {"JavaScript", "TypeScript", "Java", "C++", "C", "Go", "Rust", "C#", "Kotlin", "Swift", "PHP"}


def _strip_strings_comments(code: str, lang: str) -> str:
    """Blank out strings and comments in one left-to-right pass (so '#' or '/*' inside a string is ignored)."""
    if lang == "Python":
        rx = r'(?s)""".*?(?:"""|$)|\'\'\'.*?(?:\'\'\'|$)|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|#[^\n]*'
    elif lang == "Rust":
        # 'a lifetimes and b'x' bytes make single-quote strings unreliable; only char literals of one char.
        rx = r"(?s)/\*.*?(?:\*/|$)|//[^\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\\n])'"
    else:
        rx = r"(?s)/\*.*?(?:\*/|$)|//[^\n]*|`(?:\\.|[^`\\])*`|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'"
    return re.sub(rx, lambda m: "/*OPEN*/" if m.group(0).startswith("/*") and not m.group(0).endswith("*/")
                  else ('"""OPEN"""' if m.group(0)[:3] in ('"""', "\'\'\'") and len(m.group(0)) >= 3
                        and not (len(m.group(0)) >= 6 and m.group(0)[-3:] == m.group(0)[:3]) else '""'), code)


def _detect_truncation(doc: Doc, s: Section | None) -> list[Finding]:
    if not s or not s.body.strip():
        return []
    body = s.body.rstrip()
    end_off = s.start + len(body)
    last = body.split("\n")[-1]
    lang = doc.lang_label
    if doc.is_commit or lang in BRACE_LANGS or lang == "Python":
        stripped = _strip_strings_comments(body, lang)
        if lang == "Python":
            if '"""OPEN"""' in stripped:
                return [_make(doc, "C3", HIGH, f"Unterminated triple-quoted string at the end of '{s.name}': "
                              "the file stops in the middle.", end_off - len(last), end_off)]
            if re.search(r"[(\[{,\\]\s*$", stripped.rstrip()) or re.search(r"^\s*(?:def|class|if|for|while|elif|else|try|with)\b[^\n]*:\s*$", stripped.rstrip().split("\n")[-1]):
                return [_make(doc, "C3", MEDIUM, f"'{s.name}' ends on an open bracket, comma or block header "
                              "(':' with no body): looks cut.", end_off - len(last), end_off)]
            opens = sum(stripped.count(c) for c in "([{") - sum(stripped.count(c) for c in ")]}")
        else:
            opens = stripped.count("{") - stripped.count("}")
            if "/*OPEN*/" in stripped:
                return [_make(doc, "C3", HIGH, f"Unclosed /* comment at the end of '{s.name}'.",
                              end_off - len(last), end_off)]
        old = doc.section("old contents")
        old_balanced = True
        if old is not None and old is not s and old.body.strip():
            so = _strip_strings_comments(old.body, lang)
            old_balanced = (so.count("{") - so.count("}")) == 0 if lang != "Python" else True
        if opens >= 2 or (opens >= 1 and old_balanced and lang != "Python"):
            return [_make(doc, "C3", MEDIUM if opens >= 2 else LOW,
                          f"'{s.name}' has {opens} more opening than closing bracket(s) after removing strings and "
                          "comments: may stop mid-block. Check the last lines; macros/templates can confuse the count.",
                          end_off - len(last), end_off)]
    elif not re.search(r"[.!?\u0964\u0965\"')\]]\s*$", body):
        return [_make(doc, "C3", LOW, "Text does not end with sentence punctuation: may be cut.",
                      end_off - len(last), end_off)]
    return []


# ---------------------------------------------------------------------------
# D. Language
# ---------------------------------------------------------------------------
EXT_LANG = {
    ".py": "Python", ".pyw": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".mts": "TypeScript",
    ".java": "Java", ".go": "Go", ".rs": "Rust",
    ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++", ".hxx": "C++", ".h": "C++", ".c++": "C++",
    ".c": "C", ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".kt": "Kotlin", ".swift": "Swift", ".scala": "Scala",
}
NOT_CODE_EXTS = {".md", ".markdown", ".rst", ".txt", ".json", ".yml", ".yaml", ".csv", ".tsv", ".log", ".lock",
                 ".xml", ".html", ".htm", ".cfg", ".ini", ".toml", ".svg", ".png", ".jpg", ".gif", ".pdf", ".map"}
NOT_CODE_NAMES = {"readme", "license", "licence", "changelog", "authors", "contributors", "notice", "copying",
                  "makefile", "dockerfile", ".gitignore", "package-lock.json", "yarn.lock", "cargo.lock", "go.sum"}


def _file_path(doc: Doc) -> str:
    fp = doc.meta.get("new_file") or doc.meta.get("old_file") or ""
    if not fp:
        s = doc.section("file path")
        fp = s.body.strip() if s else ""
    return fp


def _file_ext(doc: Doc) -> str:
    return os.path.splitext(_file_path(doc))[1].lower()


def detect_language(doc: Doc) -> list[Finding]:
    out = []
    if not doc.is_commit:
        return out
    fp = _file_path(doc)
    ext = _file_ext(doc)
    base = os.path.basename(fp).lower()
    fp_sec = doc.section("file path")
    fp_off = fp_sec.start if fp_sec else -1
    fp_end = fp_off + len(fp) if fp_sec else -1
    expected = EXT_LANG.get(ext)
    if base in NOT_CODE_NAMES or ext in NOT_CODE_EXTS or base.startswith("readme"):
        out.append(_make(doc, "E6", HIGH, f"File '{base}' is not source code ({ext or 'no extension'}) but the pack "
                         f"labels it {doc.lang_label}.", fp_off, fp_end))
    elif expected and doc.lang_label and expected != doc.lang_label:
        out.append(_make(doc, "D1", HIGH, f"Extension {ext} means {expected}, label says {doc.lang_label}.",
                         fp_off, fp_end))
    elif not expected and ext:
        out.append(_make(doc, "D1", LOW, f"Unknown extension {ext} for label {doc.lang_label}; check the language.",
                         fp_off, fp_end))

    msg = doc.section("commit message")
    if msg and msg.body.strip():
        letters = [c for c in msg.body if c.isalpha()]
        non_latin = [c for c in letters if ord(c) > 0x24F]
        if letters and len(non_latin) / len(letters) > 0.3:
            scripts = Counter(_script(c) or unicodedata.name(c, "?").split()[0] for c in non_latin)
            out.append(_make(doc, "D3", LOW,
                             f"Commit message is mostly non-Latin ({scripts.most_common(1)[0][0]}) while code is "
                             "English. Natural for many projects: label it, do not judge it.",
                             msg.start, msg.start + min(40, len(msg.body))))
    return out


# ---------------------------------------------------------------------------
# E. Code
# ---------------------------------------------------------------------------
GENERATED_RE = re.compile(r"auto[- ]?generated|(?:generated|this) file[^\n]{0,40}do not edit|@generated\b|"
                          r"code generated .* DO NOT EDIT", re.I)
GENERATED_WEAK_RE = re.compile(r"\bgenerated (?:by|with|from)\b|@Generated\b(?!Value)|\bdo not (?:edit|modify)\b", re.I)
MINIFIED_NAME_RE = re.compile(r"\.min\.(?:js|css)$|\.bundle\.js$|-lock\.json$|\.pb\.(?:go|cc|h)$|_pb2\.py$", re.I)
BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{120,}={0,2}(?![A-Za-z0-9+/=])")
HEXBLOB_RE = re.compile(r"(?:0x[0-9A-Fa-f]{2},\s*){48,}|(?<![0-9A-Fa-f])[0-9A-Fa-f]{160,}(?![0-9A-Fa-f])")
NUMARRAY_RE = re.compile(r"(?:-?\d+(?:\.\d+)?\s*,\s*){64,}")
DATAURI_RE = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,", re.I)

SECRET_PATTERNS = (
    ("AWS access key", HIGH, re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")),
    ("private key block", HIGH, re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----")),
    ("GitHub token", HIGH, re.compile(r"(?<![A-Za-z0-9_])(?:gh[opusr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("Slack token", HIGH, re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("Google API key", HIGH, re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Stripe live key", HIGH, re.compile(r"(?:sk|rk)_live_[0-9A-Za-z]{20,}")),
    ("JWT", MEDIUM, re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("URL with password", MEDIUM, re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@]{1,64}:[^/\s:@]{3,64}@[^\s/]+", re.I)),
)
ASSIGNED_SECRET_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_]*(?:password|passwd|pwd|secret|api_?key|access_?key|auth_?token|token|client_secret)[A-Za-z0-9_]*)"
    r"[\"']?\s*(?:=|:|=>|:=)\s*[\"']([^\"'\s]{6,})[\"']")
PLACEHOLDER_RE = re.compile(
    r"(?i)^(?:x+|\*+|\.+|<[^>]*>|\$\{?[^}]*\}?|%\(?\w*\)?s?|changeme|change_me|password\d*|secret\d*|"
    r"test\w*|dummy\w*|example\w*|sample\w*|your[_-]?\w*|my[_-]?\w*|foo\w*|bar\w*|fake\w*|placeholder|"
    r"none|null|undefined|todo|xxx\w*|abc\w*|123\w*|qwerty|admin|root|user\w*|pass\w*|env\w*|process\.env.*)$")


def _entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(n / len(s) * math.log2(n / len(s)) for n in counts.values())


def detect_code(doc: Doc) -> list[Finding]:
    out = []
    fp = _file_path(doc)
    for s in doc.code_sections():
        body = s.body
        if not body.strip():
            continue
        lines = body.split("\n")
        longest = max(lines, key=len)
        avg = sum(len(l) for l in lines) / max(1, len(lines))
        if len(longest) >= 1000 or (avg > 200 and len(lines) < 10):
            off = s.start + body.find(longest)
            out.append(_make(doc, "E1", HIGH, f"'{s.name}' looks minified: longest line {len(longest)} chars, "
                             f"average {avg:.0f}.", off, off + 80, evidence=f"line of {len(longest)} chars"))
        gen = GENERATED_RE.search(body[:1500])
        weak = GENERATED_WEAK_RE.search(body[:1500])
        if gen:
            out.append(_make(doc, "E1", HIGH, f"Generated-code marker '{gen.group(0)}' near the top of '{s.name}'.",
                             s.start + gen.start(), s.start + gen.end()))
        elif weak:
            out.append(_make(doc, "E1", MEDIUM,
                             f"Weak generated-code marker '{weak.group(0)}' in '{s.name}'. Often an IDE template "
                             "comment ('Generated by Django 1.9', '@Generated' annotation): accept only if the "
                             "whole file is tool output.", s.start + weak.start(), s.start + weak.end()))
        if MINIFIED_NAME_RE.search(fp) and s.name == "new contents":
            out.append(_make(doc, "E1", HIGH, f"File name '{os.path.basename(fp)}' is a minified/generated artefact.",
                             -1, -1, evidence=os.path.basename(fp)))

        blobs = []
        for rx, kind in ((DATAURI_RE, "data: URI"), (BASE64_RE, "base64"), (HEXBLOB_RE, "hex blob"),
                         (NUMARRAY_RE, "long number array")):
            for m in rx.finditer(body):
                blob = m.group(0)
                if kind == "base64" and (not re.search(r"\d", blob) or not re.search(r"[A-Z]", blob)
                                         or not re.search(r"[a-z]", blob)):
                    continue  # long identifiers/paths, not base64
                blobs.append((m, kind))
        if blobs:
            m, kind = blobs[0]
            total = sum(len(b.group(0)) for b, _ in blobs)
            share = total / max(1, len(body))
            out.append(_make(doc, "E2", HIGH if share > 0.3 else MEDIUM,
                             f"{len(blobs)} data blob(s) ({kind}) in '{s.name}', {total} chars = {share:.0%} of the "
                             "section. No value for code learning; >30% usually means drop.",
                             s.start + m.start(), s.start + m.start() + min(80, len(m.group(0))),
                             evidence=f"{kind} of {len(m.group(0))} chars"))

    out += _detect_secrets(doc)
    out += _detect_licence(doc)
    return out


def _detect_secrets(doc: Doc) -> list[Finding]:
    out = []
    seen: set[str] = set()
    for kind, conf, rx in SECRET_PATTERNS:
        matches = list(rx.finditer(doc.text))
        if matches:
            if kind == "URL with password" and all(re.search(r"(?i)user:pass|:password@|example", m.group(0)) for m in matches):
                continue
            m = matches[0]
            offs = ",".join(str(x.start()) for x in matches[:5])
            out.append(Finding("E3", conf, f"{len(matches)} {kind} pattern(s). Value is not shown in evidence; "
                               "inspect in the viewer only. Test fixtures with fake keys are a false positive.",
                               f"{kind} at offset {offs}", m.start(), m.end(), doc.section_at(m.start()),
                               doc.line_of(m.start()), "", True))
            seen.add(kind)
    assigned = []
    for m in ASSIGNED_SECRET_RE.finditer(doc.text):
        value = m.group(2)
        if PLACEHOLDER_RE.match(value) or len(value) < 12 or _entropy(value) < 3.5:
            continue
        if re.search(r"[(){}\[\]]|\.\w+\(", value) or "/" in value and " " not in value and value.count("/") > 1:
            continue
        assigned.append(m)
    if assigned:
        m = assigned[0]
        offs = ",".join(str(x.start(2)) for x in assigned[:5])
        out.append(Finding("E3", MEDIUM, f"{len(assigned)} high-entropy value(s) assigned to a secret-like name "
                           f"('{m.group(1)}'). Could be a real credential or a test fixture; check the context.",
                           f"secret-like assignment at offset {offs}", m.start(2), m.end(2),
                           doc.section_at(m.start(2)), doc.line_of(m.start(2)), "", True))
    return out


LICENCE_RE = re.compile(r"Licensed under|Permission is hereby granted|GNU (?:Lesser |Affero )?General Public License|"
                        r"SPDX-License-Identifier|Apache License|Mozilla Public License|Redistribution and use in source|"
                        r"THE SOFTWARE IS PROVIDED|All rights reserved", re.I)


def _leading_comment(body: str) -> tuple[int, str]:
    """Return (offset, text) of the first comment block of a source file."""
    m = re.match(r"\s*(?:#![^\n]*\n)?\s*((?:/\*.*?\*/)|(?:(?:[ \t]*(?://|#)[^\n]*\n)+)|(?:\"\"\".*?\"\"\"))", body, re.S)
    if not m:
        return -1, ""
    return m.start(1), m.group(1)


def licence_key(doc: Doc) -> tuple[str, int, int] | None:
    """Fingerprint of the licence header in new contents, for the pack-level E4 rule."""
    s = doc.section("new contents") if doc.is_commit else doc.sections[-1]
    if not s:
        return None
    off, block = _leading_comment(s.body)
    if off < 0 or not LICENCE_RE.search(block):
        return None
    norm = re.sub(r"(?i)copyright[^\n]*|\d{4}", "", block)
    norm = re.sub(r"[\W_]+", " ", norm).strip().lower()
    return hashlib.sha1(norm.encode()).hexdigest(), s.start + off, s.start + off + len(block)


def _detect_licence(doc: Doc) -> list[Finding]:
    key = licence_key(doc)
    if not key:
        return []
    _, start, end = key
    n_lines = doc.text[start:end].count("\n") + 1
    if n_lines < 10:
        return []
    return [_make(doc, "E4", LOW, f"{n_lines}-line licence header at the top of the file. E4 is about the same "
                  "header repeated across many files; the pack-level rule raises this to high when the same text "
                  "appears in 3+ documents.", start, end, evidence=f"{n_lines}-line licence header")]


# ---------------------------------------------------------------------------
# F. Safety, privacy and contamination
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
PLACEHOLDER_EMAIL_RE = re.compile(
    r"(?i)@(?:[\w-]+\.)*(?:example\.(?:com|org|net)|example|test|invalid|localhost|domain\.com|email\.com|"
    r"mail\.com|foo\.\w+|bar\.\w+|acme\.\w+|company\.com|yourdomain\.\w+|host\.com|server\.com)$"
    r"|^(?:noreply|no-reply|user|test|foo|bar|john\.?doe|jane\.?doe|someone|name|me|you|admin|info|support|"
    r"hello|contact|email|mail)@")
ORG_EMAIL_RE = re.compile(r"(?i)^(?:security|support|info|help|contact|team|dev|devs|maintainers?|bugs?|webmaster|"
                          r"abuse|privacy|legal|press|sales|hello|noreply|no-reply|admin|root|postmaster|git)@")
AUTHOR_CTX_RE = re.compile(r"(?i)author|copyright|\(c\)|signed-off-by|co-authored-by|maintainer|contributors?|"
                           r"created by|written by|@author|<[^>]*$")
PHONE_RE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]\d{3,4}(?:[ .-]\d{2,4})?(?![\w.])")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
BENCH_RE = re.compile(r"HumanEval|\bMBPP\b|def check\(candidate\)|assert candidate\(|LeetCode|leetcode\.com/problems|"
                      r"Project Euler|projecteuler\.net|codeforces\.com|APPS dataset|GSM8K|\bMMLU\b", re.I)


def detect_safety(doc: Doc) -> list[Finding]:
    out = []
    text = doc.text
    real, placeholder = [], 0
    for m in EMAIL_RE.finditer(text):
        email = m.group(0)
        if PLACEHOLDER_EMAIL_RE.search(email):
            placeholder += 1
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        ctx = text[line_start:m.start()]
        if ORG_EMAIL_RE.match(email):
            kind, conf = "role/organisation email", LOW
        elif "users.noreply.github.com" in email:
            kind, conf = "GitHub noreply email (still a handle)", LOW
        elif AUTHOR_CTX_RE.search(ctx):
            kind, conf = "author/contributor email", HIGH
        else:
            kind, conf = "email", MEDIUM
        real.append((m, kind, conf))
    if real:
        best = min(real, key=lambda r: CONFIDENCE_RANK[r[2]])
        kinds = Counter(k for _, k, _ in real)
        offs = ",".join(str(m.start()) for m, _, _ in real[:6])
        old_new_note = (" In commit packs the same email usually appears twice (old and new contents)."
                        if doc.is_commit else "")
        out.append(Finding("F1", best[2],
                           f"{len(real)} email(s): " + ", ".join(f"{n} {k}" for k, n in kinds.items())
                           + (f"; {placeholder} placeholder email(s) ignored" if placeholder else "")
                           + ". Author emails in headers identify private people." + old_new_note,
                           f"{best[1]} at offset {offs}", best[0].start(), best[0].end(),
                           doc.section_at(best[0].start()), doc.line_of(best[0].start()), "", True))

    phones = []
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        line_start = text.rfind("\n", 0, m.start()) + 1
        ctx = text[line_start:m.start()].lower()
        if 10 <= len(digits) <= 13 and re.search(r"phone|tel|mobile|cell|contact|whatsapp|call", ctx):
            phones.append(m)
    if phones:
        m = phones[0]
        out.append(Finding("F1", MEDIUM, f"{len(phones)} phone-number-like value(s) next to a phone/contact word.",
                           f"phone number at offset {','.join(str(p.start()) for p in phones[:5])}",
                           m.start(), m.end(), doc.section_at(m.start()), doc.line_of(m.start()), "", True))

    public_ips = []
    for m in IPV4_RE.finditer(text):
        parts = [int(p) for p in m.group(0).split(".")]
        if any(p > 255 for p in parts):
            continue
        if parts[0] in (0, 10, 127, 255) or parts[:2] == [192, 168] or (parts[0] == 172 and 16 <= parts[1] <= 31) \
                or parts[0] >= 224 or parts == [8, 8, 8, 8] or parts[:3] == [1, 1, 1]:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        if re.search(r"(?i)version|v\d|semver", text[line_start:m.start()]):
            continue
        public_ips.append(m)
    if public_ips:
        m = public_ips[0]
        out.append(Finding("F1", LOW, f"{len(public_ips)} public IPv4 address(es). Usually a server, not a person; "
                           "accept only if it identifies a private individual.",
                           f"public IP at offset {','.join(str(p.start()) for p in public_ips[:5])}",
                           m.start(), m.end(), doc.section_at(m.start()), doc.line_of(m.start()), "", True))

    bench = list(BENCH_RE.finditer(text))
    if bench:
        m = bench[0]
        out.append(_make(doc, "F2", MEDIUM, f"{len(bench)} benchmark marker(s) ('{m.group(0)}'): the file may contain "
                         "a benchmark problem or solution. Compare with the benchmark before dropping.",
                         m.start(), m.end()))
    return out


# ---------------------------------------------------------------------------
# G. Damage from earlier cleaning (Indic)
# ---------------------------------------------------------------------------
INDIC_MARK_RE = re.compile(r"(?<=[\u0900-\u0DFF]) (?=[\u093e-\u094d\u0962\u0963\u09be-\u09cd\u0abe-\u0acd"
                           r"\u0b3e-\u0b4d\u0bbe-\u0bcd\u0c3e-\u0c4d\u0cbe-\u0ccd\u0d3e-\u0d4d\u0a3e-\u0a4d])")


def detect_cleaning_damage(doc: Doc) -> list[Finding]:
    hits = list(INDIC_MARK_RE.finditer(doc.text))
    if not hits:
        return []
    m = hits[0]
    span = doc.text[max(0, m.start() - 1): m.end() + 1]
    return [_make(doc, "G1", HIGH, f"{len(hits)} space(s) before a vowel sign/virama: a word was cut at the matra.",
                  m.start() - 1, m.end() + 1, hex_value=_hex(span))]


# ---------------------------------------------------------------------------
# Document + pack level
# ---------------------------------------------------------------------------
DOC_DETECTORS = (detect_characters, detect_markup, detect_structure, detect_language, detect_code,
                 detect_safety, detect_cleaning_damage)


def scan_document(doc: Doc) -> list[Finding]:
    findings: list[Finding] = []
    for detector in DOC_DETECTORS:
        findings.extend(detector(doc))
    return _merge_old_new(findings)


def _merge_old_new(findings: list[Finding]) -> list[Finding]:
    """A commit's old and new contents are near-copies: report a per-section hit once, on new contents."""
    # Keyed on code + hex so two different characters under the same code (e.g. A1 BOM vs NBSP) both survive.
    in_new = {(f.code, f.hex_value) for f in findings if f.section == "new contents"}
    out = []
    for f in findings:
        if f.section == "old contents" and (f.code, f.hex_value) in in_new:
            continue
        if f.section == "new contents" and any(g.section == "old contents" and (g.code, g.hex_value) == (f.code, f.hex_value)
                                               for g in findings):
            f.reason += " (Also present in old contents.)"
        out.append(f)
    return out


def _shingles(text: str, k: int = 5) -> set[int]:
    toks = re.findall(r"\w+|[^\w\s]", text)
    return {hash(" ".join(toks[i:i + k])) for i in range(max(1, len(toks) - k + 1))}


def scan_records(records: list[dict]) -> list[dict]:
    """Scan a whole pack. Returns one dict per document with its findings."""
    docs = [parse_doc(i, r) for i, r in enumerate(records)]
    per_doc = [scan_document(d) for d in docs]

    # H1 exact duplicates (full text, then the code payload alone).
    by_text: defaultdict[str, list[int]] = defaultdict(list)
    by_new: defaultdict[str, list[int]] = defaultdict(list)
    for d in docs:
        by_text[hashlib.sha256(d.text.encode()).hexdigest()].append(d.index)
        new = d.section("new contents")
        if new and len(new.body.strip()) > 50:
            by_new[hashlib.sha256(new.body.strip().encode()).hexdigest()].append(d.index)
    exact_groups = [g for g in by_text.values() if len(g) > 1]
    for g in exact_groups:
        for i in g:
            others = ",".join(str(j + 1) for j in g if j != i)
            per_doc[i].append(Finding("H1", HIGH, "Byte-identical document elsewhere in the pack. Keep one copy.",
                                      f"identical to record(s) {others}"))
    exact_ids = {i for g in exact_groups for i in g}
    for g in by_new.values():
        g = [i for i in g if i not in exact_ids]
        if len(g) > 1:
            for i in g:
                others = ",".join(str(j + 1) for j in g if j != i)
                per_doc[i].append(Finding("H2", HIGH, "Same new file contents as another record, different commit "
                                          "message or path (same change committed twice, or a fork).",
                                          f"same new contents as record(s) {others}"))

    # H2 near duplicates: 5-token shingle Jaccard on the code payload.
    sigs = []
    for d in docs:
        s = d.section("new contents") or d.sections[-1]
        sigs.append(_shingles(s.body) if len(s.body) > 200 else set())
    flagged = {(i, j) for g in by_new.values() for i in g for j in g}
    for i in range(len(docs)):
        a = sigs[i]
        if not a:
            continue
        for j in range(i + 1, len(docs)):
            b = sigs[j]
            if not b or (i, j) in flagged or min(len(a), len(b)) / max(len(a), len(b)) < 0.6:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= 0.7:
                conf = HIGH if jac >= 0.9 else MEDIUM
                for x, y in ((i, j), (j, i)):
                    per_doc[x].append(Finding("H2", conf, f"Code payload overlaps {jac:.0%} (5-token shingles) with "
                                              f"record {y + 1} ({docs[y].doc_id}). Same file at another commit, "
                                              "or copied code.", f"near-duplicate of record {y + 1}, Jaccard {jac:.2f}"))

    # E4 licence header repeated across documents.
    lic: defaultdict[str, list[int]] = defaultdict(list)
    for d in docs:
        key = licence_key(d)
        if key:
            lic[key[0]].append(d.index)
    for key, g in lic.items():
        if len(g) < 3:
            continue
        for i in g:
            d = docs[i]
            _, start, end = licence_key(d)
            n_lines = d.text[start:end].count("\n") + 1
            per_doc[i] = [f for f in per_doc[i] if f.code != "E4"]
            per_doc[i].append(_make(d, "E4", HIGH if n_lines >= 5 else MEDIUM,
                                    f"The same licence header appears in {len(g)} documents of this pack "
                                    f"({n_lines} lines). Strip the repeated header.", start, end,
                                    evidence=f"{n_lines}-line licence header shared by {len(g)} docs"))

    results = []
    for d, findings in zip(docs, per_doc):
        findings.sort(key=lambda f: (CONFIDENCE_RANK[f.confidence], f.code, f.start))
        context = commit_context(d)
        checks = run_checks(d, findings)
        verdict = suggest_verdict(findings)
        results.append({"index": d.index, "doc_id": d.doc_id, "lang_label": d.lang_label,
                        "chars": len(d.text), "findings": findings, "suggested_verdict": verdict,
                        "context": context, "checks": checks,
                        "why": explain_verdict(verdict, checks, findings)})
    return results


# ---------------------------------------------------------------------------
# Commit context: what changed between old and new contents
# ---------------------------------------------------------------------------
def _line_spans(s: Section) -> list[tuple[int, int, str]]:
    out, pos = [], s.start
    for line in s.body.split("\n"):
        out.append((pos, pos + len(line), line))
        pos += len(line) + 1
    while out and not out[-1][2].strip():
        out.pop()
    return out


def commit_context(doc: Doc) -> dict:
    """Old vs new contents of a commit. The repeated text between them is the commit layout, not a duplicate."""
    if not doc.is_commit:
        return {}
    old, new = doc.section("old contents"), doc.section("new contents")
    old_l = _line_spans(old) if old else []
    new_l = _line_spans(new) if new else []
    if not old_l:
        return {"kind": "new file", "old_lines": 0, "new_lines": len(new_l), "added": len(new_l), "removed": 0,
                "unchanged": 0, "similarity": 0.0, "ranges": [],
                "summary": f"New file: old contents are empty, all {len(new_l)} lines are added. "
                           "Nothing in this document is repeated by the commit layout."}
    sm = difflib.SequenceMatcher(None, [l[2] for l in old_l], [l[2] for l in new_l], autojunk=False)
    ranges, added, removed, same = [], 0, 0, 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            same += i2 - i1
            continue
        if i2 > i1:
            removed += i2 - i1
            ranges.append({"kind": "del", "start": old_l[i1][0], "end": old_l[i2 - 1][1], "line": doc.line_of(old_l[i1][0])})
        if j2 > j1:
            added += j2 - j1
            ranges.append({"kind": "add", "start": new_l[j1][0], "end": new_l[j2 - 1][1], "line": doc.line_of(new_l[j1][0])})
    sim = sm.ratio()
    kind = "small edit" if sim >= 0.8 else ("rewrite" if sim < 0.3 else "edit")
    summary = (f"{kind.capitalize()}: {same} of {len(old_l)} old lines are unchanged in new contents "
               f"({sim:.0%} similar), +{added} / -{removed} lines. ")
    if same:
        summary += ("The unchanged lines appear twice by design (old copy + new copy). They are NOT a duplicate "
                    "(H1/H2), NOT repeated lines (C2), and a licence header in both copies is counted once (E4). "
                    "Read the changed lines (green = added, red = removed) and judge the new contents.")
    return {"kind": kind, "old_lines": len(old_l), "new_lines": len(new_l), "added": added, "removed": removed,
            "unchanged": same, "similarity": round(sim, 3), "ranges": ranges, "summary": summary}


# ---------------------------------------------------------------------------
# Checklist: what was checked for every disease code, and the result
# ---------------------------------------------------------------------------
MANUAL = None
CHECKS = {
    "A1": "Every character scanned for ZWSP, BOM, soft hyphen, LRM/RLM, bidi embeddings and isolates, any other "
          "format/private-use/unassigned code point, and look-alike spaces (NBSP, U+2000-U+200A, U+202F, U+3000, "
          "line/paragraph separators); ZWJ/ZWNJ flagged only when not next to an Indic letter.",
    "A2": "C0 controls (except tab/newline) and C1 controls U+007F-U+009F; CR checked as CRLF (fix) vs lone CR.",
    "A3": "Mojibake byte-pair patterns (Ã©, Â , â€™, ðŸ, à¤…) outside code sections.",
    "A4": "Count of U+FFFD replacement characters.",
    "A5": "Every word tested with unicodedata.is_normalized('NFC').",
    "A6": "Words that mix letters from two scripts (Latin, Cyrillic, Greek, Indic, Arabic).",
    "B1": "HTML entities in the commit message (and in code only as a low hint, since &lt; etc. are normal there).",
    "B2": "HTML tags (<div>, <br/>, <p> ...) in the commit message.",
    "B3": "<|...|>, [INST], <<SYS>>, '### Assistant:' anywhere; '<s>'/'<think>' case-sensitive.",
    "B4": "{{ }}, ${ }, {% %} in the commit message ('undefined'/'NaN' ignored in commits: they talk about code).",
    "B5": "Odd number of ``` fences in the commit message.",
    "C1": "Signed-off-by / Change-Id / git-svn-id trailers, share/cookie/newsletter phrases.",
    "C2": "Same non-trivial line 3+ times in a row, or 5+ times, counted inside one section only "
          "(old/new copies never count as repeats).",
    "C3": "End of new contents: unclosed /* or triple quote, more { than } after removing strings and comments, "
          "trailing open bracket or ':' with no body.",
    "C4": "Old vs new share <5% of identifiers (two unrelated files), or a second set of ### section headers.",
    "C5": "5+ URLs in the commit message.",
    "D1": "File extension of the path vs lang_label.",
    "D2": MANUAL, "D4": MANUAL, "D5": MANUAL,
    "D3": "Commit message with more than 30% non-Latin letters.",
    "E1": "Lines of 1000+ chars or average 200+, generated-file markers in the first 1500 chars, .min.js/_pb2/.pb.go names.",
    "E2": "base64 runs of 120+ chars (mixed case + digits), hex blobs, 64+ number arrays, data: URIs.",
    "E3": "AWS/GitHub/Slack/Google/Stripe keys, private key blocks, JWTs, passwords in URLs, high-entropy values "
          "assigned to password/token/secret names (placeholders ignored).",
    "E4": "Leading comment block with licence words, fingerprinted and compared with every doc in the pack.",
    "E5": "Mojibake patterns inside old/new contents.",
    "E6": "File name/extension is README, LICENSE, JSON, YAML, lock file, CSV, log ...",
    "F1": "Emails (example/test/placeholder addresses ignored), phone numbers next to phone/tel words, public IPv4.",
    "F2": "HumanEval / MBPP / LeetCode / Project Euler / Codeforces / GSM8K / MMLU markers.",
    "F3": MANUAL,
    "G1": "Space between an Indic letter and a following vowel sign or virama.",
    "G2": MANUAL, "G3": MANUAL, "G4": MANUAL,
    "H1": "SHA-256 of the full text and of new contents compared with all other docs in the pack.",
    "H2": "5-token shingle Jaccard (>=0.7) of new contents vs every other doc in the pack. The doc's own old "
          "contents are never compared (that is the commit, not a duplicate).",
    "H3": MANUAL,
}
MANUAL_WHY = {
    "D2": "Romanised Indic: read the commit message and comments.",
    "D4": "Machine-translation quality needs a reader of the language.",
    "D5": "Translationese needs a native reader.",
    "F3": "Harmful content is not detected automatically: read it.",
    "G2": "Lost ZWJ/ZWNJ needs a visual check of conjuncts.",
    "G3": "Missing sections: compare with what the file should contain.",
    "G4": "Over-normalisation: look for Latin digits/NFKC forms in Indic text.",
    "H3": "Same content in another source or language: not checked across packs.",
}
INDIC_CODES = {"G1", "G2", "G3", "G4", "D2"}


def run_checks(doc: Doc, findings: list[Finding]) -> list[dict]:
    has_indic = any(_is_indic(c) for c in doc.text)
    by_code: defaultdict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_code[f.code].append(f)
    rows = []
    for code, how in CHECKS.items():
        hits = by_code.get(code, [])
        strong = [f for f in hits if f.confidence != LOW]
        if strong:
            status, result = "hit", f"{len(strong)} finding(s), best {min(strong, key=lambda f: CONFIDENCE_RANK[f.confidence]).confidence}"
        elif hits:
            status, result = "weak", f"{len(hits)} low-confidence hint(s) only: check, usually not a disease"
        elif code in INDIC_CODES and not has_indic:
            status, result = "na", "no Indic characters in this document"
        elif how is MANUAL:
            status, result = "manual", MANUAL_WHY[code]
        else:
            status, result = "pass", "nothing matched"
        rows.append({"code": code, "status": status, "checked": how or MANUAL_WHY[code], "result": result})
    return rows


def explain_verdict(verdict: str, checks: list[dict], findings: list[Finding]) -> str:
    n = Counter(c["status"] for c in checks)
    manual = [c["code"] for c in checks if c["status"] == "manual"]
    base = (f"{n['pass']} automated checks passed, {n['na']} not applicable, "
            f"{n['weak']} low hint(s), {n['hit']} hit(s). Still to read by eye: {', '.join(manual) or 'none'}.")
    if verdict == "clean?":
        weak = sorted({f.code for f in findings})
        return ("Looks clean because no medium/high rule fired"
                + (f" (low hints for {', '.join(weak)} are usually normal code)" if weak else "")
                + ". " + base + " Mark 'clean' only after reading the whole document.")
    hits = sorted({f.code for f in findings if f.confidence != LOW})
    if verdict == "drop?":
        return f"Suggested drop: a high-confidence drop-class finding ({', '.join(hits)}). " + base
    return f"Suggested fix: {', '.join(hits)} can be repaired in place. " + base


DROP_CODES = {"B3", "C4", "E1", "E2", "E6", "F2", "F3", "H1", "H2", "A4"}


def suggest_verdict(findings: list[Finding]) -> str:
    strong = [f for f in findings if f.confidence != LOW]
    if not strong:
        return "clean?"
    if any(f.code in DROP_CODES and f.confidence == HIGH for f in strong):
        return "drop?"
    return "fix?"


def label_from(findings: list[Finding]) -> dict[str, str]:
    """Collapse accepted findings into the label-sheet columns."""
    codes = list(dict.fromkeys(f.code for f in findings))
    evidence = "; ".join(f"{f.code}: {f.evidence}" for f in findings)
    hexes = list(dict.fromkeys(f.hex_value for f in findings
                               if f.hex_value and f.code[0] in "AG" and not f.sensitive))
    return {"diseases": ";".join(codes), "evidence": evidence, "hex": " ".join(hexes)}
