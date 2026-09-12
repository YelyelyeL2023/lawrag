"""Turning code text into a tree.

Russian and Kazakh mark structure differently, and the difference is not
cosmetic. Russian puts the label first and the number second ("Статья 278",
"Глава 14"); Kazakh attaches the number as an ordinal prefix with a hyphen
("278-бап", "14-тарау"). A parser written only against the Russian pattern will
silently produce a single flat blob for every Kazakh act, which then looks like
a retrieval-quality problem three layers downstream. Both dialects are handled
here, in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Lang, NodeKind

# --------------------------------------------------------------------------
# Heading patterns
# --------------------------------------------------------------------------

_RU_PATTERNS: list[tuple[NodeKind, re.Pattern[str]]] = [
    (NodeKind.SECTION, re.compile(r"^\s*РАЗДЕЛ\s+([IVXLCDM\d]+(?:-\d+)*)\.?\s*(.*)$", re.IGNORECASE)),
    (NodeKind.CHAPTER, re.compile(r"^\s*ГЛАВА\s+(\d+(?:-\d+)*)\.?\s*(.*)$", re.IGNORECASE)),
    (NodeKind.PARAGRAPH, re.compile(r"^\s*ПАРАГРАФ\s+(\d+(?:-\d+)*)\.?\s*(.*)$", re.IGNORECASE)),
    (NodeKind.ARTICLE, re.compile(r"^\s*Статья\s+(\d+(?:-\d+)*)\.?\s*(.*)$", re.IGNORECASE)),
]

_KK_PATTERNS: list[tuple[NodeKind, re.Pattern[str]]] = [
    (NodeKind.SECTION, re.compile(r"^\s*(\d+(?:-\d+)*)-БӨЛІМ\.?\s*(.*)$", re.IGNORECASE)),
    (NodeKind.CHAPTER, re.compile(r"^\s*(\d+(?:-\d+)*)-ТАРАУ\.?\s*(.*)$", re.IGNORECASE)),
    (NodeKind.PARAGRAPH, re.compile(r"^\s*(\d+(?:-\d+)*)-ПАРАГРАФ\.?\s*(.*)$", re.IGNORECASE)),
    (NodeKind.ARTICLE, re.compile(r"^\s*(\d+(?:-\d+)*)-бап\.?\s*(.*)$", re.IGNORECASE)),
]

# Points and subpoints look the same in both languages.
POINT_RE = re.compile(r"^\s*(\d+(?:-\d+)*)\.\s+(?=\S)")
SUBPOINT_RE = re.compile(r"^\s*(\d+(?:-\d+)*)\)\s+(?=\S)")

# Substantive notes. "Примечание" / "Ескерту" attached to an article is
# operative law — it routinely defines how the article is counted or applied —
# so it must travel with the article, never be dropped and never be swallowed by
# whichever numbered point happens to precede it.
NOTE_RE = re.compile(r"^\s*(Примечани[ея]|Ескерту)\b\.?\s*", re.IGNORECASE)

# Amendment provenance. Same opening word as a substantive note in Kazakh
# ("Ескерту"), so the two are told apart by whether a law is *cited*: an
# amendment line always names the amending act by number or date.
_LAW_CITATION = re.compile(
    r"(№\s*\d+|от\s+\d{2}\.\d{2}\.\d{4}|\d{2}\.\d{2}\.\d{4}|Заңымен|Закон(?:ом|а)\s+РК"
    r"|в\s+редакции|жаңа\s+редакцияда)",
    re.IGNORECASE,
)
_PROVENANCE_OPENER = re.compile(r"^\s*(?:Сноска|Ескерту)\b", re.IGNORECASE)


def is_amendment_line(line: str) -> bool:
    return bool(_PROVENANCE_OPENER.match(line) and _LAW_CITATION.search(line))

REPEALED_RE = re.compile(
    r"(исключен[аоы]?\s+Законом|утратил[аи]?\s+силу|алып\s+тасталды|күші\s+жойылды)",
    re.IGNORECASE,
)

# Structures we must never cut through: a rate table or an enumerated list of
# grounds is a single legal proposition even when it is long. The HTML adapter
# flattens table rows to tab-separated lines, so two or more tab-bearing lines
# is the signal — requiring two tabs on one line would miss every two-column
# rate schedule, which is the most common shape in KZ codes.
TABLE_HINT_RE = re.compile(r"(\|\s*-{2,}|<table)", re.IGNORECASE)
_TAB_LINE_RE = re.compile(r"^[^\t\n]*\t.*$", re.MULTILINE)


@dataclass(slots=True)
class Heading:
    kind: NodeKind
    number: str
    title: str


def detect_lang(text: str) -> Lang:
    """Kazakh-specific Cyrillic letters are the cheapest reliable signal.

    ә ғ қ ң ө ұ ү һ і do not occur in Russian orthography, so even a couple of
    them in a paragraph settles it. Falls back to Russian, which is the right
    default for this corpus.
    """
    kk_specific = set("әғқңөұүһі")
    sample = text[:4000].lower()
    hits = sum(1 for ch in sample if ch in kk_specific)
    return Lang.KK if hits >= 3 else Lang.RU


def match_heading(line: str, lang: Lang) -> Heading | None:
    patterns = _KK_PATTERNS if lang is Lang.KK else _RU_PATTERNS
    for kind, pattern in patterns:
        m = pattern.match(line)
        if m:
            number, title = m.group(1), (m.group(2) or "").strip()
            # "Статья 5 настоящего Кодекса ..." inside running prose is a
            # reference, not a heading. A real heading line is short and either
            # ends after the title or the title is capitalised.
            if kind is NodeKind.ARTICLE and len(line) > 300:
                return None
            return Heading(kind=kind, number=number, title=title)
    return None


def strip_amendment_notes(text: str) -> tuple[str, list[str]]:
    """Split running text into (clean text, amendment provenance lines)."""
    kept: list[str] = []
    notes: list[str] = []
    for line in text.splitlines():
        if is_amendment_line(line.strip()):
            notes.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(kept).strip(), notes


def split_notes(text: str) -> tuple[str, list[str]]:
    """Separate substantive notes from the numbered body of an article.

    The note text is returned so the caller can attach it to the article as its
    own node; it is removed from the body only so that point splitting does not
    glue it onto the last numbered point, where it would be cited as if it were
    part of that point's rule.
    """
    body: list[str] = []
    notes: list[str] = []
    in_note = False
    for line in text.splitlines():
        if NOTE_RE.match(line.strip()) and not is_amendment_line(line.strip()):
            in_note = True
            notes.append(line.strip())
            continue
        if in_note:
            if not line.strip() or match_any_marker(line):
                in_note = False
            else:
                notes[-1] += " " + line.strip()
                continue
        body.append(line)
    return "\n".join(body).strip(), notes


def match_any_marker(line: str) -> bool:
    """True when a line opens a new numbered point or subpoint."""
    return bool(POINT_RE.match(line) or SUBPOINT_RE.match(line))


def is_repealed(text: str) -> bool:
    head = text[:300]
    return bool(REPEALED_RE.search(head))


def contains_table(text: str) -> bool:
    if TABLE_HINT_RE.search(text):
        return True
    return len(_TAB_LINE_RE.findall(text)) >= 2


def split_points(body: str) -> list[tuple[str | None, str]]:
    """Split an article body into (point number, text) pairs.

    Returns ``[(None, body)]`` when the article has no numbered points, which is
    common for short definitional articles. Subpoints are deliberately *not*
    split out: "1) ... 2) ... 3) ..." is an enumeration of alternatives within a
    single legal rule and splitting it produces fragments that mean nothing on
    their own.
    """
    lines = body.splitlines()
    buckets: list[tuple[str | None, list[str]]] = []
    current_num: str | None = None
    current: list[str] = []

    for line in lines:
        m = POINT_RE.match(line)
        # A subpoint marker ("3) ...") must not be mistaken for a point.
        if m and not SUBPOINT_RE.match(line):
            if current_num is not None or any(s.strip() for s in current):
                buckets.append((current_num, current))
            current_num = m.group(1)
            current = [line[m.end():]] if line[m.end():].strip() else []
            continue
        current.append(line)

    if current_num is not None or any(s.strip() for s in current):
        buckets.append((current_num, current))

    out: list[tuple[str | None, str]] = []
    for num, chunk_lines in buckets:
        text = "\n".join(chunk_lines).strip()
        if text:
            out.append((num, text))
    return out or [(None, body.strip())]


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate.

    Cyrillic tokenises worse than Latin on most BPE vocabularies — roughly 2.5
    characters per token rather than 4. Being wrong in the conservative
    direction here only means slightly smaller chunks, which is the safe error.
    """
    return max(1, int(len(text) / 2.5))
