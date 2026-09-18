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
    # adilet's Kazakh pages are not consistent: "2-БӨЛІМ. ...", "1 бөлім. ...",
    # "II бөлім. ..." and "Параграф 1. ..." all occur. The container word must
    # end there — "1-бөлімше" is a subsection, not a section — and a spaced
    # form must be followed by a full stop or nothing, so prose that happens
    # to start with "2 бөлім" is not taken for a heading.
    (NodeKind.SECTION, re.compile(
        r"^\s*([IVXLCDM]+|\d+(?:-\d+)*)(?:-|\s+)БӨЛІМ(?![А-Яа-яӘ-ӿ])(?:\.\s*(.*)|\s*)$", re.IGNORECASE)),
    (NodeKind.CHAPTER, re.compile(
        r"^\s*(\d+(?:-\d+)*)(?:-|\s+)ТАРАУ(?![А-Яа-яӘ-ӿ])(?:\.\s*(.*)|\s*)$", re.IGNORECASE)),
    (NodeKind.PARAGRAPH, re.compile(
        r"^\s*(?:(\d+(?:-\d+)*)(?:-|\s+)ПАРАГРАФ|ПАРАГРАФ\s+(\d+(?:-\d+)*))(?![А-Яа-яӘ-ӿ])(?:\.\s*(.*)|\s*)$",
        re.IGNORECASE)),
    # "434-3 бап." with a space occurs too. "-бап" must end the word, or prose
    # such as "5-бапта көзделген ..." would open a phantom article.
    (NodeKind.ARTICLE, re.compile(r"^\s*(\d+(?:-\d+)*)(?:-|\s+)бап(?![А-Яа-яӘ-ӿ])\.?\s*(.*)$",
                                  re.IGNORECASE)),
]

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _arabic(number: str) -> str:
    """Section numbers are written "II" on one page and "2" on the other; the
    node id must be the same for both languages."""
    if not number or not re.fullmatch(r"[IVXLCDM]+", number, re.IGNORECASE):
        return number
    total, prev = 0, 0
    for ch in reversed(number.upper()):
        value = _ROMAN[ch]
        total += -value if value < prev else value
        prev = max(prev, value)
    return str(total)


# Points and subpoints look the same in both languages.
POINT_RE = re.compile(r"^\s*(\d+(?:-\d+)*)\.\s+(?=\S)")
SUBPOINT_RE = re.compile(r"^\s*(\d+(?:-\d+)*)\)\s+(?=\S)")

# Substantive notes. "Примечание" / "Ескерту" attached to an article is
# operative law — it routinely defines how the article is counted or applied —
# so it must travel with the article, never be dropped and never be swallowed by
# whichever numbered point happens to precede it.
#
# Kazakh codes head a note "Ескертпе" / "Ескертпелер"; without them every
# numbered KK note became a duplicate point. The heading word is capitalised and
# followed by a stop or colon: case-insensitive matching took the KoAP sanction
# line "ескерту жасауға ..." ("влечет предупреждение") for a note 177 times and
# moved it out of the article body.
NOTE_RE = re.compile(
    r"^\s*(?:Примечани[ея]|ПРИМЕЧАНИ[ЕЯ]|Ескерту|Ескертпе(?:лер)?|ЕСКЕРТПЕ(?:ЛЕР)?)\s*(?:[.:]\s*|$)")

# Amendment provenance. Same opening word as a substantive note in Kazakh
# ("Ескерту"), so the two are told apart by whether a law is *cited*: an
# amendment line always names the amending act by number or date.
_LAW_CITATION = re.compile(
    r"(№\s*\d+|\bN\s*\d+|от\s+\d{2}\.\d{2}\.\d{4}|\d{2}\.\d{2}\.\d{4}"
    r"|от\s+\d{1,2}\s+[а-яё]+\s+\d{4}|Заңымен|Закон(?:ом|а)\s+РК"
    r"|в\s+редакции|жаңа\s+редакцияда)",
    re.IGNORECASE,
)
# adilet's typists sometimes use a Latin "C" or "E" in the opening word
# ("Cноска"), which is invisible on the page and made those footnotes read as
# article text — both left in the body and never used for dating.
_PROVENANCE_OPENER = re.compile(
    r"^\s*(?:[СC]носка|[ЕE]скерту)\b"
    # adilet occasionally drops the "Сноска." opener and starts the footnote
    # with the article itself: "Статья 68 с изменениями, внесенными законами РК
    # от ...". With no full stop after the number it was taken for a second
    # heading of article 68 and its text replaced the real article in the index.
    r"|^\s*[СC]тать[яи]\s+\d+(?:-\d+)*\s+(?:с\s+изменени|с\s+дополнени|в\s+редакции|исключен|дополнен)"
    r"|^\s*\d+(?:-\d+)*-бап(?:қа|тың|ы)?\s+(?:өзгеріс|өзгерістер|толықтыру|жаңа\s+редакцияда|алып\s+тасталды)",
    re.IGNORECASE)


# Editorial notes from the publisher (ИЗПИ / РЦПИ, in Kazakh ЗҚАИ / РҚАО) —
# "Примечание ИЗПИ! В ч.3 статьи 44 предусматривается в редакции ...". adilet
# prints them in red inside the article. They are not law: they announce changes
# that have not commenced, or point at the commencement article. They share the
# word "Примечание" with a substantive note, so without this they would be
# attached to the article as an operative note and cited as its rule.
_EDITORIAL_OPENER = re.compile(
    r"^\s*(?:Примечание\s+(?:ИЗПИ|РЦПИ)\s*!|(?:ЗҚАИ|РҚАО)-ның\s+ескертпесі\s*!|\[ИЗПИ\])",
    re.IGNORECASE,
)


def is_editorial_line(line: str) -> bool:
    return bool(_EDITORIAL_OPENER.match(line))


def is_footnote_line(line: str) -> bool:
    """An amendment footnote proper — the lines that carry redaction dates."""
    return bool(_PROVENANCE_OPENER.match(line) and _LAW_CITATION.search(line))


def is_amendment_line(line: str) -> bool:
    return is_footnote_line(line) or is_editorial_line(line)

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


# Latin letters typed in place of their Cyrillic look-alikes. adilet's Kazakh
# pages write "2-БӨЛIМ" with a Latin I in four of five section headings, which
# hid those sections from the parser: their chapters were attached to section 1,
# took different node ids from the Russian text, and re-parented every Russian
# article on import. Normalised for heading detection only.
_HOMOGLYPHS = str.maketrans({"I": "І", "i": "і", "C": "С", "E": "Е", "O": "О",
                             "A": "А", "B": "В", "H": "Н", "K": "К", "M": "М",
                             "P": "Р", "T": "Т", "X": "Х"})


def _cyrillic_heading(line: str) -> str:
    head, _, rest = line.partition(".")
    # Only the heading word itself, plus the title if the whole line is Cyrillic
    # apart from look-alikes (a Roman section numeral must survive).
    if re.search(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", head):
        return re.sub(r"(?<=[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі])[IiCEOABHKMPTX]|[IiCEOABHKMPTX](?=[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі])",
                      lambda m: m.group(0).translate(_HOMOGLYPHS), line)
    return line


_ARTICLE_WITH_STOP = re.compile(r"^\s*(?:Статья\s+\d+(?:-\d+)*|\d+(?:-\d+)*(?:-|\s+)бап)\.", re.IGNORECASE)


def _plausible_article_heading(line: str, title: str) -> bool:
    """Tell an article heading from prose or a footnote that starts the same way.

    "Статья 5 настоящего Кодекса ..." in running text and "Статья 68 с
    изменениями, внесенными ..." (a footnote without its opener) are not
    headings. A real heading puts a full stop after the number, and then its
    length says nothing: KoAP has dozens of article titles over 300 characters,
    and a length cap silently merged each of those articles into the one before
    it — 28 Russian and 31 Kazakh KoAP articles were missing from the index.
    """
    if is_footnote_line(line):
        return False
    if _ARTICLE_WITH_STOP.match(line):
        return len(line) <= 2000
    # No stop ("701-бап Халықтың ..."): accept only a short line whose title is
    # empty or capitalised.
    return len(line) <= 300 and (not title or title[:1].isupper())


def match_heading(line: str, lang: Lang) -> Heading | None:
    line = _cyrillic_heading(line)
    patterns = _KK_PATTERNS if lang is Lang.KK else _RU_PATTERNS
    for kind, pattern in patterns:
        m = pattern.match(line)
        if m:
            groups = [g for g in m.groups()]
            if kind is NodeKind.PARAGRAPH and pattern in (p for _, p in _KK_PATTERNS):
                number, title = groups[0] or groups[1], (groups[2] or "").strip()
            else:
                number, title = m.group(1), (m.group(2) or "").strip()
            if kind is NodeKind.SECTION:
                number = _arabic(number)
            if kind is NodeKind.ARTICLE and not _plausible_article_heading(line, title):
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
    # "Примечания." on its own line (or "Примечания. 1. ...") introduces a
    # numbered list that belongs to the note. Ending the note at the first
    # "1." turned those items into points that collided with the article's own
    # point 1, 2, ... and overwrote them in the index.
    numbered_note = False
    for line in text.splitlines():
        m = NOTE_RE.match(line.strip())
        if m and not is_amendment_line(line.strip()):
            in_note = True
            rest = line.strip()[m.end():]
            numbered_note = not rest or bool(POINT_RE.match(rest))
            notes.append(line.strip())
            continue
        if in_note:
            if not line.strip() or (match_any_marker(line) and not numbered_note):
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
