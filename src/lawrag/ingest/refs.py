"""Cross-reference and definition extraction.

Legal text is a graph pretending to be prose. "в порядке, установленном статьей
76 Налогового кодекса" is an edge, and a retrieval system that does not follow
it hands the model a rule whose operative content lives somewhere else. Every
reference found here becomes a row in ``xref`` and is available for one-hop
hydration at query time.
"""

from __future__ import annotations

import re

from ..ids import article_id
from ..models import Lang, XRef
from .acts import find_act_mentions, resolve_act

# ---------------------------------------------------------------------------
# Russian
# ---------------------------------------------------------------------------

# "статьей 76", "статьи 76-1", "статьями 5, 6 и 7"
# Note "статей" (genitive plural) has no soft sign, so the stem is "стат", not
# "стать" — a detail that silently drops every multi-article citation if missed.
_RU_ARTICLE = re.compile(
    r"стат(?:ь(?:я|и|ёй|ей|ю|е|ями|ям|ях)|ей)"
    r"\s+(\d+(?:-\d+)*(?:\s*,\s*\d+(?:-\d+)*)*(?:\s*(?:и|-)\s*\d+(?:-\d+)*)?)",
    re.IGNORECASE,
)
# "пунктом 2 статьи 278"
_RU_POINT_OF_ARTICLE = re.compile(
    r"пункт(?:ом|а|е|у|ами|ов)?\s+(\d+(?:-\d+)*)\s+стать(?:и|ей|ёй)\s+(\d+(?:-\d+)*)",
    re.IGNORECASE,
)
_RU_SAME_ACT = re.compile(r"настоящ(?:его|ем|ий|им)\s+Кодекс", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Kazakh — the number precedes the label, and case endings agglutinate onto it.
# ---------------------------------------------------------------------------

# Kazakh consonant alternation turns "бап" into "баб-" before a vowel suffix,
# so "76-бабына" and "76-бабының" both refer to article 76. Matching only "бап"
# loses the majority of real citations, since most appear in an oblique case.
_KK_ARTICLE = re.compile(
    r"(\d+(?:-\d+)*)-ба[пб][\u0400-\u04FF]{0,8}", re.IGNORECASE
)
_KK_POINT_OF_ARTICLE = re.compile(
    r"(\d+(?:-\d+)*)-бабының\s+(\d+(?:-\d+)*)-тармағ", re.IGNORECASE
)
_KK_SAME_ACT = re.compile(r"осы\s+Кодекс", re.IGNORECASE)

_NUM_SPLIT = re.compile(r"[,\s]+и\s+|\s*,\s*")

# Word order decides where the governing act name sits. Russian names the act
# *after* the number ("статьей 76 Налогового кодекса"); Kazakh names it *before*
# ("Салық кодексінің 76-бабы"). Scanning both directions indiscriminately makes
# a reference inherit the act of the previous, unrelated citation in the same
# sentence, so the primary direction is tried first and alone.
_WINDOW = 70


def _nearest_act(window: str, from_end: bool) -> str | None:
    mentions = find_act_mentions(window)
    if not mentions:
        return None
    return mentions[-1][1] if from_end else mentions[0][1]


def _act_for(text: str, start: int, end: int, lang: Lang, self_act: str) -> str | None:
    after = text[end:end + _WINDOW]
    before = text[max(0, start - _WINDOW):start]
    same = _KK_SAME_ACT if lang is Lang.KK else _RU_SAME_ACT

    if lang is Lang.KK:
        # "осы Кодекстің 278-бабы" — the qualifier immediately precedes.
        primary, secondary = before, after
        primary_from_end, secondary_from_end = True, False
    else:
        primary, secondary = after, before
        primary_from_end, secondary_from_end = False, True

    same_pos = None
    for m in same.finditer(primary):
        same_pos = m.start()
    act_mentions = find_act_mentions(primary)

    if same_pos is not None and not act_mentions:
        return self_act
    if act_mentions:
        nearest = act_mentions[-1][1] if primary_from_end else act_mentions[0][1]
        if same_pos is None:
            return nearest
        # Both present in the same window: whichever sits closer to the
        # reference governs it.
        nearest_pos = act_mentions[-1][0] if primary_from_end else act_mentions[0][0]
        if primary_from_end:
            return self_act if same_pos > nearest_pos else nearest
        return self_act if same_pos < nearest_pos else nearest

    if same.search(secondary):
        return self_act
    return _nearest_act(secondary, secondary_from_end) or self_act


def _numbers(blob: str) -> list[str]:
    return [n.strip() for n in _NUM_SPLIT.split(blob) if n.strip()]


def extract_refs(node_id: str, self_act: str, text: str, lang: Lang) -> list[XRef]:
    refs: list[XRef] = []
    seen: set[tuple[str | None, str | None]] = set()

    def add(raw: str, act: str | None, article: str | None) -> None:
        key = (act, article)
        if key in seen or article is None:
            return
        seen.add(key)
        dst = article_id(act, article) if act else None
        refs.append(
            XRef(src_node_id=node_id, raw=raw.strip(), dst_act_code=act,
                 dst_article=article, dst_node_id=dst)
        )

    point_re = _KK_POINT_OF_ARTICLE if lang is Lang.KK else _RU_POINT_OF_ARTICLE
    for m in point_re.finditer(text):
        # Kazakh writes article-then-point, Russian point-then-article.
        article = m.group(1) if lang is Lang.KK else m.group(2)
        act = _act_for(text, m.start(), m.end(), lang, self_act)
        add(m.group(0), act, article)

    article_re = _KK_ARTICLE if lang is Lang.KK else _RU_ARTICLE
    for m in article_re.finditer(text):
        act = _act_for(text, m.start(), m.end(), lang, self_act)
        for number in _numbers(m.group(1)):
            add(m.group(0), act, number)

    # A self-reference carries no information and would make every article its
    # own neighbour during hydration.
    own_article = "/".join(node_id.split("/")[:2])
    return [r for r in refs if r.dst_node_id != own_article]


# ---------------------------------------------------------------------------
# Defined terms
# ---------------------------------------------------------------------------

# KZ codes concentrate their vocabulary in one article, almost always titled
# "Основные понятия, используемые в настоящем Кодексе" / "Осы Кодексте
# пайдаланылатын негізгі ұғымдар", laid out as "1) термин — определение;".
DEFINITION_ARTICLE_RE = re.compile(
    r"(основные\s+понятия|негізгі\s+ұғымдар|осы\s+кодексте\s+пайдаланылатын)",
    re.IGNORECASE,
)

_DEF_ITEM = re.compile(
    r"^\s*\d+(?:-\d+)*\)\s*(?P<term>[^—–-]{2,120}?)\s*[—–]\s*(?P<body>.+)$",
    re.MULTILINE,
)


def is_definition_article(heading: str) -> bool:
    return bool(DEFINITION_ARTICLE_RE.search(heading or ""))


def extract_definitions(node_id: str, act_code: str, text: str) -> list[tuple[str, str]]:
    """Return (term, definition) pairs from a definitions article."""
    out: list[tuple[str, str]] = []
    for m in _DEF_ITEM.finditer(text):
        term = " ".join(m.group("term").split()).strip(" \"«»")
        body = " ".join(m.group("body").split()).rstrip(";. ")
        if 2 <= len(term) <= 120 and body:
            out.append((term.lower(), body))
    return out
