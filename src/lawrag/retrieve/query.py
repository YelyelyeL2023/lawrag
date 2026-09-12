"""Query analysis.

Cheap, deterministic, and done before any model runs. Three things get decided
here, and each of them changes what the retriever is allowed to return:

*Language* — which index and which answer language.
*As-of date* — the hard filter. Retrieving a repealed article is the failure
 mode that gets a user fined, so the date is a WHERE clause, never a ranking
 signal.
*Explicit citations* — "что говорит статья 278 КоАП" is a lookup, not a search,
 and treating it as a search is how a system returns the semantically-similar
 article 277 instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..ingest.acts import find_act_mentions
from ..ingest.structure import detect_lang
from ..models import Lang

_RU_ARTICLE_Q = re.compile(r"стат(?:ь(?:я|и|ей|ю|е)|ей)\s*(\d+(?:-\d+)*)", re.IGNORECASE)
_KK_ARTICLE_Q = re.compile(r"(\d+(?:-\d+)*)-ба[пб][Ѐ-ӿ]{0,8}", re.IGNORECASE)
_BARE_ARTICLE = re.compile(r"\bст\.?\s*(\d+(?:-\d+)*)", re.IGNORECASE)

_DATE_DMY = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR_ONLY = re.compile(r"\b(?:в|за|на)\s+(\d{4})\s*(?:год|году|жыл|жылы|ж\.|г\.)", re.IGNORECASE)

# Common shorthand that the act-alias matcher will not catch on its own.
_ABBREV = {
    "коап": "koap", "әқбтк": "koap",
    "нк": "nk", "гк": "gk", "ук": "uk", "тк": "tk", "пк": "pk",
    "упк": "upk", "гпк": "gpk", "бк": "bk", "зк": "zk", "эк": "ek",
}
_ABBREV_RE = re.compile(r"\b(" + "|".join(sorted(_ABBREV, key=len, reverse=True)) + r")\b",
                        re.IGNORECASE)


@dataclass(slots=True)
class QueryAnalysis:
    raw: str
    lang: Lang
    as_of: date
    as_of_explicit: bool = False
    act_hints: list[str] = field(default_factory=list)
    article_hints: list[tuple[str | None, str]] = field(default_factory=list)
    is_citation_lookup: bool = False

    @property
    def has_hard_filter(self) -> bool:
        return bool(self.act_hints or self.article_hints)


def _parse_date(text: str) -> tuple[date | None, bool]:
    if m := _DATE_ISO.search(text):
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), True
        except ValueError:
            # "2025-02-30" is a typo, not a reason to 500 the whole request.
            pass
    if m := _DATE_DMY.search(text):
        d, mo, y = (int(g) for g in m.groups())
        try:
            return date(y, mo, d), True
        except ValueError:
            return None, False
    if m := _YEAR_ONLY.search(text):
        # A bare year means "the law as it stood during that year". Mid-year is
        # the least-wrong single point, and the answer states the date it used.
        return date(int(m.group(1)), 7, 1), True
    return None, False


def analyze(question: str, *, as_of: date | None = None,
            lang: Lang | None = None, today: date | None = None) -> QueryAnalysis:
    today = today or date.today()
    parsed_date, explicit = _parse_date(question)
    effective = as_of or parsed_date or today

    qlang = lang or detect_lang(question)

    acts: list[str] = []
    for _pos, code in find_act_mentions(question):
        if code not in acts:
            acts.append(code)
    for m in _ABBREV_RE.finditer(question):
        code = _ABBREV[m.group(1).lower()]
        if code not in acts:
            acts.append(code)

    articles: list[tuple[str | None, str]] = []
    article_re = _KK_ARTICLE_Q if qlang is Lang.KK else _RU_ARTICLE_Q
    for m in list(article_re.finditer(question)) + list(_BARE_ARTICLE.finditer(question)):
        number = m.group(1)
        act = acts[0] if len(acts) == 1 else None
        if (act, number) not in articles:
            articles.append((act, number))

    return QueryAnalysis(
        raw=question.strip(),
        lang=qlang,
        as_of=effective,
        as_of_explicit=explicit or as_of is not None,
        act_hints=acts,
        article_hints=articles,
        is_citation_lookup=bool(articles),
    )
