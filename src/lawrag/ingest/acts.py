"""Registry of the acts we index, with the name variants that appear in citations.

Articles cite each other by name, in two languages, in whatever case the
sentence demands ("Налогового кодекса", "Салық кодексінің"). Resolving those
names to a stable act code is what turns loose prose references into a usable
citation graph, so the alias list is matched case-insensitively on a stemmed
prefix rather than exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import ActKind


@dataclass(frozen=True, slots=True)
class ActSpec:
    code: str
    kind: ActKind
    title_ru: str
    title_kk: str
    # The adilet / reference-bank document id. NOT all of these are confirmed —
    # see ``verified``. An unverified id produces a 404 rather than a wrong
    # document, but check before relying on one.
    official_id: str = ""
    # Space-separated *stems*, matched with any case ending appended, so a
    # single alias covers "Налоговый"/"Налогового"/"Налоговым" and the Kazakh
    # equivalents. Multi-stem aliases match across intervening endings, e.g.
    # "гражданск процессуальн" matches "Гражданского процессуального".
    aliases: tuple[str, ...] = ()
    # True only where official_id has been confirmed against a live adilet URL.
    # Keyword-only and last so that adding it cannot shift positional arguments.
    verified: bool = False


REGISTRY: dict[str, ActSpec] = {
    a.code: a
    for a in [
        # verified=True means the official_id has been checked against a live
        # adilet URL. The rest are best-effort and WILL need confirming before
        # any fetch or export request is built around them — see
        # https://adilet.zan.kz/rus/index/docs to look one up.
        ActSpec("konstitucia", ActKind.CONSTITUTION,
                "Конституция Республики Казахстан",
                "Қазақстан Республикасының Конституциясы",
                "K950001000",
                ("конституц", "конституци")),

        # The Civil Code is published as TWO documents — Общая часть (this one)
        # and Особенная часть, which has its own id. Indexing only this one
        # gives you half the Civil Code and no error to tell you so.
        ActSpec("gk", ActKind.CODE,
                "Гражданский кодекс Республики Казахстан (Общая часть)",
                "Қазақстан Республикасының Азаматтық кодексі (Жалпы бөлім)",
                "K940001000_",
                ("гражданск", "азаматтық"), verified=True),
        ActSpec("gpk", ActKind.CODE, "Гражданский процессуальный кодекс",
                "Азаматтық процестік кодекс", "K1500000377",
                ("гражданск процессуальн", "азаматтық процестік")),
        ActSpec("uk", ActKind.CODE, "Уголовный кодекс Республики Казахстан",
                "Қазақстан Республикасының Қылмыстық кодексі", "K1400000226",
                ("уголовн", "қылмыстық"), verified=True),
        ActSpec("upk", ActKind.CODE, "Уголовно-процессуальный кодекс",
                "Қылмыстық-процестік кодекс", "K1400000231",
                ("уголовно-процессуальн", "қылмыстық-процестік"), verified=True),
        ActSpec("uik", ActKind.CODE, "Уголовно-исполнительный кодекс",
                "Қылмыстық-атқару кодексі", "K1400000234",
                ("уголовно-исполнительн", "қылмыстық-атқару")),
        ActSpec("koap", ActKind.CODE,
                "Кодекс Республики Казахстан об административных правонарушениях",
                "Әкімшілік құқық бұзушылық туралы кодекс", "K1400000235",
                ("об административных правонарушени", "административн правонарушени",
                 "әкімшілік құқық бұзушылық"), verified=True),
        ActSpec("appk", ActKind.CODE,
                "Административный процедурно-процессуальный кодекс",
                "Әкімшілік рәсімдік-процестік кодекс", "K2000000350",
                ("административн процедурно-процессуальн", "әкімшілік рәсімдік-процестік")),

        # A NEW Tax Code (K2500000214) was adopted in 2025 and supersedes the
        # 2017 one. Both are registered: the 2017 code still governs periods
        # before the new one took effect, and point-in-time questions about
        # those periods cannot be answered without it indexed.
        ActSpec("nk", ActKind.CODE, "Налоговый кодекс Республики Казахстан",
                "Салық кодексі", "K2500000214",
                ("налогов", "салық"), verified=True),
        ActSpec("nk2017", ActKind.CODE,
                "Налоговый кодекс Республики Казахстан (кодекс 2017 года)",
                "Салық кодексі (2017 жылғы кодекс)", "K1700000120",
                (), verified=True),

        ActSpec("tk", ActKind.CODE, "Трудовой кодекс Республики Казахстан",
                "Еңбек кодексі", "K1500000414",
                ("трудов", "еңбек"), verified=True),
        ActSpec("pk", ActKind.CODE, "Предпринимательский кодекс Республики Казахстан",
                "Кәсіпкерлік кодексі", "K1500000375",
                ("предпринимательск", "кәсіпкерлік"), verified=True),
        ActSpec("bk", ActKind.CODE, "Бюджетный кодекс Республики Казахстан",
                "Бюджет кодексі", "K080000095_",
                ("бюджетн", "бюджет")),
        ActSpec("zk", ActKind.CODE, "Земельный кодекс Республики Казахстан",
                "Жер кодексі", "K030000442_",
                ("земельн", "жер"), verified=True),
        ActSpec("ek", ActKind.CODE, "Экологический кодекс Республики Казахстан",
                "Экология кодексі", "K2100000400",
                ("экологическ", "экология"), verified=True),
        ActSpec("vk", ActKind.CODE, "Водный кодекс Республики Казахстан",
                "Су кодексі", "K030000481_", ("водн", "су")),
        ActSpec("lk", ActKind.CODE, "Лесной кодекс Республики Казахстан",
                "Орман кодексі", "K030000477_", ("лесн", "орман")),
        ActSpec("tamk", ActKind.CODE,
                "Кодекс Республики Казахстан о таможенном регулировании",
                "Кеден реттеу туралы кодекс", "K1700000123",
                ("о таможенном регулировани", "таможенн", "кеден")),
        ActSpec("kobs", ActKind.CODE, "Кодекс о браке (супружестве) и семье",
                "Неке (ерлі-зайыптылық) және отбасы туралы кодекс", "K1100000518",
                ("о браке", "неке")),
        ActSpec("kozn", ActKind.CODE,
                "Кодекс о здоровье народа и системе здравоохранения",
                "Халық денсаулығы және денсаулық сақтау жүйесі туралы кодекс",
                "K2000000360",
                ("о здоровье народа", "халық денсаулығы")),
        ActSpec("sk", ActKind.CODE, "Социальный кодекс Республики Казахстан",
                "Әлеуметтік кодекс", "K2300000224",
                ("социальн", "әлеуметтік"), verified=True),
        ActSpec("nedra", ActKind.CODE, "Кодекс о недрах и недропользовании",
                "Жер қойнауы және жер қойнауын пайдалану туралы кодекс", "K1700000125",
                ("о недрах", "жер қойнауы")),
    ]
}

_WORD_TAIL = r"[\w\u0400-\u04FF]*"


# Several Kazakh act names are very short common words — "су" (water), "жер"
# (land), "орман" (forest). Without a left word boundary "су" matches inside
# "бизнесу"; without requiring a nearby "кодекс" it matches the ordinary noun.
# Both guards are needed, and both were found the hard way.
_LEFT_BOUNDARY = r"(?<![\w\u0400-\u04FF])"
_CODE_WORD = re.compile(r"кодекс|конституц", re.IGNORECASE)
_CODE_WORD_WINDOW = 40


def _alias_regex(alias: str) -> re.Pattern[str]:
    stems = alias.split()
    body = (_WORD_TAIL + r"\s+").join(re.escape(s) for s in stems) + _WORD_TAIL
    return re.compile(_LEFT_BOUNDARY + body, re.IGNORECASE)


# Longest alias first so that "гражданск процессуальн" is tried before the bare
# "гражданск" and wins the tie on an equal-position match.
_ALIAS_INDEX: list[tuple[re.Pattern[str], str, int]] = sorted(
    (
        (_alias_regex(alias), spec.code, len(alias.split()))
        for spec in REGISTRY.values()
        for alias in spec.aliases
    ),
    key=lambda t: -t[2],
)


def find_act_mentions(text: str) -> list[tuple[int, str]]:
    """All act mentions in ``text`` as (position, act_code), earliest first.

    Position matters more than alias length: in "статьи 76 Налогового кодекса и
    статей 5-7 Предпринимательского кодекса" both acts are present, and the one
    that governs a given article number is simply the nearer one.
    """
    hits: dict[int, str] = {}
    for pattern, code, _ in _ALIAS_INDEX:
        for m in pattern.finditer(text):
            # An act is being *named* only if the word "кодекс"/"конституция"
            # sits next to the name. "в 2024 году бизнесу" must not resolve to
            # the Water Code because "су" appears inside a word.
            window = text[m.start():m.end() + _CODE_WORD_WINDOW]
            if not _CODE_WORD.search(window):
                continue
            hits.setdefault(m.start(), code)
    return sorted(hits.items())


def resolve_act(name_fragment: str) -> str | None:
    """Map a citation fragment such as 'Налогового кодекса' to an act code."""
    mentions = find_act_mentions(name_fragment)
    return mentions[0][1] if mentions else None


def get(code: str) -> ActSpec | None:
    return REGISTRY.get(code)
