"""Redaction dates recovered from adilet's own amendment footnotes.

adilet serves the *current* text of an act, and every change to it is recorded
in a footnote directly under the article it touched:

    Сноска. Статья 8 с изменениями, внесенными законами РК от 05.11.2022
    № 158-VII (вводится в действие с 01.01.2023); от 11.06.2026 № 307-VIII
    (вводится в действие с 01.07.2026).

    Ескерту. 8-бапқа өзгеріс енгізілді - ҚР 05.11.2022 № 158-VII (01.01.2023
    бастап қолданысқа енгізіледі); 11.06.2026 № 307-VIII (...) Заңдарымен.

The latest commencement date in that footnote is the day the text now on the
page started to apply. An article with no footnote has not been amended since
the act itself was adopted, and the adoption date is printed in the page header
("Кодекс Республики Казахстан от 5 июля 2014 года № 235-V ЗРК.").

What this gives, and what it does not:

* It dates the text we *have*. It does not reconstruct the text an article had
  before its last amendment — that needs a snapshot fetched as of an earlier
  date. So an as-of question before an article's last amendment finds no text
  for that article, and the system abstains rather than quoting today's wording
  as if it had always applied.
* Commencement expressed relative to publication ("по истечении десяти
  календарных дней после дня его первого официального опубликования") is
  resolved against the law's signing date, because the publication date is not
  on the page. Publication follows signing by a few days, so these dates can be
  a few days early. They are counted and logged as approximate.
* Whole-text terminology replacements ("по всему тексту Кодекса слова ...
  заменены") are not attributed to every article; they would stamp every
  article with the same date and erase the real amendment history.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

_DMY = r"(\d{2})\.(\d{2})\.(\d{4})"
_YMD = re.compile(r"\b(\d{4})\.(\d{2})\.(\d{2})\b")

# A cited amending law: the law's own date, then its number. Russian writes
# "от 05.11.2022 № 158-VII", Kazakh "ҚР 05.11.2022 № 158-VII" or just the date
# after a semicolon. The number is required so that a bare commencement date is
# never mistaken for a law.
_LAW = re.compile(_DMY + r"\s*(?:г\.|ж\.)?\s*(?:жылғы\s*)?№\s*[\w\-–/]+", re.UNICODE)

_MONTHS_RU = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_MONTHS_KK = {
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4, "мамыр": 5, "маусым": 6,
    "шілде": 7, "тамыз": 8, "қыркүйек": 9, "қазан": 10, "қараша": 11, "желтоқсан": 12,
}

_NUMBERS = {
    # Russian, genitive as used after "по истечении"
    "одного": 1, "двух": 2, "трех": 3, "трёх": 3, "четырех": 4, "пяти": 5,
    "шести": 6, "семи": 7, "восьми": 8, "девяти": 9, "десяти": 10,
    "пятнадцати": 15, "двадцати": 20, "тридцати": 30, "сорока": 40,
    "сорока пяти": 45, "шестидесяти": 60, "девяноста": 90,
    "двадцати одного": 21, "двенадцати": 12,
    # Kazakh
    "бір": 1, "екі": 2, "үш": 3, "төрт": 4, "бес": 5, "алты": 6, "жеті": 7,
    "сегіз": 8, "тоғыз": 9, "он": 10, "он бес": 15, "жиырма": 20,
    "жиырма бір": 21, "отыз": 30, "қырық": 40, "қырық бес": 45, "алпыс": 60,
    "тоқсан": 90, "он екі": 12,
}
_NUM_ALT = "|".join(sorted((re.escape(k) for k in _NUMBERS), key=len, reverse=True))

_REL_RU = re.compile(
    rf"по\s+истечении\s+(?:по\s+истечении\s+)?(?P<n>\d+|{_NUM_ALT})\s+"
    r"(?:(?P<days>календарн\w*\s+дн\w*|дн\w*)|(?P<months>месяц\w*)|(?P<years>год\w*|лет))",
    re.IGNORECASE,
)
_REL_KK = re.compile(
    rf"(?:күнтізбелік\s+)?(?P<n>\d+|{_NUM_ALT})\s+"
    r"(?:(?P<days>күн)|(?P<months>ай)|(?P<years>жыл))\s+өткен\s+соң",
    re.IGNORECASE,
)
_FROM_PUBLICATION = re.compile(
    r"(со\s+дня\s+(?:его\s+)?(?:первого\s+)?официального\s+опубликования"
    r"|ресми\s+жарияланған\s+күнінен\s+бастап\s+қолданысқа)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Amendment:
    law_date: date
    effective: date
    approximate: bool


def _mkdate(d: str, m: str, y: str) -> date | None:
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    y += d.year
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _commencement(law_date: date, clause: str) -> tuple[date, bool]:
    """Resolve one commencement clause against the amending law's date."""
    explicit = re.search(_DMY, clause)
    if explicit:
        found = _mkdate(*explicit.groups())
        if found:
            return found, False
    for rel in (_REL_RU, _REL_KK):
        m = rel.search(clause)
        if not m:
            continue
        raw = m.group("n").lower()
        n = int(raw) if raw.isdigit() else _NUMBERS.get(raw)
        if n is None:
            break
        if m.group("days"):
            # Counted from publication, which is after signing: +1 for the day
            # of publication itself, and the result is still a lower bound.
            return law_date + timedelta(days=n + 1), True
        if m.group("months"):
            return _add_months(law_date, n) + timedelta(days=1), True
        return _add_months(law_date, 12 * n) + timedelta(days=1), True
    if _FROM_PUBLICATION.search(clause):
        return law_date + timedelta(days=1), True
    # No commencement stated: in force from signing, as far as the page says.
    return law_date, True


_WORD_DATE_RU = re.compile(r"\b(\d{1,2})\s+([а-яё]+)\s+(\d{4})\s*(?:г\.|года)?", re.IGNORECASE)


def _normalise_dates(line: str) -> str:
    """Bring the older footnote spellings to the modern "DD.MM.YYYY № N" form.

    * year-first Kazakh dates: "ҚР 2011.03.25 № 421-IV"
    * 1990s Russian footnotes: "Законом РК от 2 марта 1998 г. N 211"
    """
    line = _YMD.sub(r"\3.\2.\1", line)

    def word(m: re.Match) -> str:
        month = _month(m.group(2), _MONTHS_RU)
        if not month:
            return m.group(0)
        return f"{int(m.group(1)):02d}.{month:02d}.{m.group(3)}"

    line = _WORD_DATE_RU.sub(word, line)
    return re.sub(r"(\d{4})\s*N\s*(?=\d)", r"\1 № ", line)


def parse_amendments(line: str) -> list[Amendment]:
    """Every amending law cited in one footnote, with its commencement date."""
    line = _normalise_dates(line)
    laws = list(_LAW.finditer(line))
    out: list[Amendment] = []
    for i, m in enumerate(laws):
        law_date = _mkdate(m.group(1), m.group(2), m.group(3))
        if law_date is None:
            continue
        end = laws[i + 1].start() if i + 1 < len(laws) else len(line)
        tail = line[m.end():end]
        # The commencement clause is the first parenthesised group after the
        # law's number and before the next cited law.
        paren = re.search(r"\(([^()]*)\)", tail)
        clause = paren.group(1) if paren else ""
        effective, approx = _commencement(law_date, clause)
        out.append(Amendment(law_date, effective, approx))
    return out


def latest_effective(lines: list[str], *, not_after: date | None = None
                     ) -> tuple[date | None, bool]:
    """The day the current text of a node started to apply.

    Commencement dates after ``not_after`` (the snapshot date) are ignored:
    adilet shows amendments that have not commenced yet as editorial notes
    rather than applying them, so a later date here is an approximation that
    overshot, not a text from the future.
    """
    best: Amendment | None = None
    for line in lines:
        for a in parse_amendments(line):
            if not_after is not None and a.effective > not_after:
                continue
            if best is None or a.effective > best.effective:
                best = a
    if best is None:
        return None, False
    return best.effective, best.approximate


# ------------------------------------------------------------------ header --

_ADOPTED_RU = re.compile(r"от\s+(\d{1,2})\s+([А-Яа-яё]+)\s+(\d{4})\s+года", re.IGNORECASE)
_ADOPTED_KK = re.compile(r"(\d{4})\s+жылғы\s+(\d{1,2})\s+([А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІіIi]+)",
                         re.IGNORECASE)


def _month(word: str, table: dict[str, int]) -> int | None:
    # adilet sometimes types the Kazakh "і" as a Latin "i" ("шiлдедегi").
    word = word.lower().replace("i", "і")
    for stem, num in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if word.startswith(stem):
            return num
    return None


def parse_adoption_line(text: str) -> date | None:
    """"Кодекс Республики Казахстан от 5 июля 2014 года № 235-V" -> 2014-07-05."""
    m = _ADOPTED_RU.search(text)
    if m:
        month = _month(m.group(2), _MONTHS_RU)
        if month:
            return _mkdate(m.group(1), str(month), m.group(3))
    m = _ADOPTED_KK.search(text)
    if m:
        month = _month(m.group(3), _MONTHS_KK)
        if month:
            return _mkdate(m.group(2), str(month), m.group(1))
    return None


@dataclass(frozen=True, slots=True)
class PageHeader:
    adopted_on: date | None
    status: str          # adilet's own badge: "Обновленный", "Утративший силу", ...


def adilet_header(html: str) -> PageHeader:
    from bs4 import BeautifulSoup

    # Only the head of the page is needed, and the documents run to megabytes.
    cut = html.find("<article")
    soup = BeautifulSoup(html[:cut] if cut > 0 else html[:200_000], "lxml")
    slogan = soup.select_one(".slogan")
    if slogan is None:
        return PageHeader(None, "")
    line = slogan.select_one("p")
    status = slogan.select_one(".status")
    return PageHeader(
        adopted_on=parse_adoption_line(line.get_text(" ", strip=True)) if line else None,
        status=status.get_text(" ", strip=True) if status else "",
    )
