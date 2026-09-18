"""Redaction dates read from adilet's amendment footnotes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from lawrag.ingest.dating import latest_effective, parse_adoption_line, parse_amendments
from lawrag.ingest.parser import UNDATED, adilet_html_to_lines, parse_adilet
from lawrag.models import NodeKind

PAGE = """
<html><body>
<div class="container_alpha slogan"><h1>Об административных правонарушениях</h1>
<span class="status status_upd">Обновленный</span>
<p>Кодекс Республики Казахстан от 5 июля 2014 года № 235-V ЗРК.</p></div>
<article>
<h3 id="z2"> РАЗДЕЛ 1. ОБЩИЕ ПОЛОЖЕНИЯ<br>Глава 1. ЗАКОНОДАТЕЛЬСТВО<br>ОБ АДМИНИСТРАТИВНЫХ ПРАВОНАРУШЕНИЯХ</h3>
<p><b><a name="z4"></a>Статья 1. Законодательство</b></p>
<p>&nbsp; 1. Изменения вносятся в <a href="/rus/docs/K1400000226">Уголовный кодекс</a> Республики Казахстан.</p>
<span class="note">&nbsp; Сноска. Статья 1 с изменениями, внесенными Законами РК от 30.12.2019 </span><a href="#">№ 300-VІ</a><span class="note"> (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 05.11.2022 </span><a href="#">№ 158-VII</a><span class="note"> (вводится в действие с 01.01.2023).</span><br>
<p><b><a name="z5"></a>Статья 2. Основание</b></p>
<p>&nbsp; Основанием является деяние.</p>
<p><b><a name="z6"></a>Статья 3. Штраф</b></p>
<p><font color="#FF0000">&nbsp; Примечание ИЗПИ!</font><br>
<font color="#FF0000">&nbsp; В ч.1 статьи 3 предусматривается изменение Законом РК от 12.06.2026 <a href="#">№ 312-VIII</a> (вводится в действие с 01.01.2027).</font><br></p>
<p>&nbsp; 1. Штраф есть денежное взыскание.</p>
<span class="note">&nbsp; Сноска. Статья 3 в редакции Закона РК от 11.06.2026 </span><a href="#">№ 307-VIII</a><span class="note"> (вводится в действие с 01.07.2026).</span><br>
</article></body></html>
"""

META = {"file": "x.ru.html", "code": "koap", "kind": "code", "official_id": "K1400000235",
        "lang": "ru", "title": "КоАП", "url": "https://adilet.zan.kz/rus/docs/K1400000235",
        "fetched_at": "2026-09-08T17:26:04+0500"}


def test_inline_links_do_not_break_lines():
    lines = adilet_html_to_lines(PAGE)
    assert "1. Изменения вносятся в Уголовный кодекс Республики Казахстан." in lines
    assert not any(line.startswith("№ ") for line in lines)
    assert "РАЗДЕЛ 1. ОБЩИЕ ПОЛОЖЕНИЯ" in lines          # section and chapter stay apart
    assert "Глава 1. ЗАКОНОДАТЕЛЬСТВО" in lines


def test_latin_c_in_the_footnote_opener_is_still_a_footnote():
    from lawrag.ingest.structure import is_footnote_line

    assert is_footnote_line("Cноска. Статья 16 с изменениями, внесенными законами РК от "
                            "12.01.2007 № 224")


def test_footnote_dates_each_article():
    _, root, doc = parse_adilet(PAGE, META)
    arts = {a.number: a for a in root.articles()}
    assert doc.adopted_on == date(2014, 7, 5)
    assert doc.valid_from == date(2026, 9, 8)
    assert arts["1"].texts[doc.lang].valid_from == date(2023, 1, 1)
    assert arts["2"].texts[doc.lang].valid_from == date(2014, 7, 5)   # never amended
    assert arts["3"].texts[doc.lang].valid_from == date(2026, 7, 1)


def test_children_inherit_the_article_date():
    _, root, doc = parse_adilet(PAGE, META)
    art = next(a for a in root.articles() if a.number == "1")
    for child in art.children:
        assert child.texts[doc.lang].valid_from == date(2023, 1, 1)


def test_editorial_note_is_not_an_operative_note_and_does_not_date():
    """"Примечание ИЗПИ!" announces a 2027 change; it is neither law nor a date."""
    _, root, doc = parse_adilet(PAGE, META)
    art = next(a for a in root.articles() if a.number == "3")
    operative = [c for c in art.children if c.kind is NodeKind.NOTE and not c.id.endswith("/amend")]
    assert operative == []
    assert "ИЗПИ" not in art.texts[doc.lang].body
    assert art.texts[doc.lang].valid_from == date(2026, 7, 1)


def test_declared_valid_from_overrides_footnotes():
    _, root, doc = parse_adilet(PAGE, {**META, "valid_from": "2026-01-01"})
    assert {a.texts[doc.lang].valid_from for a in root.articles()} == {date(2026, 1, 1)}


def test_no_adoption_date_is_reported_as_undated():
    page = PAGE.replace("от 5 июля 2014 года", "")
    _, root, doc = parse_adilet(page, META)
    art = next(a for a in root.articles() if a.number == "2")
    assert art.texts[doc.lang].valid_from == UNDATED
    assert doc.dating["undated"] == 1


@pytest.mark.parametrize("line, effective, approximate", [
    ("Сноска. Статья 5 с изменениями, внесенными Законом РК от 29.12.2014 № 272-V "
     "(вводится в действие 01.01.2015).", date(2015, 1, 1), False),
    ("Сноска. Статья 12 исключена Законом РК от 03.07.2020 № 359-VI (вводится в действие "
     "со дня его первого официального опубликования).", date(2020, 7, 4), True),
    ("Сноска. Статья 9 в редакции Закона РК от 10.01.2025 № 155-VIII (вводится в действие "
     "по истечении шести месяцев после дня его первого официального опубликования).",
     date(2025, 7, 11), True),
    ("Ескерту. 5-бапқа өзгеріс енгізілді - ҚР 29.12.2014 № 272-V Заңымен "
     "(01.01.2015 бастап қолданысқа енгізіледі).", date(2015, 1, 1), False),
    ("Ескерту. 1-бапқа өзгеріс енгізілді - ҚР 30.12.2019 № 300-VІ (алғашқы ресми "
     "жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі)",
     date(2020, 1, 10), True),
    ("Ескерту. 117-бап жаңа редакцияда - ҚР 2011.03.25 № 421-IV (алғашқы ресми "
     "жарияланғанынан кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі) Заңымен.",
     date(2011, 4, 5), True),
])
def test_commencement_clauses(line, effective, approximate):
    (a,) = parse_amendments(line)
    assert a.effective == effective
    assert a.approximate is approximate


def test_commencement_after_the_snapshot_is_ignored():
    line = ("Сноска. Статья 1 с изменениями, внесенными Законами РК от 05.11.2022 № 158-VII "
            "(вводится в действие с 01.01.2023); от 12.06.2026 № 312-VIII "
            "(вводится в действие с 01.01.2027).")
    assert latest_effective([line], not_after=date(2026, 9, 8))[0] == date(2023, 1, 1)


def test_adoption_line_in_both_languages():
    assert parse_adoption_line("Кодекс Республики Казахстан от 18 июля 2025 года № 214-VIII ЗРК.") \
        == date(2025, 7, 18)
    assert parse_adoption_line("Қазақстан Республикасының Кодексі 2015 жылғы 23 қарашадағы № 414-V ҚРЗ.") \
        == date(2015, 11, 23)
    # Latin "i" standing in for the Kazakh letter, as on the Tax Code page.
    assert parse_adoption_line("Қазақстан Республикасының 2025 жылғы 18 шiлдедегi № 214-VIII ҚРЗ Кодексі.") \
        == date(2025, 7, 18)


def test_mixed_editions_under_one_act_code_are_refused(tmp_path: Path):
    from lawrag.ingest.pipeline import _refuse_mixed_editions

    fixture = Path(__file__).parent / "fixtures" / "synthetic" / "koap.ru.act.txt"
    copy = tmp_path / "koap.ru.act.txt"
    copy.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="more than one document"):
        _refuse_mixed_editions([copy], [("", META)])
