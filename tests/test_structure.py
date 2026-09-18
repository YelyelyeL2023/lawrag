"""The heading dialects and the note/amendment distinction."""

from __future__ import annotations

import pytest

from lawrag.ingest import structure as S
from lawrag.models import Lang, NodeKind


@pytest.mark.parametrize("line,lang,kind,number", [
    ("Статья 278. Нарушение срока", Lang.RU, NodeKind.ARTICLE, "278"),
    ("Статья 278-1. Дополнение", Lang.RU, NodeKind.ARTICLE, "278-1"),
    ("Глава 14. Правонарушения", Lang.RU, NodeKind.CHAPTER, "14"),
    ("РАЗДЕЛ 2. ОСОБЕННАЯ ЧАСТЬ", Lang.RU, NodeKind.SECTION, "2"),
    ("278-бап. Мерзімді бұзу", Lang.KK, NodeKind.ARTICLE, "278"),
    ("14-тарау. Құқық бұзушылық", Lang.KK, NodeKind.CHAPTER, "14"),
    ("2-БӨЛІМ. ЕРЕКШЕ БӨЛІК", Lang.KK, NodeKind.SECTION, "2"),
])
def test_headings_in_both_dialects(line, lang, kind, number):
    heading = S.match_heading(line, lang)
    assert heading is not None
    assert heading.kind is kind
    assert heading.number == number


def test_russian_pattern_does_not_match_kazakh_dialect():
    """The failure mode this guards: a Kazakh act parsing to one flat blob."""
    assert S.match_heading("278-бап. Мерзімді бұзу", Lang.RU) is None


def test_language_detection():
    assert S.detect_lang("Осы Кодекстің мақсаты — қатынастарды реттеу") is Lang.KK
    assert S.detect_lang("Настоящий Кодекс регулирует отношения") is Lang.RU


def test_amendment_provenance_is_separated_from_substantive_note():
    text = (
        "1. Основная норма.\n"
        "Примечание. Для целей настоящей статьи срок исчисляется со дня "
        "вступления решения в законную силу.\n"
        "Сноска. Статья 278 в редакции Закона РК от 14.11.2025 № 99-VIII."
    )
    body, provenance = S.strip_amendment_notes(text)
    assert len(provenance) == 1 and "Сноска" in provenance[0]
    # The substantive note survives — it is operative law.
    assert "Примечание" in body

    point_body, notes = S.split_notes(body)
    assert len(notes) == 1
    assert "Примечание" not in point_body


def test_kazakh_note_and_amendment_share_an_opening_word():
    substantive = "Ескерту. Осы баптың мақсаттары үшін мерзім есептеледі."
    amendment = "Ескерту. 278-бап жаңа редакцияда - ҚР 14.11.2025 № 99-VIII Заңымен."
    assert not S.is_amendment_line(substantive)
    assert S.is_amendment_line(amendment)


def test_subpoints_are_not_split_as_points():
    body = "1. Первый.\n2. Второй:\n1) подпункт;\n2) подпункт два.\n3. Третий."
    parts = S.split_points(body)
    assert [n for n, _ in parts] == ["1", "2", "3"]
    assert "подпункт" in parts[1][1]


def test_table_is_detected_so_it_is_never_split():
    assert S.contains_table("Категория\tРазмер\nМалый\t15")


# --- regressions found on the real adilet KoAP / TK snapshot (2026-09-16) ---

def test_long_article_title_is_still_a_heading():
    # KoAP article 199's title runs past 300 characters; a length cap merged
    # 28 RU and 31 KK articles into the article before them.
    line = "Статья 199. " + "Нарушение требований законодательства " * 12
    heading = S.match_heading(line, Lang.RU)
    assert heading is not None and heading.number == "199"
    kk = "199-бап. " + "Қазақстан Республикасы заңнамасының талаптарын бұзу " * 8
    assert S.match_heading(kk, Lang.KK).number == "199"


@pytest.mark.parametrize("line,lang", [
    ("Статья 68 с изменениями, внесенными законами РК от 20.04.2023 № 226-VII "
     "(вводится в действие с 01.07.2023).", Lang.RU),
    ("Статья 5 настоящего Кодекса применяется ...", Lang.RU),
    ("5-бапта көзделген жағдайларда ...", Lang.KK),
])
def test_footnote_or_prose_is_not_an_article_heading(line, lang):
    assert S.match_heading(line, lang) is None


def test_footnote_without_opener_is_a_footnote():
    assert S.is_footnote_line(
        "Статья 68 с изменениями, внесенными законами РК от 20.04.2023 № 226-VII "
        "(вводится в действие с 01.07.2023)")


def test_kazakh_article_heading_variants():
    assert S.match_heading("434-3 бап. Киім-кешекті киіп жүру", Lang.KK).number == "434-3"
    assert S.match_heading("701-бап Халықтың саламаттылығы", Lang.KK).number == "701"


def test_numbered_note_items_stay_in_the_note():
    body = ("1. Первая часть.\n2. Вторая часть.\nПримечания.\n"
            "1. Длящимся признается ...\n2. Под действием непреодолимой силы ...")
    points, notes = S.split_notes(body)
    assert [n for n, _ in S.split_points(points)] == ["1", "2"]
    assert len(notes) == 1 and "Длящимся" in notes[0] and "непреодолимой" in notes[0]


def test_kazakh_note_word_and_sanction_word():
    body = "1. Бірінші бөлік.\nЕскертпелер.\n1. Осы баптың мақсаттары үшін ..."
    points, notes = S.split_notes(body)
    assert [n for n, _ in S.split_points(points)] == ["1"]
    assert len(notes) == 1
    # "ескерту жасауға" is the sanction "a warning", not a note.
    sanction = "1. Құқық бұзушылық –\nескерту жасауға не жеке тұлғаларға айыппұл салуға әкеп соғады."
    points, notes = S.split_notes(sanction)
    assert notes == [] and "ескерту жасауға" in points
