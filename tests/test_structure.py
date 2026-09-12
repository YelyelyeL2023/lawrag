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
