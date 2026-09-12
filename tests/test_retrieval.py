"""Retrieval behaviour that the system's safety depends on."""

from __future__ import annotations

from datetime import date

from lawrag.models import Lang
from lawrag.retrieve.engine import RetrievalEngine
from lawrag.retrieve.hybrid import or_tsquery
from lawrag.retrieve.query import analyze


def test_or_tsquery_does_not_and_every_word():
    """websearch_to_tsquery ANDs terms, so a whole sentence matches nothing."""
    q = or_tsquery("Какой штраф грозит малому предприятию за просрочку")
    assert "|" in q
    assert "&" not in q


def test_query_analysis_extracts_date_act_and_article():
    a = analyze("Какой штраф по статье 278 КоАП?", today=date(2026, 8, 22))
    assert a.act_hints == ["koap"]
    assert a.article_hints == [("koap", "278")]
    assert a.is_citation_lookup

    b = analyze("Что было по состоянию на 15.03.2025?", today=date(2026, 8, 22))
    assert b.as_of == date(2025, 3, 15) and b.as_of_explicit


def test_point_in_time_returns_the_redaction_in_force(live_db):
    engine = RetrievalEngine(live_db)

    now = engine.retrieve("Какой штраф за просрочку постановки на учет?")
    then = engine.retrieve("Какой штраф за просрочку постановки на учет?",
                           as_of=date(2025, 3, 15))

    def redaction(result):
        hit = next(h for h in result.hits if h.article_node_id == "koap/st278")
        return hit.valid_from

    assert redaction(now) == date(2026, 1, 1)
    assert redaction(then) == date(2024, 1, 1)


def test_out_of_corpus_question_returns_nothing(live_db):
    """The relevance floor: rank fusion must not answer from the least-bad match."""
    engine = RetrievalEngine(live_db)
    result = engine.retrieve("Какая ставка налога на добычу редкоземельных металлов?")
    assert result.hits == []
    assert result.packet.sources == []


def test_explicit_citation_outranks_similarity(live_db):
    engine = RetrievalEngine(live_db)
    result = engine.retrieve("что говорит статья 279 КоАП")
    assert result.hits[0].article_node_id == "koap/st279"
    assert "citation" in result.hits[0].provenance


def test_hydration_pulls_in_referenced_definitions(live_db):
    """nk/st76 refers to pk/st24; the definition must travel with it."""
    engine = RetrievalEngine(live_db)
    result = engine.retrieve("В какой срок малый бизнес встает на регистрационный учет?")
    hydrated = next(h for h in result.hydrated if h.hit.article_node_id == "nk/st76")
    supporting = {s.node_id for s in hydrated.supporting}
    assert "pk/st24" in supporting


def test_kazakh_query_retrieves_kazakh_text(live_db):
    engine = RetrievalEngine(live_db)
    result = engine.retrieve("Тіркеу есебіне қою мерзімі қандай?")
    assert result.analysis.lang is Lang.KK
    assert result.hits and result.hits[0].lang == "kk"


def test_both_language_faces_are_one_source_not_two(live_db):
    engine = RetrievalEngine(live_db)
    result = engine.retrieve("Тіркеу есебіне қою мерзімі қандай?")
    keys = [f"{h.node_id}@{h.valid_from}" for h in result.hits]
    assert len(keys) == len(set(keys))


def test_packet_states_validity_and_location(live_db):
    engine = RetrievalEngine(live_db)
    result = engine.retrieve("В какой срок нужно встать на регистрационный учет?")
    text = result.packet.text
    assert "ИСТОЧНИК 1" in text
    assert "Действует с" in text
    assert "Расположение" in text
