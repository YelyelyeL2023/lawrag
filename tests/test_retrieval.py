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


def test_named_article_without_text_on_the_date_abstains(live_db):
    """koap/st278 is stored from 2024 only; asking about 2020 must not substitute
    some other article's rule for the one the user named."""
    from lawrag.generate.answer import AnswerService

    answer = AnswerService(live_db).ask("что говорит статья 278 КоАП",
                                        as_of=date(2020, 6, 1), log_query=False)
    assert answer.abstained
    assert answer.abstain_reason == "cited_text_not_in_force_on_date"
    assert answer.unavailable_citations[0]["node_id"] == "koap/st278"
    assert answer.unavailable_citations[0]["earliest"] == "2024-01-01"


def test_calibration_finds_a_gap_on_the_fixture(live_db):
    from pathlib import Path

    from lawrag.evalkit.calibrate import calibrate

    questions = Path(__file__).resolve().parents[1] / "src" / "lawrag" / "evalkit" / "questions.sample.yaml"
    report = calibrate(live_db, questions)
    dense = next(s for s in report.signals if s.name == "dense")
    assert dense.answerable and dense.unanswerable
    assert dense.separable
    assert dense.highest_unanswerable < dense.suggested < dense.lowest_answerable
    # The live config is untouched by the probe.
    assert live_db.retrieval.min_dense_score == dense.current
