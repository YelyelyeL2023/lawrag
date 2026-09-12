"""The citation verifier — the layer that catches what a non-lawyer cannot."""

from __future__ import annotations

import json
from datetime import date

import pytest

from lawrag.generate.answer import AnswerService, _parse_model_json
from lawrag.models import Lang
from lawrag.verify.citations import CitationVerifier, normalize


class ScriptedLLM:
    """Returns a canned payload, so verification can be tested without weights."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_prompt = ""

    def complete(self, prompt: str, **_kw) -> str:
        self.last_prompt = prompt
        return json.dumps(self.payload, ensure_ascii=False)


def test_parse_model_json_tolerates_fences_and_prose():
    assert _parse_model_json('```json\n{"a": 1}\n```')["a"] == 1
    assert _parse_model_json('Вот ответ:\n{"a": 2}\nГотово.')["a"] == 2
    with pytest.raises(ValueError):
        _parse_model_json("no json here at all")


def test_quote_normalisation_ignores_cosmetic_differences():
    assert normalize("«Срок»  —  10 дней") == normalize('"Срок" - 10 дней')


@pytest.mark.parametrize("citation,expect_ok,expect_problem", [
    # Faithful: right article, actually retrieved, quoted verbatim.
    ({"source": 1, "article": "76",
      "quote": "не позднее десяти рабочих дней со дня начала осуществления деятельности"},
     True, None),
    # The dangerous one: a real article the model was never shown.
    ({"source": 99, "act_code": "tk", "article": "91",
      "quote": "работодатель обязан вести учет рабочего времени"},
     False, "not among the retrieved sources"),
    # Right article, invented quote.
    ({"source": 1, "article": "76",
      "quote": "срок составляет девяносто календарных дней с момента регистрации"},
     False, "verbatim"),
])
def test_verifier_catches_each_failure_mode(live_db, citation, expect_ok, expect_problem):
    service = AnswerService(live_db)
    result = service.engine.retrieve(
        "В какой срок нужно подать заявление о постановке на регистрационный учет?"
    )
    sources = result.packet.sources
    assert sources, "sample corpus should retrieve something for this question"

    verifier = CitationVerifier(service.db)
    verdict = verifier.verify([citation], sources, date.today())
    check = verdict.checks[0]

    assert check.ok is expect_ok, check.problems
    if expect_problem:
        assert any(expect_problem in p for p in check.problems), check.problems


def test_unverified_answer_is_downgraded_not_published(live_db):
    service = AnswerService(live_db)
    service.llm = ScriptedLLM({
        "answer": "Штраф составляет 500 МРП.",
        "citations": [{"source": 1, "article": "76",
                       "quote": "штраф в размере пятисот месячных расчетных показателей"}],
        "confidence": "high",
        "abstain": False,
        "escalate": False,
    })
    answer = service.ask("Какой штраф за просрочку постановки на учет?", log_query=False)

    assert not answer.verified
    assert answer.confidence == "low"
    assert answer.escalate
    # The invented conclusion must not reach the user.
    assert "500 МРП" not in answer.text
    assert answer.sources


def test_answer_with_no_citations_is_downgraded(live_db):
    service = AnswerService(live_db)
    service.llm = ScriptedLLM({
        "answer": "Срок составляет десять рабочих дней.",
        "citations": [],
        "confidence": "high",
        "abstain": False,
        "escalate": False,
    })
    answer = service.ask("В какой срок нужно встать на учет?", log_query=False)
    assert not answer.verified
    assert answer.verification == "no_citations"
    assert answer.escalate


def test_faithful_answer_passes_verification(live_db):
    service = AnswerService(live_db)
    service.llm = ScriptedLLM({
        "answer": "Заявление подается не позднее десяти рабочих дней со дня начала "
                  "деятельности; для малого бизнеса срок продлевается до двадцати дней.",
        "citations": [{
            "source": 1, "act": "Налоговый кодекс", "article": "76", "point": "1",
            "quote": "не позднее десяти рабочих дней со дня начала осуществления деятельности",
        }],
        "confidence": "high",
        "abstain": False,
        "escalate": False,
    })
    answer = service.ask(
        "В какой срок нужно подать заявление о постановке на регистрационный учет?",
        log_query=False,
    )
    assert answer.verified, answer.verification_problems
    assert "десяти рабочих дней" in answer.text
    assert answer.disclaimer


def test_echo_backend_abstains_rather_than_inventing(live_db):
    service = AnswerService(live_db)
    answer = service.ask("В какой срок нужно встать на учет?", log_query=False)
    assert answer.abstained
    assert answer.abstain_reason == "no_model_loaded"
    assert answer.escalate
