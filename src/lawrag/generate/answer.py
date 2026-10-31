"""Answer assembly: retrieve -> generate -> verify -> log."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import date

from ..config import Config
from ..index.db import Database
from ..logging_setup import get_logger
from ..models import Lang, OPEN_ENDED
from ..retrieve.engine import RetrievalEngine, RetrievalResult
from ..verify.citations import CitationVerifier, Verdict
from . import prompts
from .llm import LLM, build_llm

log = get_logger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class Answer:
    question: str
    lang: str
    as_of: str
    text: str = ""
    citations: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    confidence: str = "low"
    abstained: bool = False
    abstain_reason: str = ""
    escalate: bool = False
    verified: bool = True
    verification: str = ""
    verification_problems: list[str] = field(default_factory=list)
    uses_superseded_text: bool = False
    # Named articles with no text on the as-of date: [{node_id, earliest, latest}]
    unavailable_citations: list[dict] = field(default_factory=list)
    disclaimer: str = ""
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_model_json(raw: str) -> dict:
    """Recover the JSON object from a model response.

    Local quantised models wrap JSON in prose or a code fence often enough that
    failing hard on the first stray character would make the system look broken
    when it is merely chatty.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if m := _JSON_BLOCK.search(text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("model did not return parseable JSON")


class AnswerService:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.engine = RetrievalEngine(cfg)
        self.db = Database(cfg)
        self._llm: LLM | None = None
        self.verifier = CitationVerifier(self.db)

    @property
    def llm(self) -> LLM:
        # Built on first use, not here: the API's /search goes through this
        # service for its engine, and retrieval must keep working when the
        # generator is misconfigured (no API key) — that failure belongs to
        # /ask, where it becomes a generation_failed abstention.
        if self._llm is None:
            self._llm = build_llm(self.cfg.llm)
        return self._llm

    @llm.setter
    def llm(self, value: LLM) -> None:
        self._llm = value

    def ask(self, question: str, *, as_of: date | None = None,
            lang: Lang | None = None, log_query: bool = True) -> Answer:
        started = time.perf_counter()
        result = self.engine.retrieve(question, as_of=as_of, lang=lang)
        qlang = result.analysis.lang
        answer = Answer(
            question=question,
            lang=qlang.value,
            as_of=result.analysis.as_of.isoformat(),
            sources=result.packet.sources,
            disclaimer=prompts.DISCLAIMER[qlang],
            timings_ms=dict(result.timings_ms),
            unavailable_citations=[
                {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in u.items()}
                for u in result.unavailable_citations
            ],
        )

        # The question named articles, and none of them has text on the date.
        # Answering from whatever similar provisions came back would put a
        # different article's rule in place of the one the user asked about.
        def is_named(node_id: str) -> bool:
            return any(node_id == f"{a}/st{n}" if a else node_id.endswith(f"/st{n}")
                       for a, n in result.analysis.article_hints)

        if (result.analysis.is_citation_lookup and answer.unavailable_citations
                and not any(is_named(h.article_node_id) for h in result.hits)):
            answer.abstained = True
            answer.abstain_reason = "cited_text_not_in_force_on_date"
            answer.escalate = True
            answer.text = prompts.no_text_on_date(qlang, result.analysis.as_of,
                                                  answer.unavailable_citations)
            answer.sources = []
            self._finish(answer, result, started, log_query)
            return answer

        if not result.packet.sources:
            answer.abstained = True
            answer.abstain_reason = "no_matching_provisions"
            answer.escalate = True
            answer.text = prompts.NO_SOURCES[qlang]
            self._finish(answer, result, started, log_query)
            return answer

        prompt = prompts.build_prompt(question, result.packet.text, qlang,
                                      result.analysis.as_of)
        t = time.perf_counter()
        try:
            raw = self.llm.complete(prompt, max_tokens=self.cfg.llm.max_tokens)
            payload = _parse_model_json(raw)
        except Exception as exc:  # noqa: BLE001
            log.error("generation failed: %s", exc)
            answer.abstained = True
            answer.abstain_reason = f"generation_failed: {exc}"
            answer.escalate = True
            # Retrieval found provisions; it is the generator that failed.
            # NO_SOURCES here would tell the user the law is silent.
            answer.text = prompts.GENERATION_FAILED[qlang]
            answer.verified = False
            answer.verification = "not_verified: generation_failed"
            self._finish(answer, result, started, log_query)
            return answer
        answer.timings_ms["generate"] = int((time.perf_counter() - t) * 1000)

        answer.text = (payload.get("answer") or "").strip()
        answer.citations = list(payload.get("citations") or [])
        answer.confidence = str(payload.get("confidence") or "low")
        answer.abstained = bool(payload.get("abstain"))
        answer.abstain_reason = str(payload.get("abstain_reason") or "")
        answer.escalate = bool(payload.get("escalate"))

        if answer.abstained:
            if not answer.text:
                # "No provisions found" and "no model is loaded" are different
                # facts, and conflating them tells an operator the corpus is
                # empty when it is the runtime that is not configured.
                answer.text = (
                    prompts.NO_MODEL[qlang]
                    if answer.abstain_reason == "no_model_loaded"
                    else prompts.NO_SOURCES[qlang]
                )
            answer.escalate = True
            answer.verification = f"not_verified: {answer.abstain_reason}"
            self._finish(answer, result, started, log_query)
            return answer

        t = time.perf_counter()
        verdict = self.verifier.verify(answer.citations, result.packet.sources,
                                       result.analysis.as_of)
        answer.timings_ms["verify"] = int((time.perf_counter() - t) * 1000)
        self._apply_verdict(answer, verdict, qlang)

        answer.uses_superseded_text = any(
            not s["is_current"] for s in result.packet.sources
        )
        self._finish(answer, result, started, log_query)
        return answer

    # ---------------------------------------------------------------------
    @staticmethod
    def _apply_verdict(answer: Answer, verdict: Verdict, lang: Lang) -> None:
        answer.verification = verdict.summary
        answer.verification_problems = verdict.reasons
        answer.verified = verdict.passed

        if not answer.citations:
            # An unsourced conclusion is exactly what this system exists to
            # prevent, however confident the model sounded.
            answer.verified = False
            answer.confidence = "low"
            answer.escalate = True
            answer.verification = "no_citations"
            answer.text = _downgrade_text(answer, lang)
            return

        if verdict.downgrade:
            answer.confidence = "low"
            answer.escalate = True
            answer.text = _downgrade_text(answer, lang)

    def _finish(self, answer: Answer, result: RetrievalResult, started: float,
                log_query: bool) -> None:
        answer.timings_ms["total"] = int((time.perf_counter() - started) * 1000)
        if log_query:
            self._log(answer, result)

    def _log(self, answer: Answer, result: RetrievalResult) -> None:
        try:
            with self.db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_log (question, lang, as_of, release_id, retrieved,
                                           cited, verdict, abstained, answer, latency_ms)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s)
                    """,
                    (answer.question, answer.lang, answer.as_of,
                     result.packet.sources[0].get("release_id") if result.packet.sources else None,
                     json.dumps([s["chunk_id"] for s in result.packet.sources]),
                     json.dumps(answer.citations, ensure_ascii=False),
                     answer.verification, answer.abstained, answer.text,
                     answer.timings_ms.get("total", 0)),
                )
        except Exception as exc:  # noqa: BLE001 — auditing must not break answering
            log.error("failed to write query log: %s", exc)


_DOWNGRADE = {
    Lang.RU: ("Автоматическая проверка ссылок не подтвердила ответ, поэтому вывод не "
              "приводится. Ниже — нормы, которые относятся к вопросу; прочитайте их "
              "сами и при необходимости обратитесь к юристу."),
    Lang.KK: ("Сілтемелерді автоматты тексеру жауапты растамады, сондықтан қорытынды "
              "берілмейді. Төменде сұраққа қатысты нормалар келтірілген; оларды өзіңіз "
              "оқып шығыңыз, қажет болса заңгерге жүгініңіз."),
}


_STATUS = {
    Lang.RU: {True: "действует", False: "прежняя редакция", "repealed": "исключена"},
    Lang.KK: {True: "қолданылады", False: "бұрынғы редакция", "repealed": "алып тасталды"},
}


def _downgrade_text(answer: Answer, lang: Lang) -> str:
    words = _STATUS[lang]
    lines = []
    for s in answer.sources:
        status = words["repealed"] if s.get("repealed") else words[bool(s["is_current"])]
        lines.append(f"— {s['heading']} ({s['node_id']}, {status})")
    return f"{_DOWNGRADE[lang]}\n\n" + "\n".join(lines)
