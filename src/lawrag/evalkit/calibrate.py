"""Relevance-floor calibration.

``retrieval.min_dense_score`` / ``min_lexical_score`` / ``min_trigram_score``
are what make the system abstain on a question the corpus cannot answer. They
are on the scales of the embedding model and of Postgres full-text ranking, so
values tuned for one model are meaningless for another — and a wrong floor
fails silently: set too low, every out-of-corpus question gets answered from
the least-bad match; set too high, answerable questions abstain.

This measures instead of guessing. It runs the golden question set with the
floor switched off and records, per signal:

* for each *answerable* question, the best score any expected article reached
  (the floor must stay below these), and
* for each *expect_abstain* question, the best score anything reached (the
  floor must stay above these).

A threshold is proposed in the gap between the two. When the two overlap, the
signal cannot separate the question set on its own, which is reported as such
rather than papered over with a number.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import Config
from ..models import Lang
from .harness import load_questions

SIGNALS = (
    ("dense", "dense_score", "min_dense_score"),
    ("lexical", "sparse_score", "min_lexical_score"),
    ("trigram", "trigram_score", "min_trigram_score"),
)


@dataclass(slots=True)
class SignalReport:
    name: str
    setting: str
    current: float
    answerable: list[tuple[str, float]] = field(default_factory=list)
    unanswerable: list[tuple[str, float]] = field(default_factory=list)

    @property
    def lowest_answerable(self) -> float | None:
        return min((s for _, s in self.answerable), default=None)

    @property
    def highest_unanswerable(self) -> float | None:
        return max((s for _, s in self.unanswerable), default=None)

    @property
    def separable(self) -> bool | None:
        lo, hi = self.lowest_answerable, self.highest_unanswerable
        if lo is None or hi is None:
            return None
        return hi < lo

    @property
    def suggested(self) -> float | None:
        lo, hi = self.lowest_answerable, self.highest_unanswerable
        if lo is None:
            # Nothing answerable depends on this signal, so there is no evidence
            # for moving it. A floor derived from out-of-corpus scores alone
            # could only ever be lowered towards them.
            return None
        if hi is None:
            # No out-of-corpus questions: nothing to keep out, so stay just
            # under the weakest answerable score. Add expect_abstain questions.
            return round(lo * 0.9, 4)
        if hi >= lo:
            # Cannot separate: favour abstaining, which is the safe failure.
            return round(hi * 1.02 + 1e-4, 4)
        # Three quarters of the way towards the weakest answerable score rather
        # than the midpoint: with a small question set the next out-of-corpus
        # question is more likely to land above the observed maximum than an
        # answerable one is to land below the observed minimum, and a wrong
        # abstention is the cheaper mistake.
        return round(hi + 0.75 * (lo - hi), 4)

    def verdict(self) -> str:
        if self.separable is None:
            if self.lowest_answerable is None and self.highest_unanswerable is None:
                return "no data"
            if self.highest_unanswerable is None:
                return "no expect_abstain questions — cannot check the floor keeps anything out"
            return "no answerable question reached this signal"
        if self.separable:
            return "separates the question set"
        return ("OVERLAP — this signal alone cannot tell answerable from out-of-corpus; "
                "the suggestion favours abstaining")


@dataclass(slots=True)
class CalibrationReport:
    signals: list[SignalReport]
    not_retrieved: list[tuple[str, list[str]]]
    questions: int
    embedding_backend: str


def calibrate(cfg: Config, path: Path) -> CalibrationReport:
    from ..retrieve.hybrid import HybridRetriever
    from ..retrieve.query import analyze
    from ..index.db import Database
    from ..index.embed import build_embedder

    probe = copy.deepcopy(cfg)
    for _, _, setting in SIGNALS:
        setattr(probe.retrieval, setting, float("-inf"))
    retriever = HybridRetriever(Database(probe), build_embedder(probe.embedding), probe.retrieval)

    reports = [SignalReport(name, setting, getattr(cfg.retrieval, setting))
               for name, _, setting in SIGNALS]
    not_retrieved: list[tuple[str, list[str]]] = []
    items = load_questions(path)

    for item in items:
        qid = str(item.get("id", "?"))
        as_of = item.get("as_of")
        if isinstance(as_of, str):
            as_of = datetime.strptime(as_of, "%Y-%m-%d").date()
        lang = Lang(item["lang"]) if item.get("lang") else None
        analysis = analyze(item["question"], as_of=as_of, lang=lang)
        hits = retriever.search(analysis)
        expected = set(item.get("expect") or [])

        if item.get("expect_abstain"):
            for report, (_, attr, _) in zip(reports, SIGNALS):
                # A signal that found nothing at all scored 0 — that is still
                # evidence about where the floor sits.
                best = max((getattr(h, attr) for h in hits), default=0.0)
                report.unanswerable.append((qid, float(best or 0.0)))
            continue

        relevant = [h for h in hits if h.article_node_id in expected]
        found = {h.article_node_id for h in relevant}
        if expected - found:
            not_retrieved.append((qid, sorted(expected - found)))
        for report, (_, attr, _) in zip(reports, SIGNALS):
            best = max((getattr(h, attr) for h in relevant), default=None)
            if best:
                report.answerable.append((qid, float(best)))

    return CalibrationReport(reports, not_retrieved, len(items), cfg.embedding.backend)


def print_calibration(report: CalibrationReport, console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    table = Table(title=f"relevance floor calibration — {report.questions} questions, "
                        f"embedding: {report.embedding_backend}")
    for col in ("signal", "current", "weakest answerable", "strongest out-of-corpus",
                "suggested", "verdict"):
        table.add_column(col)

    def fmt(v):
        return "—" if v is None else f"{v:.4f}"

    for s in report.signals:
        lo = s.lowest_answerable
        hi = s.highest_unanswerable
        lo_q = next((q for q, v in s.answerable if v == lo), "")
        hi_q = next((q for q, v in s.unanswerable if v == hi), "")
        table.add_row(s.name, fmt(s.current),
                      f"{fmt(lo)} {lo_q}".strip(), f"{fmt(hi)} {hi_q}".strip(),
                      fmt(s.suggested), s.verdict())
    console.print(table)

    if report.not_retrieved:
        console.print("[yellow]expected articles not retrieved even with the floor off "
                      "(a recall problem, not a threshold problem):[/yellow]")
        for qid, missing in report.not_retrieved:
            console.print(f"  {qid}: {', '.join(missing)}")

    if report.embedding_backend == "hash":
        console.print("[yellow]The hash embedder is a placeholder: dense scores measured "
                      "with it say nothing about bge-m3. Calibrate with the production "
                      "config.[/yellow]")

    console.print("\nretrieval:")
    for s in report.signals:
        if s.suggested is not None:
            console.print(f"  {s.setting}: {s.suggested}")
    console.print("[dim]A floor is tuned to the question set it was measured on. Use real "
                  "questions — at least a dozen answerable and a handful that the corpus "
                  "cannot answer — and re-run after changing the embedding model or the "
                  "corpus.[/dim]")
