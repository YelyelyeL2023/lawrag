"""Retrieval evaluation.

Measures the thing that actually determines answer quality: whether the correct
*set* of articles reached the model. Top-1 accuracy is close to meaningless for
legal questions, where the answer routinely lives across a penalty article, a
definition in another code, and an exception that refers out to a third.

Two recalls are reported side by side:

``recall@chunk``   fraction of expected articles that were directly retrieved
``recall@packet``  fraction that reached the model at all, counting hydration

The gap between them is the value hydration is adding. If it is zero, either the
citation graph is empty or the questions do not actually need it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from ..config import Config
from ..logging_setup import get_logger
from ..models import Lang
from ..retrieve.engine import RetrievalEngine

log = get_logger(__name__)


@dataclass(slots=True)
class QuestionResult:
    id: str
    question: str
    expected: list[str]
    retrieved_articles: list[str]
    packet_articles: list[str]
    hit_chunk: list[str] = field(default_factory=list)
    hit_packet: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    abstain_expected: bool = False
    abstained: bool = False
    superseded_expected: bool = False
    superseded_returned: bool = False
    rank_of_first: int | None = None
    ok: bool = False
    notes: str = ""


@dataclass(slots=True)
class EvalReport:
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def recall_chunk(self) -> float:
        return _mean([len(r.hit_chunk) / len(r.expected)
                      for r in self.results if r.expected])

    @property
    def recall_packet(self) -> float:
        return _mean([len(r.hit_packet) / len(r.expected)
                      for r in self.results if r.expected])

    @property
    def mrr(self) -> float:
        return _mean([1.0 / r.rank_of_first if r.rank_of_first else 0.0
                      for r in self.results if r.expected])

    @property
    def abstain_accuracy(self) -> float:
        cases = [r for r in self.results if r.abstain_expected]
        if not cases:
            return float("nan")
        return sum(1 for r in cases if r.abstained) / len(cases)

    @property
    def pass_rate(self) -> float:
        return _mean([1.0 if r.ok else 0.0 for r in self.results])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_questions(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def run_eval(cfg: Config, path: Path, top_k: int | None = None) -> EvalReport:
    engine = RetrievalEngine(cfg)
    if top_k:
        engine.cfg.retrieval.final_k = top_k
    report = EvalReport()

    for item in load_questions(path):
        expected = list(item.get("expect") or [])
        as_of = item.get("as_of")
        if isinstance(as_of, str):
            as_of = datetime.strptime(as_of, "%Y-%m-%d").date()
        lang = Lang(item["lang"]) if item.get("lang") else None

        result = engine.retrieve(item["question"], as_of=as_of, lang=lang)

        retrieved = [h.article_node_id for h in result.hits]
        packet: list[str] = []
        for hh in result.hydrated:
            packet.append(hh.hit.article_node_id)
            packet.extend(s.node_id for s in hh.supporting)

        hit_chunk = [e for e in expected if e in retrieved]
        hit_packet = [e for e in expected if e in packet]
        rank = next((i for i, a in enumerate(retrieved, 1) if a in expected), None)

        qr = QuestionResult(
            id=str(item.get("id", "?")),
            question=item["question"],
            expected=expected,
            retrieved_articles=retrieved,
            packet_articles=packet,
            hit_chunk=hit_chunk,
            hit_packet=hit_packet,
            missed=[e for e in expected if e not in packet],
            abstain_expected=bool(item.get("expect_abstain")),
            abstained=not result.packet.sources,
            superseded_expected=bool(item.get("expect_superseded")),
            superseded_returned=any(not s["is_current"] for s in result.packet.sources),
            rank_of_first=rank,
            notes=str(item.get("note", "")),
        )
        qr.ok = (
            (not qr.missed)
            and (qr.abstained == qr.abstain_expected)
            and (not qr.superseded_expected or qr.superseded_returned)
        )
        report.results.append(qr)

    return report


def print_report(report: EvalReport, console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    table = Table(title="retrieval eval")
    table.add_column("id")
    table.add_column("ok", justify="center")
    table.add_column("chunk", justify="right")
    table.add_column("packet", justify="right")
    table.add_column("rank", justify="right")
    table.add_column("missed")

    for r in report.results:
        denom = len(r.expected) or 1
        table.add_row(
            r.id,
            "[green]✓[/green]" if r.ok else "[red]✗[/red]",
            f"{len(r.hit_chunk)}/{denom}" if r.expected else "—",
            f"{len(r.hit_packet)}/{denom}" if r.expected else "—",
            str(r.rank_of_first or "—"),
            ", ".join(r.missed) or "",
        )
    console.print(table)
    console.print(
        f"recall@chunk  {report.recall_chunk:.2f}\n"
        f"recall@packet {report.recall_packet:.2f}   "
        f"[dim](the gap is what hydration is buying you)[/dim]\n"
        f"MRR           {report.mrr:.2f}\n"
        f"abstain acc.  {report.abstain_accuracy:.2f}\n"
        f"pass rate     {report.pass_rate:.2f}"
    )
