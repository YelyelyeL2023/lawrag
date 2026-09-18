"""The retrieval pipeline, end to end."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

from ..config import Config
from ..index.db import Database
from ..index.embed import build_embedder
from ..logging_setup import get_logger
from ..models import Lang
from .hybrid import HybridRetriever, Hit
from .hydrate import Hydrator, HydratedHit
from .packet import Packet, build_packet
from .query import QueryAnalysis, analyze
from .rerank import build_reranker

log = get_logger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    analysis: QueryAnalysis
    hits: list[Hit]
    hydrated: list[HydratedHit]
    packet: Packet
    timings_ms: dict[str, int]
    # Articles the question named that exist but have no text on the as-of date.
    unavailable_citations: list[dict] = field(default_factory=list)


class RetrievalEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.db = Database(cfg)
        self.embedder = build_embedder(cfg.embedding)
        self.retriever = HybridRetriever(self.db, self.embedder, cfg.retrieval)
        self.reranker = build_reranker(cfg.rerank)
        self.hydrator = Hydrator(self.db, cfg.retrieval)

    def retrieve(self, question: str, *, as_of: date | None = None,
                 lang: Lang | None = None) -> RetrievalResult:
        timings: dict[str, int] = {}
        t0 = time.perf_counter()

        analysis = analyze(question, as_of=as_of, lang=lang)
        timings["analyze"] = int((time.perf_counter() - t0) * 1000)

        t = time.perf_counter()
        hits = self.retriever.search(analysis)
        timings["search"] = int((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        top = self.reranker.rerank(analysis.raw, hits, self.cfg.retrieval.final_k)
        timings["rerank"] = int((time.perf_counter() - t) * 1000)

        top = self._dedupe_languages(top, analysis.lang)

        t = time.perf_counter()
        hydrated = self.hydrator.hydrate(top, analysis.as_of)
        timings["hydrate"] = int((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        packet = build_packet(hydrated, analysis.lang, analysis.as_of,
                              self.cfg.retrieval.max_packet_tokens)
        timings["packet"] = int((time.perf_counter() - t) * 1000)
        timings["total"] = int((time.perf_counter() - t0) * 1000)

        unavailable = self.retriever.citations_without_text(analysis)
        log.info("retrieved %d hits for %r (as of %s) in %dms",
                 len(top), question[:60], analysis.as_of, timings["total"])
        return RetrievalResult(analysis, top, hydrated, packet, timings, unavailable)

    @staticmethod
    def _dedupe_languages(hits: list[Hit], prefer: Lang) -> list[Hit]:
        """Both language faces of one norm are one source, not two.

        Cross-language retrieval is worth having, but handing the model the
        Russian and Kazakh text of the same article as separate sources wastes
        the context budget and invites it to cite the same norm twice.
        """
        best: dict[str, Hit] = {}
        for hit in hits:
            # Every hit is in force on the same as-of date, so one node id is one
            # norm. Keying on valid_from as well kept both faces whenever the RU
            # and KK footnotes dated the same article differently.
            key = hit.node_id
            current = best.get(key)
            if current is None:
                best[key] = hit
                continue
            if hit.lang == prefer.value and current.lang != prefer.value:
                best[key] = hit
        return sorted(best.values(), key=lambda h: -(h.rerank_score
                                                     if h.rerank_score is not None
                                                     else h.fused_score))
