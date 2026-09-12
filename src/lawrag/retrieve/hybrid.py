"""Hybrid retrieval: dense + lexical, fused with reciprocal rank.

Lexical matters more here than in most RAG systems. Legal queries carry exact
article numbers and fixed statutory terminology, and a dense model will happily
rank article 277 next to article 278. Fusing ranks rather than scores avoids
having to calibrate a cosine against a ts_rank, which are not on the same scale
and never will be.

The as-of date is applied as a WHERE clause in every branch. It is not a
ranking feature and must never become one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..config import RetrievalConfig
from ..logging_setup import get_logger
from ..models import Lang
from ..index.db import Database
from ..index.embed import Embedder
from .query import QueryAnalysis

log = get_logger(__name__)

_SELECT = """
    c.id, c.node_id, c.article_node_id, c.act_code, c.lang, c.heading,
    c.hierarchy_path, c.context_sentence, c.display_text, c.repealed,
    c.valid_from, c.valid_to, c.aligned_chunk_id, c.release_id,
    a.norm_rank, a.title_ru, a.title_kk
"""


@dataclass(slots=True)
class Hit:
    chunk_id: str
    node_id: str
    article_node_id: str
    act_code: str
    lang: str
    heading: str
    hierarchy_path: str
    context_sentence: str
    display_text: str
    valid_from: date
    valid_to: date
    norm_rank: int
    act_title: str
    release_id: str = ""
    repealed: bool = False
    dense_rank: int | None = None
    sparse_rank: int | None = None
    exact_rank: int | None = None
    dense_score: float = 0.0
    sparse_score: float = 0.0
    trigram_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None
    provenance: list[str] = field(default_factory=list)

    @property
    def is_current(self) -> bool:
        return self.valid_to >= date(9999, 12, 31)


def _row_to_hit(row: dict) -> Hit:
    return Hit(
        chunk_id=row["id"], node_id=row["node_id"],
        article_node_id=row["article_node_id"], act_code=row["act_code"],
        lang=row["lang"], heading=row["heading"],
        hierarchy_path=row["hierarchy_path"],
        context_sentence=row["context_sentence"],
        display_text=row["display_text"], valid_from=row["valid_from"],
        valid_to=row["valid_to"], norm_rank=row["norm_rank"],
        act_title=row["title_ru"] if row["lang"] == "ru" else (row["title_kk"] or row["title_ru"]),
        release_id=row.get("release_id") or "",
        repealed=bool(row.get("repealed")),
    )


def _vec_literal(vec) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


_WORD = re.compile(r"[\w\u0400-\u04FF]{3,}", re.UNICODE)


def or_tsquery(text: str) -> str:
    """Build an OR tsquery from a natural-language question.

    ``websearch_to_tsquery`` and ``plainto_tsquery`` both AND every term, so a
    full sentence — "Какой штраф грозит малому предприятию за просрочку
    постановки на регистрационный учет?" — matches nothing at all, and the
    lexical branch silently contributes zero to every fused result. OR plus
    ts_rank_cd is the behaviour actually wanted: documents matching more of the
    query rank higher, documents matching some still surface.
    """
    words = {w.lower() for w in _WORD.findall(text)}
    return " | ".join(sorted(words))


class HybridRetriever:
    def __init__(self, db: Database, embedder: Embedder, cfg: RetrievalConfig) -> None:
        self.db = db
        self.embedder = embedder
        self.cfg = cfg

    def _langs(self, q: QueryAnalysis) -> list[str]:
        if not self.cfg.cross_language:
            return [q.lang.value]
        # Retrieve in both languages: the Russian and Kazakh texts of a norm can
        # differ in wording, and a query phrased in one language sometimes only
        # matches the other's vocabulary. The answer layer decides what to quote.
        return [Lang.RU.value, Lang.KK.value]

    # ------------------------------------------------------------- branches --
    def dense(self, q: QueryAnalysis, k: int) -> list[Hit]:
        vec = _vec_literal(self.embedder.encode([q.raw], is_query=True)[0])
        sql = f"""
            SELECT {_SELECT},
                   1 - (e.embedding <=> %(vec)s::vector) AS cosine_similarity
              FROM chunk c
              JOIN chunk_embedding e ON e.chunk_id = c.id
              JOIN act a ON a.code = c.act_code
             WHERE c.valid_from <= %(as_of)s AND c.valid_to >= %(as_of)s
               AND c.lang = ANY(%(langs)s)
               AND (%(acts)s::text[] IS NULL OR c.act_code = ANY(%(acts)s))
             ORDER BY e.embedding <=> %(vec)s::vector
             LIMIT %(k)s
        """
        params = {"as_of": q.as_of, "langs": self._langs(q),
                  "acts": q.act_hints or None, "vec": vec, "k": k}
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            hits = []
            for row in cur.fetchall():
                hit = _row_to_hit(row)
                hit.dense_score = float(row["cosine_similarity"])
                hits.append(hit)
            return hits

    def sparse(self, q: QueryAnalysis, k: int) -> list[list[Hit]]:
        """Full-text search — one independently ranked list per language.

        Returning a flat concatenation and then enumerating it as if it were one
        ranked list is a subtle scoring bug: the second language always starts
        at rank len(first)+1, so on a full corpus the entire Kazakh list lands
        at ranks 51-100 and contributes roughly half the RRF weight it should —
        including for Kazakh questions. The scores are not comparable across
        configurations either ('russian' vs 'simple'), which is exactly why RRF
        fuses ranks rather than scores; the ranks just have to be per-branch.

        Russian gets the built-in snowball stemmer. Kazakh has no Postgres
        dictionary and falls back to 'simple' — exact word forms only — which is
        weak for an agglutinative language; the word-similarity branch below
        partially covers the gap. Wiring in a proper Kazakh morphological
        analyser remains the single highest-value corpus improvement available.
        """
        branches: list[list[Hit]] = []
        for lang in self._langs(q):
            config = "russian" if lang == "ru" else "simple"
            tsq = or_tsquery(q.raw)
            if not tsq:
                continue
            sql = f"""
                SELECT {_SELECT}, ts_rank_cd(c.tsv, query) AS score
                  FROM chunk c
                  JOIN act a ON a.code = c.act_code,
                       to_tsquery('{config}', %(tsq)s) query
                 WHERE c.tsv @@ query
                   AND c.lang = %(lang)s
                   AND c.valid_from <= %(as_of)s AND c.valid_to >= %(as_of)s
                   AND (%(acts)s::text[] IS NULL OR c.act_code = ANY(%(acts)s))
                 ORDER BY score DESC
                 LIMIT %(k)s
            """
            params = {"tsq": tsq, "lang": lang, "as_of": q.as_of,
                      "acts": q.act_hints or None, "k": k}
            with self.db.connect() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            trigram = False
            if not rows and lang == "kk":
                rows = self._trigram(q, lang, k)
                trigram = bool(rows)
            branch: list[Hit] = []
            for row in rows:
                hit = _row_to_hit(row)
                if trigram:
                    hit.trigram_score = float(row["score"])
                else:
                    hit.sparse_score = float(row["score"])
                branch.append(hit)
            if branch:
                branches.append(branch)
        return branches

    def _trigram(self, q: QueryAnalysis, lang: str, k: int) -> list[dict]:
        """Fuzzy fallback for Kazakh, scored per matching window.

        ``similarity()`` and the ``%`` operator compare the query against the
        *whole* ~1200-character chunk, so even a verbatim quotation of a chunk's
        own opening scores below 0.15 and the branch can never fire.
        ``word_similarity`` / ``<%`` score a short query against its best
        matching window, which is the intended semantics.
        """
        sql = f"""
            SELECT {_SELECT}, word_similarity(%(q)s, c.embed_text) AS score
              FROM chunk c JOIN act a ON a.code = c.act_code
             WHERE c.lang = %(lang)s
               AND c.valid_from <= %(as_of)s AND c.valid_to >= %(as_of)s
               AND (%(acts)s::text[] IS NULL OR c.act_code = ANY(%(acts)s))
               AND %(q)s <%% c.embed_text
             ORDER BY score DESC
             LIMIT %(k)s
        """
        params = {"q": q.raw, "lang": lang, "as_of": q.as_of,
                  "acts": q.act_hints or None, "k": k}
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def exact_citations(self, q: QueryAnalysis) -> list[Hit]:
        """Direct lookup for articles the user named outright.

        A question that says "статья 278" is asking for article 278, and no
        similarity score should be able to outrank it.
        """
        if not q.article_hints:
            return []
        out: list[Hit] = []
        with self.db.connect() as conn, conn.cursor() as cur:
            for act, number in q.article_hints:
                sql = f"""
                    SELECT {_SELECT}
                      FROM chunk c JOIN act a ON a.code = c.act_code
                     WHERE c.article_node_id = ANY(%(nodes)s)
                       AND c.lang = ANY(%(langs)s)
                       AND c.valid_from <= %(as_of)s AND c.valid_to >= %(as_of)s
                     ORDER BY c.node_id
                """
                acts = [act] if act else (q.act_hints or list_all_acts(self.db))
                nodes = [f"{a}/st{number}" for a in acts]
                cur.execute(sql, {"nodes": nodes, "langs": self._langs(q),
                                  "as_of": q.as_of})
                out.extend(_row_to_hit(r) for r in cur.fetchall())
        return out

    # ---------------------------------------------------------------- fuse --
    def search(self, q: QueryAnalysis) -> list[Hit]:
        dense = self.dense(q, self.cfg.dense_k)
        sparse = self.sparse(q, self.cfg.sparse_k)
        exact = self.exact_citations(q)

        merged: dict[str, Hit] = {}
        for rank, hit in enumerate(dense, 1):
            target = merged.setdefault(hit.chunk_id, hit)
            target.dense_rank = rank
            target.dense_score = max(target.dense_score, hit.dense_score)
        for branch in sparse:
            for rank, hit in enumerate(branch, 1):
                target = merged.setdefault(hit.chunk_id, hit)
                # Best rank across branches: a chunk found by two branches
                # should not be penalised for the weaker one.
                target.sparse_rank = min(target.sparse_rank or rank, rank)
                target.sparse_score = max(target.sparse_score, hit.sparse_score)
                target.trigram_score = max(target.trigram_score, hit.trigram_score)
        for rank, hit in enumerate(exact, 1):
            merged.setdefault(hit.chunk_id, hit).exact_rank = rank

        k = self.cfg.rrf_k
        for hit in merged.values():
            score = 0.0
            if hit.dense_rank:
                score += 1.0 / (k + hit.dense_rank)
                hit.provenance.append("dense")
            if hit.sparse_rank:
                score += 1.0 / (k + hit.sparse_rank)
                hit.provenance.append("lexical")
            if hit.exact_rank:
                # Weighted above the fused branches on purpose: an explicit
                # citation is an instruction, not a hint.
                score += 2.0 / (k + hit.exact_rank)
                hit.provenance.append("citation")
            # Tie-break by legal force, so the Constitution outranks a
            # ministerial order when both are equally similar.
            score += (10 - hit.norm_rank) * 1e-6
            hit.fused_score = score

        ranked = sorted(merged.values(), key=lambda h: -h.fused_score)
        kept = self.apply_relevance_floor(ranked, q)
        log.debug("hybrid: dense=%d lexical=%d citation=%d fused=%d kept=%d",
                  len(dense), sum(len(b) for b in sparse), len(exact),
                  len(ranked), len(kept))
        return kept

    def apply_relevance_floor(self, hits: list[Hit], q: QueryAnalysis) -> list[Hit]:
        """Drop hits that are merely the least-bad match.

        Rank fusion always returns *something*: ask an employment-law corpus
        about subsoil royalties and it will still hand back its top eight
        articles, ranked. For a compliance tool that is the dangerous failure —
        the model is then handed irrelevant law and asked to answer anyway.
        A hit has to clear an absolute bar, not just a relative one.

        The thresholds are scale-dependent on the embedding model and must be
        re-tuned when it changes; see ``retrieval.min_dense_score`` in the config.
        """
        floor: list[Hit] = []
        for hit in hits:
            if hit.exact_rank is not None:
                floor.append(hit)          # the user named this article outright
                continue
            if hit.dense_score >= self.cfg.min_dense_score:
                floor.append(hit)
                continue
            if hit.sparse_score >= self.cfg.min_lexical_score:
                floor.append(hit)
                continue
            # Word-similarity scores are on 0..1 and unrelated to ts_rank, so
            # they need their own threshold rather than reusing the FTS one.
            if hit.trigram_score >= self.cfg.min_trigram_score:
                floor.append(hit)
        return floor


def list_all_acts(db: Database) -> list[str]:
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT code FROM act ORDER BY norm_rank, code")
        return [r["code"] for r in cur.fetchall()]
