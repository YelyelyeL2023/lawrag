"""Writing a parsed corpus into Postgres.

Two things happen here that are easy to get wrong and expensive to discover
later.

*Version closing.* When a snapshot brings a new redaction of an article, the
previous redaction is not deleted — its ``valid_to`` is closed to the day before
the new one takes effect. Deleting superseded text would make the system unable
to answer "what was the rule when this happened", which for a compliance tool is
most of the hard questions.

*Alignment.* Russian and Kazakh chunks of the same node get pointed at each
other by primary key, because both were parsed onto the same structural
skeleton. No similarity matching, no drift.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Sequence

from ..ids import content_sha256
from ..logging_setup import get_logger
from ..models import (Act, Chunk, CorpusRelease, Lang, Node, NodeKind, OPEN_ENDED,
                      NORM_RANK, XRef)
from .db import Database
from .embed import Embedder

log = get_logger(__name__)

BATCH = 200


def _vec_literal(vec) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


class Indexer:
    def __init__(self, db: Database, embedder: Embedder) -> None:
        self.db = db
        self.embedder = embedder

    # ------------------------------------------------------------- release --
    def register_release(self, release: CorpusRelease) -> None:
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corpus_release (id, source, fetched_at, manifest_sha256, note)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET note = EXCLUDED.note
                """,
                (release.id, release.source, release.fetched_at,
                 release.manifest_sha256, release.note),
            )

    # ----------------------------------------------------------------- act --
    def upsert_act(self, act: Act) -> None:
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO act (code, kind, official_id, norm_rank, title_ru, title_kk,
                                 adopted_on, url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (code) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    official_id = COALESCE(NULLIF(EXCLUDED.official_id, ''), act.official_id),
                    norm_rank = EXCLUDED.norm_rank,
                    title_ru = COALESCE(NULLIF(EXCLUDED.title_ru, ''), act.title_ru),
                    title_kk = COALESCE(NULLIF(EXCLUDED.title_kk, ''), act.title_kk),
                    adopted_on = COALESCE(EXCLUDED.adopted_on, act.adopted_on),
                    url = COALESCE(EXCLUDED.url, act.url)
                """,
                (act.code, act.kind.value, act.official_id, NORM_RANK[act.kind],
                 act.title.get("ru", ""), act.title.get("kk", ""),
                 act.adopted_on, act.url),
            )

    # --------------------------------------------------------------- nodes --
    def upsert_tree(self, root: Node, lang: Lang, release_id: str) -> None:
        # Pre-order walk already yields every parent before its children, which
        # is what the self-referencing FK needs. Sorting by id depth would not:
        # an article is "koap/st278" while its parent chapter is
        # "koap/sec2/cha14", because articles are cited by act and number rather
        # than by their position in the hierarchy.
        nodes = list(root.walk())
        with self.db.connect() as conn, conn.cursor() as cur:
            for node in nodes:
                cur.execute(
                    """
                    INSERT INTO node (id, act_code, kind, number, parent_id, ordinal)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        kind = EXCLUDED.kind, number = EXCLUDED.number,
                        parent_id = EXCLUDED.parent_id, ordinal = EXCLUDED.ordinal
                    """,
                    (node.id, node.act_code, node.kind.value, node.number,
                     node.parent_id, node.ordinal),
                )
            for node in nodes:
                text = node.texts.get(lang)
                if text is None:
                    continue
                self._close_superseded(cur, node.id, lang, text.valid_from)
                valid_to = self._effective_valid_to(
                    cur, "node_text", node.id, lang, text.valid_from, text.valid_to)
                cur.execute(
                    """
                    INSERT INTO node_text
                        (node_id, lang, valid_from, valid_to, heading, body, sha256, release_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (node_id, lang, valid_from) DO UPDATE SET
                        valid_to = EXCLUDED.valid_to, heading = EXCLUDED.heading,
                        body = EXCLUDED.body, sha256 = EXCLUDED.sha256,
                        release_id = EXCLUDED.release_id
                    """,
                    (node.id, lang.value, text.valid_from, valid_to,
                     text.heading, text.body,
                     content_sha256(text.heading, text.body), release_id),
                )

    @staticmethod
    def _close_superseded(cur, node_id: str, lang: Lang, new_valid_from: date) -> None:
        """Close any open-ended earlier redaction the day before this one starts."""
        cur.execute(
            """
            UPDATE node_text
               SET valid_to = %s
             WHERE node_id = %s AND lang = %s
               AND valid_from < %s AND valid_to > %s
            """,
            (new_valid_from - timedelta(days=1), node_id, lang.value,
             new_valid_from, new_valid_from - timedelta(days=1)),
        )

    @staticmethod
    def _effective_valid_to(cur, table: str, node_id: str, lang: Lang,
                            valid_from: date, declared_valid_to: date) -> date:
        """Clamp a row's validity against any *later* redaction already stored.

        ``_close_superseded`` only looks backwards, so it is correct only when
        snapshots are imported oldest-first. Import a historic snapshot after a
        current one — which is just a matter of what the files happen to be
        called, since ingest globs and sorts filenames — and the historic text
        would be stored open-ended and answer as current law alongside the real
        current text.

        Computed *before* the insert, not patched up afterwards: the database
        rejects overlapping ranges outright (``node_text_no_overlap``), so an
        after-the-fact correction never gets the chance to run.
        """
        cur.execute(
            f"""
            SELECT MIN(valid_from) AS next_from FROM {table}
             WHERE node_id = %(node)s AND lang = %(lang)s AND valid_from > %(vf)s
            """,
            {"node": node_id, "lang": lang.value, "vf": valid_from},
        )
        row = cur.fetchone()
        next_from = row["next_from"] if row else None
        if next_from is None:
            return declared_valid_to
        return min(declared_valid_to, next_from - timedelta(days=1))

    def close_dropped(self, act_code: str, lang: Lang, snapshot_from: date) -> int:
        """Close norms the new snapshot no longer contains.

        A repealed article simply *disappears* from the next snapshot. Nothing
        in the per-node update path ever sees it again, so without this pass it
        stays open-ended and keeps answering as current law forever — the exact
        failure the as-of filter exists to prevent.
        """
        cutoff = snapshot_from - timedelta(days=1)
        closed = 0
        with self.db.connect() as conn, conn.cursor() as cur:
            for table in ("node_text", "chunk"):
                cur.execute(
                    f"""
                    UPDATE {table} SET valid_to = %(cutoff)s
                     WHERE lang = %(lang)s
                       AND valid_to > %(cutoff)s
                       AND valid_from < %(vf)s
                       AND node_id IN (SELECT id FROM node WHERE act_code = %(act)s)
                    """,
                    {"cutoff": cutoff, "lang": lang.value, "vf": snapshot_from,
                     "act": act_code},
                )
                closed += cur.rowcount
        if closed:
            log.info("closed %d superseded/dropped rows in %s [%s] as of %s",
                     closed, act_code, lang.value, cutoff)
        return closed

    # -------------------------------------------------------------- chunks --
    def index_chunks(self, chunks: Sequence[Chunk], release_id: str) -> int:
        if not chunks:
            return 0
        with self.db.connect() as conn, conn.cursor() as cur:
            for chunk in chunks:
                cur.execute(
                    """
                    UPDATE chunk SET valid_to = %s
                     WHERE node_id = %s AND lang = %s
                       AND valid_from < %s AND valid_to > %s
                    """,
                    (chunk.valid_from - timedelta(days=1), chunk.node_id,
                     chunk.lang.value, chunk.valid_from,
                     chunk.valid_from - timedelta(days=1)),
                )
                chunk_valid_to = self._effective_valid_to(
                    cur, "chunk", chunk.node_id, chunk.lang, chunk.valid_from,
                    chunk.valid_to)
                cur.execute(
                    """
                    INSERT INTO chunk (id, node_id, article_node_id, act_code, lang,
                                       valid_from, valid_to, heading, hierarchy_path,
                                       context_sentence, lead_in, repealed,
                                       display_text, embed_text,
                                       token_estimate, release_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        valid_to = EXCLUDED.valid_to,
                        heading = EXCLUDED.heading,
                        hierarchy_path = EXCLUDED.hierarchy_path,
                        context_sentence = EXCLUDED.context_sentence,
                        lead_in = EXCLUDED.lead_in,
                        repealed = EXCLUDED.repealed,
                        display_text = EXCLUDED.display_text,
                        embed_text = EXCLUDED.embed_text,
                        token_estimate = EXCLUDED.token_estimate,
                        release_id = EXCLUDED.release_id
                    """,
                    (chunk.id, chunk.node_id, chunk.article_node_id, chunk.act_code,
                     chunk.lang.value, chunk.valid_from, chunk_valid_to, chunk.heading,
                     chunk.hierarchy_path, chunk.context_sentence, chunk.lead_in,
                     chunk.repealed, chunk.display_text, chunk.embed_text,
                     chunk.token_estimate, release_id),
                )

        self._embed_chunks(chunks)
        return len(chunks)

    def _embed_chunks(self, chunks: Sequence[Chunk]) -> None:
        for start in range(0, len(chunks), BATCH):
            window = chunks[start:start + BATCH]
            vectors = self.embedder.encode([c.embed_text for c in window])
            with self.db.connect() as conn, conn.cursor() as cur:
                for chunk, vec in zip(window, vectors):
                    cur.execute(
                        """
                        INSERT INTO chunk_embedding (chunk_id, model, dim, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            model = EXCLUDED.model, dim = EXCLUDED.dim,
                            embedding = EXCLUDED.embedding
                        """,
                        (chunk.id, self.embedder.name, len(vec), _vec_literal(vec)),
                    )
            log.info("embedded %d/%d chunks", min(start + BATCH, len(chunks)), len(chunks))

    def link_aligned_chunks(self) -> int:
        """Point each chunk at its other-language twin — where one exists.

        Computed by join rather than by constructing the id string: an article
        long enough to split into points in Russian but not in Kazakh produces
        different node sets, and a constructed id would point at a chunk that
        was never created.
        """
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chunk c
                   SET aligned_chunk_id = t.id
                  FROM chunk t
                 WHERE t.node_id = c.node_id
                   AND t.valid_from = c.valid_from
                   AND t.lang <> c.lang
                """
            )
            linked = cur.rowcount
            cur.execute(
                "UPDATE chunk SET aligned_chunk_id = NULL "
                " WHERE aligned_chunk_id IS NOT NULL"
                "   AND NOT EXISTS (SELECT 1 FROM chunk x WHERE x.id = chunk.aligned_chunk_id)"
            )
        log.info("linked %d bilingual chunk pairs", linked)
        return linked

    # ---------------------------------------------------------------- refs --
    def index_xrefs(self, refs: Iterable[XRef], lang: Lang, release_id: str,
                    valid_from: date, valid_to: date) -> int:
        """Store the citation graph for one redaction.

        Scoped by validity, not wiped wholesale: an article's references change
        when it is amended, and deleting the old edges means a question asked
        as-of 2024 gets hydrated through the 2026 graph.
        """
        refs = list(refs)
        if not refs:
            return 0
        with self.db.connect() as conn, conn.cursor() as cur:
            src_ids = sorted({r.src_node_id for r in refs})
            cur.execute(
                "DELETE FROM xref WHERE lang = %s AND src_node_id = ANY(%s) "
                "  AND valid_from = %s",
                (lang.value, src_ids, valid_from),
            )
            cur.execute(
                """
                UPDATE xref SET valid_to = %s
                 WHERE lang = %s AND src_node_id = ANY(%s)
                   AND valid_from < %s AND valid_to > %s
                """,
                (valid_from - timedelta(days=1), lang.value, src_ids,
                 valid_from, valid_from - timedelta(days=1)),
            )
            for ref in refs:
                cur.execute(
                    """
                    INSERT INTO xref (src_node_id, lang, raw, dst_act_code, dst_article,
                                      dst_node_id, kind, valid_from, valid_to, release_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (ref.src_node_id, lang.value, ref.raw, ref.dst_act_code,
                     ref.dst_article, ref.dst_node_id, ref.kind,
                     valid_from, valid_to, release_id),
                )
        return len(refs)

    def index_terms(self, terms: Sequence[tuple[str, str, str, str, date, date]],
                    lang: Lang, release_id: str) -> int:
        """terms: (term, definition, act_code, node_id, valid_from, valid_to)."""
        if not terms:
            return 0
        with self.db.connect() as conn, conn.cursor() as cur:
            node_ids = sorted({t[3] for t in terms})
            valid_from = min(t[4] for t in terms)
            # Scoped to this redaction. An unscoped delete destroys the historic
            # definitions, and hydration for an as-of date in that period then
            # silently attaches nothing.
            cur.execute(
                "DELETE FROM term WHERE lang = %s AND node_id = ANY(%s) "
                "  AND valid_from = %s",
                (lang.value, node_ids, valid_from),
            )
            cur.execute(
                """
                UPDATE term SET valid_to = %s
                 WHERE lang = %s AND node_id = ANY(%s)
                   AND valid_from < %s AND valid_to > %s
                """,
                (valid_from - timedelta(days=1), lang.value, node_ids,
                 valid_from, valid_from - timedelta(days=1)),
            )
            for term, definition, act_code, node_id, vf, vt in terms:
                cur.execute(
                    """
                    INSERT INTO term (term, lang, act_code, node_id, definition,
                                      valid_from, valid_to, release_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (term, lang.value, act_code, node_id, definition, vf, vt, release_id),
                )
        return len(terms)

    # --------------------------------------------------------------- index --
    def build_vector_index(self) -> None:
        """Build the HNSW index after bulk load, not before.

        Building it first makes every insert pay maintenance cost for an index
        nobody is querying yet.
        """
        with self.db.connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM chunk_embedding")
            n = cur.fetchone()["n"]
            if n == 0:
                log.warning("no embeddings present; skipping vector index build")
                return
            log.info("building HNSW index over %d embeddings", n)
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw
                    ON chunk_embedding USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """
            )
            cur.execute("ANALYZE chunk_embedding")
            cur.execute("ANALYZE chunk")
        log.info("vector index ready")
