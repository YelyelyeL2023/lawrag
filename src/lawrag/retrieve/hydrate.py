"""Reference and definition hydration — the step most pipelines skip.

A retrieved point often says less than it appears to: "в порядке, установленном
статьей 76 Налогового кодекса" is the whole rule, and article 76 is somewhere
else. Hydration walks the citation graph one hop and pulls in what the hit
actually depends on, plus the definitions of any statutory terms it uses, plus
the parent article when only a point was retrieved.

One hop, deliberately. Two hops pulls in most of the corpus and buries the
answer in supporting material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..config import RetrievalConfig
from ..logging_setup import get_logger
from ..index.db import Database
from .hybrid import Hit

log = get_logger(__name__)

MAX_REFS_PER_HIT = 4
MAX_TERMS_PER_HIT = 4


@dataclass(slots=True)
class Supporting:
    kind: str            # "article" | "reference" | "definition"
    label: str
    node_id: str
    act_code: str
    text: str
    valid_from: date | None = None
    valid_to: date | None = None


@dataclass(slots=True)
class HydratedHit:
    hit: Hit
    article_text: str = ""
    article_heading: str = ""
    supporting: list[Supporting] = field(default_factory=list)


class Hydrator:
    def __init__(self, db: Database, cfg: RetrievalConfig) -> None:
        self.db = db
        self.cfg = cfg

    def hydrate(self, hits: list[Hit], as_of: date) -> list[HydratedHit]:
        out: list[HydratedHit] = []
        seen_support: set[str] = set()
        for hit in hits:
            hydrated = HydratedHit(hit=hit)
            self._parent_article(hydrated, as_of)
            if self.cfg.hydrate_refs:
                self._references(hydrated, as_of, seen_support)
            if self.cfg.hydrate_definitions:
                self._definitions(hydrated, as_of, seen_support)
            out.append(hydrated)
        return out

    # ---------------------------------------------------------------------
    def _parent_article(self, h: HydratedHit, as_of: date) -> None:
        """Small-to-big: the point was indexed, the article is what gets read."""
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT heading, body FROM node_text
                 WHERE node_id = %s AND lang = %s
                   AND valid_from <= %s AND valid_to >= %s
                 ORDER BY valid_from DESC LIMIT 1
                """,
                (h.hit.article_node_id, h.hit.lang, as_of, as_of),
            )
            row = cur.fetchone()
        if row:
            h.article_heading = row["heading"]
            h.article_text = row["body"]

    def _references(self, h: HydratedHit, as_of: date, seen: set[str]) -> None:
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT x.dst_node_id, x.dst_act_code, x.raw
                  FROM xref x
                 WHERE x.src_node_id IN (%s, %s)
                   AND x.lang = %s
                   AND x.dst_node_id IS NOT NULL
                   AND x.valid_from <= %s AND x.valid_to >= %s
                 LIMIT %s
                """,
                (h.hit.node_id, h.hit.article_node_id, h.hit.lang, as_of, as_of,
                 MAX_REFS_PER_HIT),
            )
            refs = cur.fetchall()

            for ref in refs:
                key = f"ref:{ref['dst_node_id']}:{h.hit.lang}"
                if key in seen:
                    continue
                cur.execute(
                    """
                    SELECT heading, body FROM node_text
                     WHERE node_id = %s AND lang = %s
                       AND valid_from <= %s AND valid_to >= %s
                     ORDER BY valid_from DESC LIMIT 1
                    """,
                    (ref["dst_node_id"], h.hit.lang, as_of, as_of),
                )
                row = cur.fetchone()
                if not row or not (row["body"] or row["heading"]):
                    continue
                seen.add(key)
                h.supporting.append(Supporting(
                    kind="reference",
                    label=f"{ref['dst_node_id']} — {row['heading']}".strip(" —"),
                    node_id=ref["dst_node_id"],
                    act_code=ref["dst_act_code"] or "",
                    text=row["body"][:2500],
                ))

    def _definitions(self, h: HydratedHit, as_of: date, seen: set[str]) -> None:
        """Attach definitions of statutory terms that actually appear in the hit."""
        haystack = f"{h.hit.heading} {h.hit.display_text}".lower()
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT term, definition, act_code, node_id FROM term
                 WHERE lang = %s AND valid_from <= %s AND valid_to >= %s
                """,
                (h.hit.lang, as_of, as_of),
            )
            rows = cur.fetchall()

        # Longest term first: "субъект малого предпринимательства" should win
        # over a bare "субъект" if both are defined.
        found = 0
        for row in sorted(rows, key=lambda r: -len(r["term"])):
            if found >= MAX_TERMS_PER_HIT:
                break
            if row["term"] not in haystack:
                continue
            key = f"term:{row['node_id']}:{row['term']}"
            if key in seen:
                continue
            seen.add(key)
            found += 1
            h.supporting.append(Supporting(
                kind="definition",
                label=f"«{row['term']}» — {row['node_id']}",
                node_id=row["node_id"],
                act_code=row["act_code"],
                text=row["definition"],
            ))
