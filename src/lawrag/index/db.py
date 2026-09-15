from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from ..config import Config
from ..logging_setup import get_logger

log = get_logger(__name__)

def _find_schema() -> Path:
    """Locate the schema file packaged alongside lawrag/ (checkout or wheel)."""
    candidates = [Path(__file__).resolve().parents[1] / "sql" / "001_schema.sql"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "001_schema.sql not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


SCHEMA_PATH = None


class Database:
    def __init__(self, cfg: Config) -> None:
        self.dsn = cfg.db.dsn
        self.statement_timeout_ms = cfg.db.statement_timeout_ms

    @contextmanager
    def connect(self, autocommit: bool = False) -> Iterator[psycopg.Connection]:
        conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=autocommit)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
            yield conn
            if not autocommit:
                conn.commit()
        except Exception:
            if not autocommit:
                conn.rollback()
            raise
        finally:
            conn.close()

    def apply_schema(self, path: Path | None = None) -> None:
        sql = (Path(path) if path else _find_schema()).read_text(encoding="utf-8")
        with self.connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(sql)
        log.info("schema applied")

    def reset(self) -> None:
        """Drop everything lawrag owns. Used by tests and by a full re-import."""
        with self.connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                """
                DROP VIEW  IF EXISTS xref_unresolved CASCADE;
                DROP TABLE IF EXISTS query_log, term, xref, chunk_embedding, chunk,
                                     node_text, node, act, corpus_release CASCADE;
                """
            )
        log.warning("database reset")

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self.connect() as conn, conn.cursor() as cur:
            for table in ("act", "node", "node_text", "chunk", "chunk_embedding",
                          "xref", "term", "corpus_release"):
                cur.execute(f"SELECT count(*) AS n FROM {table}")
                out[table] = cur.fetchone()["n"]
        return out
