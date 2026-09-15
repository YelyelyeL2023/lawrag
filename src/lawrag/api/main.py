"""HTTP API.

Deliberately small. The endpoints mirror the CLI so that whatever the UI does,
an operator can reproduce it from a shell — which matters a great deal when the
machine is air-gapped and you cannot attach a debugger from your laptop.

The operator console (the browser UI) is served from ``static/`` at ``/``, and
the extra read-only endpoints it needs live in ``console.py`` rather than here.
Nothing in this module depends on the UI: delete ``static/`` and the API is
exactly what it was.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import load_config
from ..generate.answer import AnswerService
from ..index.db import Database
from ..logging_setup import get_logger, setup_logging
from ..models import Lang
from . import console

log = get_logger(__name__)

cfg = load_config()
setup_logging(cfg.log_level)

app = FastAPI(
    title="lawrag",
    version="0.1.0",
    description=("Retrieval over the Constitution and Codes of the Republic of "
                 "Kazakhstan. Returns statutory text with pinpoint citations. "
                 "Not legal advice."),
)

_service: AnswerService | None = None


def service() -> AnswerService:
    """Build the answering stack once, and report a failure as a failure.

    Constructing this loads the embedder, the reranker and the LLM client. When
    one of them cannot be imported — most often because the server was started
    from an interpreter that does not have the weights stack installed — the
    only useful thing to do is say which module is missing and from which
    Python. Letting the ImportError escape produces a bare 500 and a traceback
    in the server log, which tells the person at the browser nothing at all.
    """
    global _service
    if _service is None:
        try:
            _service = AnswerService(cfg)
        except Exception as exc:  # noqa: BLE001
            detail = _startup_failure(exc)
            console.record_answering_error(detail)
            log.error("answering stack unavailable: %s", detail)
            raise HTTPException(503, detail) from exc
        console.clear_answering_error()
    return _service


def _startup_failure(exc: Exception) -> str:
    """Turn a load failure into something an operator can act on."""
    base = f"answering stack unavailable: {type(exc).__name__}: {exc}"
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        return (f"{base}. Module {exc.name!r} is not installed in "
                f"{sys.executable}. The weights stack is the optional `models` "
                f"extra: install `lawrag[models]` into that interpreter, or set "
                f"embedding.backend / llm.backend back to the placeholders.")
    return base


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    as_of: Optional[date] = Field(
        None, description="Answer as the law stood on this date. Defaults to today."
    )
    lang: Optional[Literal["ru", "kk"]] = None


class SearchRequest(AskRequest):
    include_packet: bool = False


@app.get("/health")
def health() -> dict:
    db = Database(cfg)
    try:
        counts = db.stats()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"database unavailable: {exc}") from exc
    return {
        "status": "ok",
        "corpus": counts,
        "embedding_backend": cfg.embedding.backend,
        "llm_backend": cfg.llm.backend,
        # Surfaced on purpose: an operator hitting /health must be able to see
        # that the system is running on placeholder models.
        "production_ready": cfg.embedding.backend != "hash" and cfg.llm.backend != "echo",
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    answer = service().ask(req.question, as_of=req.as_of,
                           lang=Lang(req.lang) if req.lang else None)
    return answer.to_dict()


@app.post("/search")
def search(req: SearchRequest) -> dict:
    result = service().engine.retrieve(
        req.question, as_of=req.as_of, lang=Lang(req.lang) if req.lang else None
    )
    payload = {
        "as_of": result.analysis.as_of.isoformat(),
        "lang": result.analysis.lang.value,
        "act_hints": result.analysis.act_hints,
        "hits": [
            {
                "chunk_id": h.chunk_id,
                "node_id": h.node_id,
                "article_node_id": h.article_node_id,
                "act_code": h.act_code,
                # act_title and hierarchy_path are what make a hit readable to a
                # human rather than only to whoever wrote the ingest code.
                "act_title": h.act_title,
                "hierarchy_path": h.hierarchy_path,
                "heading": h.heading,
                "lang": h.lang,
                "repealed": h.repealed,
                "valid_from": h.valid_from.isoformat(),
                "valid_to": h.valid_to.isoformat(),
                "is_current": h.is_current,
                "dense_score": round(h.dense_score, 4),
                "lexical_score": round(h.sparse_score, 4),
                "fused_score": round(h.fused_score, 6),
                "rerank_score": h.rerank_score,
                "provenance": h.provenance,
                "text": h.display_text,
            }
            for h in result.hits
        ],
        "sources": result.packet.sources,
        "packet_tokens": result.packet.token_estimate,
        "packet_truncated": result.packet.truncated,
        "dropped_sources": result.packet.dropped_sources,
        "timings_ms": result.timings_ms,
    }
    if req.include_packet:
        payload["packet"] = result.packet.text
    return payload


@app.get("/article/{act_code}/{number}")
def article(act_code: str, number: str, as_of: Optional[date] = None) -> dict:
    """Direct point-in-time lookup, no search involved."""
    when = as_of or date.today()
    db = Database(cfg)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT lang, heading, body, valid_from, valid_to
              FROM node_text
             WHERE node_id = %s AND valid_from <= %s AND valid_to >= %s
            """,
            (f"{act_code}/st{number}", when, when),
        )
        rows = cur.fetchall()
    if not rows:
        raise HTTPException(404, f"{act_code}/st{number} not in force on {when}")
    return {
        "node_id": f"{act_code}/st{number}",
        "as_of": when.isoformat(),
        "texts": [
            {"lang": r["lang"], "heading": r["heading"], "body": r["body"],
             "valid_from": r["valid_from"].isoformat(),
             "valid_to": r["valid_to"].isoformat()}
            for r in rows
        ],
    }


# ------------------------------------------------------------------ console --
app.include_router(console.bind(cfg), prefix="/api", tags=["console"])

_STATIC = Path(__file__).resolve().parent / "static"

if _STATIC.is_dir():
    # Mounted under a prefix so it can never shadow an API route: a future
    # endpoint named after a file in static/ should win, and with the SPA
    # mounted at "/" it would not.
    app.mount("/ui", StaticFiles(directory=_STATIC, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")
else:  # pragma: no cover — a source checkout without the console
    @app.get("/", include_in_schema=False)
    def index() -> JSONResponse:
        return JSONResponse({"service": "lawrag", "docs": "/docs",
                             "console": "not installed"})
