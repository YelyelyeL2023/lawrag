"""Endpoints the operator console needs, over and above the answering API.

Kept in its own module so that ``api/main.py`` stays what it says it is: the
small, CLI-mirroring surface. Nothing here is required to answer a question —
if this router failed to import, ``/ask`` would still work.

Three constraints shaped the design:

* No new dependencies. The target machine is air-gapped and every package has
  to be carried across as a wheel, so file upload is JSON-with-text rather than
  multipart (which would pull in python-multipart).
* Read-only by default. Everything except ``POST /ingest`` only reads, so the
  console can be pointed at a production instance without a second thought.
* Degraded state is data, not an error. Placeholder backends, an unreachable
  database and an empty corpus are all reported as facts the UI can render,
  because an operator's first question is always "is this thing actually
  working", and a blank page does not answer it.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import Config
from ..index.db import Database
from ..logging_setup import get_logger
from . import security

log = get_logger(__name__)

router = APIRouter()

_cfg: Config | None = None


def bind(cfg: Config) -> APIRouter:
    """Give the router the process-wide config. Called once from main.py."""
    global _cfg
    _cfg = cfg
    return router


def cfg() -> Config:
    if _cfg is None:  # pragma: no cover — main.py binds at import time
        raise RuntimeError("console router used before bind()")
    return _cfg


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    db = Database(cfg())
    try:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params or None)
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"database unavailable: {exc}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


# --------------------------------------------------- runtime readiness --
# Config says which backends to use; it cannot say whether the interpreter
# running the server can actually load them. Those are different facts, and
# conflating them is how a system reports "production ready" and then returns
# HTTP 500 on the first question because the server is running the `runtime`
# image, which carries no FlagEmbedding, against a config asking for bge-m3.

def _required_modules(c: Config) -> list[dict]:
    """The third-party modules this configuration will import on first use."""
    out: list[dict] = []
    emb = c.embedding.backend.lower()
    if emb in ("bge-m3", "bgem3"):
        out.append({"component": "embedding", "backend": c.embedding.backend,
                    "module": "FlagEmbedding"})
    elif emb in ("sentence-transformers", "st"):
        out.append({"component": "embedding", "backend": c.embedding.backend,
                    "module": "sentence_transformers"})
    rr = c.rerank.backend.lower()
    if rr in ("bge-reranker", "bge_reranker"):
        out.append({"component": "rerank", "backend": c.rerank.backend,
                    "module": "FlagEmbedding"})
    elif rr in ("qwen3-reranker", "cross-encoder"):
        out.append({"component": "rerank", "backend": c.rerank.backend,
                    "module": "sentence_transformers"})
    if c.llm.backend.lower() not in ("echo", "none", ""):
        out.append({"component": "llm", "backend": c.llm.backend, "module": "httpx"})
    return out


def _missing_modules(c: Config) -> list[dict]:
    """Which of them are absent — checked by spec, so nothing is imported.

    find_spec does not load the module, so this stays cheap enough to run on
    every /stats call and never drags a multi-gigabyte model onto the GPU just
    to answer "are you healthy".
    """
    missing = []
    for req in _required_modules(c):
        try:
            found = importlib.util.find_spec(req["module"]) is not None
        except (ImportError, ValueError):  # broken or namespace-shadowed package
            found = False
        if not found:
            missing.append(req)
    return missing


# The generator is a remote API (Gemini), so "configured" and "reachable" are
# yet another pair of facts that must not be conflated: a perfectly loaded
# embedder and no route to the API (or a rejected key) produce an abstention
# with a connection error buried in it, which reads like a corpus problem.
_LLM_PROBE_TTL_S = 20.0
_llm_probe: dict = {"at": 0.0, "value": None}


def _probe_llm(c: Config) -> dict:
    """Is the generator actually listening? Cached, and never fatal."""
    if c.llm.backend.lower() in ("echo", "none", ""):
        return {"required": False, "reachable": None, "error": "",
                "base_url": c.llm.base_url}

    now = time.monotonic()
    cached = _llm_probe["value"]
    if cached is not None and now - _llm_probe["at"] < _LLM_PROBE_TTL_S:
        return cached

    base = c.llm.base_url.rstrip("/")
    out = {"required": True, "reachable": False, "error": "", "base_url": base}
    try:
        import httpx

        # A short timeout on purpose: this runs on every /stats call, and a
        # health check that hangs for the generator's 180s request timeout is
        # worse than no health check at all.
        resp = httpx.get(f"{base}/models", timeout=1.5,
                         headers={"Authorization": f"Bearer {c.llm.api_key}"})
        out["reachable"] = resp.status_code < 500
        if not out["reachable"]:
            out["error"] = f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001 — any failure means "not usable"
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]

    _llm_probe["at"] = now
    _llm_probe["value"] = out
    return out


# Set by api/main.py when constructing the answering stack raises. A model that
# fails to load once will fail the same way on the next request, and the console
# should say so before the operator types a question.
_answering_error: str = ""


def record_answering_error(message: str) -> None:
    global _answering_error
    _answering_error = message


def clear_answering_error() -> None:
    global _answering_error
    _answering_error = ""


# ------------------------------------------------------------------ system --
@router.get("/stats")
def stats() -> dict:
    """Everything the Monitor view needs in one round trip."""
    c = cfg()
    missing = _missing_modules(c)
    llm = _probe_llm(c)
    runtime = {
        "llm": llm,
        "embedding_backend": c.embedding.backend,
        "embedding_model": c.embedding.model_path,
        "rerank_backend": c.rerank.backend,
        "llm_backend": c.llm.backend,
        "llm_model": c.llm.model,
        "llm_base_url": c.llm.base_url,
        "final_k": c.retrieval.final_k,
        "max_packet_tokens": c.retrieval.max_packet_tokens,
        "cross_language": c.retrieval.cross_language,
        "dsn": re.sub(r"//[^@/]*@", "//***@", c.db.dsn),
        # The interpreter is reported because it is the single most common cause
        # of a missing module here: the weights stack is the optional `models`
        # extra, and a server started from an environment installed without it
        # cannot load bge-m3.
        "python": sys.executable,
        # Which settings the environment overrode. Without it, a config file
        # saying one thing and a running system doing another is a silent
        # contradiction the operator has no way to see.
        "env_overrides": sorted(c.env_overrides),
        "missing_modules": missing,
        "answering_error": _answering_error,
        # Named, not inferred, so the UI never has to guess why a banner is up.
        "placeholder_embedding": c.embedding.backend == "hash",
        "placeholder_llm": c.llm.backend in ("echo", "none", ""),
        "access": security.describe(),
    }
    runtime["configured_for_production"] = not (
        runtime["placeholder_embedding"] or runtime["placeholder_llm"]
    )
    # "Production ready" now means configured for production, able to load, and
    # — for the one component that lives in another process — actually running.
    runtime["production_ready"] = (
        runtime["configured_for_production"]
        and not missing
        and not _answering_error
        and llm.get("reachable") is not False
    )

    db = Database(c)
    try:
        counts = db.stats()
    except Exception as exc:  # noqa: BLE001
        # A dead database is a normal state for this console to render, so the
        # shape of the payload does not change — only its contents.
        return {"database": {"reachable": False, "error": str(exc)},
                "runtime": runtime, "counts": {}, "languages": {},
                "coverage": {}, "unembedded_chunks": 0, "dating": {},
                "unresolved": [], "releases": [], "queries": {}}

    unresolved = _rows("SELECT dst_act_code, n FROM xref_unresolved LIMIT 12")
    releases = _rows(
        """SELECT id, source, fetched_at, imported_at, note, manifest_sha256
             FROM corpus_release ORDER BY imported_at DESC LIMIT 8"""
    )
    languages = _rows("SELECT lang, count(*) AS n FROM chunk GROUP BY 1 ORDER BY 1")
    coverage = _rows(
        """SELECT count(*) FILTER (WHERE valid_to = DATE '9999-12-31') AS current,
                  count(*) FILTER (WHERE valid_to <> DATE '9999-12-31') AS superseded,
                  count(*) FILTER (WHERE repealed) AS repealed
             FROM chunk"""
    )
    embedded = _rows(
        """SELECT count(*) AS n FROM chunk c
            WHERE NOT EXISTS (SELECT 1 FROM chunk_embedding e WHERE e.chunk_id = c.id)"""
    )
    # Are there real redaction dates in here at all? adilet snapshots are dated
    # per article from their amendment footnotes (ingest/dating.py), but a text
    # with no footnote and no known adoption date — or a corpus imported before
    # that existed — still sits on the parser's 1900-01-01 fallback, and every
    # point-in-time feature then silently answers "as it has always been".
    #
    # ``closed`` separately says whether any *earlier* redaction is stored. A
    # corpus of current snapshots only is dated but has no history: an as-of
    # question before an article's last amendment finds no text for it.
    dating = _rows(
        """SELECT count(*) AS total,
                  count(*) FILTER (WHERE valid_from = DATE '1900-01-01') AS undated,
                  count(DISTINCT valid_from) AS distinct_starts,
                  count(*) FILTER (WHERE valid_to <> DATE '9999-12-31') AS closed
             FROM node_text"""
    )
    queries = _rows(
        """SELECT count(*) AS total,
                  count(*) FILTER (WHERE asked_at > now() - interval '24 hours') AS last_24h,
                  count(*) FILTER (WHERE abstained) AS abstained,
                  coalesce(round(avg(latency_ms))::int, 0) AS avg_latency_ms
             FROM query_log"""
    )
    return _jsonable({
        "database": {"reachable": True},
        "runtime": runtime,
        "counts": counts,
        "languages": {r["lang"]: r["n"] for r in languages},
        "coverage": coverage[0] if coverage else {},
        "unembedded_chunks": embedded[0]["n"] if embedded else 0,
        "dating": dating[0] if dating else {},
        "unresolved": unresolved,
        "releases": releases,
        "queries": queries[0] if queries else {},
    })


@router.get("/releases")
def releases() -> list[dict]:
    return _jsonable(_rows(
        """SELECT r.id, r.source, r.fetched_at, r.imported_at, r.note,
                  r.manifest_sha256,
                  (SELECT count(*) FROM chunk c WHERE c.release_id = r.id) AS chunks
             FROM corpus_release r ORDER BY r.imported_at DESC"""
    ))


# ------------------------------------------------------------------- acts --
@router.get("/acts")
def acts() -> list[dict]:
    """The corpus, act by act, with enough numbers to spot a bad import."""
    return _jsonable(_rows(
        """
        SELECT a.code, a.kind, a.norm_rank, a.official_id, a.title_ru, a.title_kk,
               a.adopted_on, a.url,
               (SELECT count(*) FROM node n
                 WHERE n.act_code = a.code AND n.kind = 'article')      AS articles,
               (SELECT count(*) FROM chunk c WHERE c.act_code = a.code) AS chunks,
               (SELECT count(*) FROM chunk c
                 WHERE c.act_code = a.code AND c.repealed)              AS repealed,
               (SELECT array_agg(DISTINCT c.lang ORDER BY c.lang) FROM chunk c
                 WHERE c.act_code = a.code)                             AS langs,
               (SELECT min(t.valid_from) FROM node_text t
                  JOIN node n2 ON n2.id = t.node_id
                 WHERE n2.act_code = a.code)                            AS earliest,
               (SELECT max(t.valid_from) FROM node_text t
                  JOIN node n2 ON n2.id = t.node_id
                 WHERE n2.act_code = a.code)                            AS latest
          FROM act a
         ORDER BY a.norm_rank, a.code
        """
    ))


@router.get("/acts/{act_code}/toc")
def toc(act_code: str,
        lang: Literal["ru", "kk"] = "ru",
        as_of: Optional[date] = None) -> dict:
    """The act's structure as it stood on a date — headings only, no bodies.

    Chapter headings come from the same versioned table as article text, so a
    table of contents is also point-in-time: an article added in 2023 does not
    appear in the 2019 view.
    """
    when = as_of or date.today()
    rows = _rows(
        """
        SELECT n.id, n.kind, n.number, n.parent_id, n.ordinal,
               t.heading, t.valid_from, t.valid_to,
               length(t.body) AS body_chars
          FROM node n
          LEFT JOIN node_text t
            ON t.node_id = n.id AND t.lang = %s
           AND t.valid_from <= %s AND t.valid_to >= %s
         WHERE n.act_code = %s
           AND n.kind IN ('act','section','chapter','paragraph','article')
         ORDER BY n.ordinal, n.id
        """,
        (lang, when, when, act_code),
    )
    if not rows:
        raise HTTPException(404, f"no act {act_code!r} in the corpus")

    by_id = {r["id"]: {**r, "children": []} for r in rows}
    roots: list[dict] = []
    for row in by_id.values():
        parent = by_id.get(row["parent_id"]) if row["parent_id"] else None
        (parent["children"] if parent else roots).append(row)

    meta = _rows(
        "SELECT code, kind, title_ru, title_kk, adopted_on, url FROM act WHERE code = %s",
        (act_code,),
    )
    return _jsonable({
        "act": meta[0] if meta else {"code": act_code},
        "as_of": when.isoformat(),
        "lang": lang,
        "tree": roots,
    })


@router.get("/nodes/{node_id:path}")
def node_text(node_id: str, as_of: Optional[date] = None) -> dict:
    """Every redaction of one node, with the one in force on ``as_of`` marked.

    The history is returned in full rather than filtered to the requested date:
    "this article was amended twice since you last read it" is the single most
    useful thing a legal UI can tell someone, and it is invisible if the server
    only ever returns the current text.
    """
    when = as_of or date.today()
    rows = _rows(
        """SELECT lang, heading, body, valid_from, valid_to, release_id
             FROM node_text WHERE node_id = %s
            ORDER BY valid_from DESC, lang""",
        (node_id,),
    )
    if not rows:
        raise HTTPException(404, f"no text on file for {node_id}")
    for row in rows:
        row["in_force_on_as_of"] = row["valid_from"] <= when <= row["valid_to"]
        row["is_current"] = row["valid_to"] == date(9999, 12, 31)

    refs = _rows(
        """SELECT DISTINCT raw, dst_act_code, dst_article, dst_node_id, kind
             FROM xref WHERE src_node_id = %s ORDER BY dst_node_id NULLS LAST LIMIT 40""",
        (node_id,),
    )
    citing = _rows(
        """SELECT DISTINCT src_node_id FROM xref
            WHERE dst_node_id = %s ORDER BY src_node_id LIMIT 40""",
        (node_id,),
    )
    return _jsonable({
        "node_id": node_id,
        "as_of": when.isoformat(),
        "texts": rows,
        "references": refs,
        "cited_by": [r["src_node_id"] for r in citing],
    })


# ------------------------------------------------------------- audit trail --
@router.get("/queries")
def queries(limit: int = Query(50, ge=1, le=500),
            offset: int = Query(0, ge=0),
            only: Literal["all", "abstained", "unverified", "slow"] = "all",
            q: str = "") -> dict:
    where = ["TRUE"]
    params: list[Any] = []
    if only == "abstained":
        where.append("abstained")
    elif only == "unverified":
        where.append("(verdict = 'no_citations' OR verdict LIKE 'not_verified%%'"
                     " OR verdict ~ '^0/')")
    elif only == "slow":
        where.append("latency_ms > 5000")
    if q:
        where.append("question ILIKE %s")
        params.append(f"%{q}%")
    clause = " AND ".join(where)

    total = _rows(f"SELECT count(*) AS n FROM query_log WHERE {clause}", tuple(params))
    rows = _rows(
        f"""SELECT id, asked_at, question, lang, as_of, release_id, verdict,
                   abstained, latency_ms,
                   jsonb_array_length(retrieved) AS n_retrieved,
                   jsonb_array_length(cited)     AS n_cited,
                   left(answer, 400) AS answer_preview
              FROM query_log WHERE {clause}
             ORDER BY asked_at DESC LIMIT %s OFFSET %s""",
        tuple(params) + (limit, offset),
    )
    return _jsonable({"total": total[0]["n"] if total else 0, "rows": rows})


@router.get("/queries/{query_id}")
def query_detail(query_id: int) -> dict:
    rows = _rows(
        """SELECT id, asked_at, question, lang, as_of, release_id, retrieved,
                  cited, verdict, abstained, answer, latency_ms
             FROM query_log WHERE id = %s""",
        (query_id,),
    )
    if not rows:
        raise HTTPException(404, f"no query {query_id}")
    return _jsonable(rows[0])


@router.get("/metrics")
def metrics(days: int = Query(14, ge=1, le=365)) -> dict:
    daily = _rows(
        """
        SELECT date_trunc('day', asked_at)::date AS day,
               count(*)                                        AS n,
               count(*) FILTER (WHERE abstained)                AS abstained,
               count(*) FILTER (WHERE verdict = 'no_citations'
                                   OR verdict LIKE 'not_verified%%'
                                   OR verdict ~ '^0/')          AS unverified,
               coalesce(percentile_disc(0.5)  WITHIN GROUP (ORDER BY latency_ms), 0) AS p50,
               coalesce(percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS p95
          FROM query_log
         WHERE asked_at >= now() - make_interval(days => %s)
         GROUP BY 1 ORDER BY 1
        """,
        (days,),
    )
    verdicts = _rows(
        """SELECT CASE
                    WHEN abstained THEN 'abstained'
                    WHEN verdict = 'no_citations' THEN 'no citations'
                    WHEN verdict ~ '^0/' THEN 'verification failed'
                    WHEN verdict ~ '^([0-9]+)/\\1 ' THEN 'fully verified'
                    ELSE 'partly verified'
                  END AS bucket, count(*) AS n
             FROM query_log
            WHERE asked_at >= now() - make_interval(days => %s)
            GROUP BY 1 ORDER BY 2 DESC""",
        (days,),
    )
    slowest = _rows(
        """SELECT id, question, latency_ms, asked_at FROM query_log
            WHERE asked_at >= now() - make_interval(days => %s)
            ORDER BY latency_ms DESC LIMIT 8""",
        (days,),
    )
    top_articles = _rows(
        """SELECT c ->> 'act_code' AS act_code, c ->> 'article' AS article,
                  count(*) AS n
             FROM query_log q, jsonb_array_elements(q.cited) AS c
            WHERE q.asked_at >= now() - make_interval(days => %s)
              AND c ->> 'article' IS NOT NULL
            GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 12""",
        (days,),
    )
    return _jsonable({"days": days, "daily": daily, "verdicts": verdicts,
                      "slowest": slowest, "top_articles": top_articles})


# ----------------------------------------------------------------- ingest --
@dataclass
class Job:
    id: str
    path: str
    started_at: str
    files: list[str]
    options: dict
    status: str = "running"          # running | done | failed | interrupted
    log: list[str] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    error: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "path": self.path, "started_at": self.started_at,
            "finished_at": self.finished_at, "files": self.files,
            "options": self.options, "status": self.status, "log": self.log,
            "totals": self.totals, "error": self.error,
        }


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_jobs_loaded = False
_JOB_FILE = "job.json"


def _uploads_dir() -> Path:
    return Path(cfg().data_dir) / "uploads"


def _save_job(job: Job) -> None:
    """Persist the job record next to its staged files.

    Kept on disk rather than in the database on purpose: an import with
    ``reset`` drops the schema, and a job table would take the record of the
    very import that dropped it along with everything else.
    """
    import json

    try:
        path = Path(job.path) / _JOB_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        _replace_with_retry(tmp, path)
    except Exception as exc:  # noqa: BLE001 — history must never break an import
        log.warning("could not save ingest job %s: %s", job.id, exc)


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 20) -> None:
    """Atomic rename that tolerates a concurrent reader on Windows.

    On POSIX a rename over an open file just works. On Windows it fails with
    "Access is denied" while any handle to the target is open — which is
    exactly when the console is polling the record. Left unretried, the
    finished job stayed marked "running" on disk and came back as
    "interrupted" after the next restart.
    """
    for i in range(attempts):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.02 * (i + 1))


def _load_jobs() -> None:
    """Read job history from data/uploads once per process.

    A job still marked running belonged to a process that is gone, so it is
    reported as interrupted — never as running, which would claim an import
    lock nobody holds.
    """
    import json

    global _jobs_loaded
    with _jobs_lock:
        if _jobs_loaded:
            return
        _jobs_loaded = True
        root = _uploads_dir()
        if not root.is_dir():
            return
        for path in root.glob(f"*/{_JOB_FILE}"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job(**{k: data[k] for k in Job.__dataclass_fields__ if k in data})
            except Exception as exc:  # noqa: BLE001
                log.warning("unreadable ingest job record %s: %s", path, exc)
                continue
            if job.id in _jobs:
                continue
            if job.status == "running":
                job.status = "interrupted"
                job.error = job.error or "the server stopped before this import finished"
                job.log.append(job.error)
            _jobs[job.id] = job
# One import at a time. ingest_directory() can reset the schema, and two of
# those interleaving would corrupt the corpus in a way that is very hard to
# diagnose afterwards.
_ingest_lock = threading.Lock()

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class UploadFile(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    content: str


class IngestRequest(BaseModel):
    files: list[UploadFile] = Field(min_length=1)
    reset: bool = False
    verify: bool = True
    build_index: bool = True


@router.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    """Stage uploaded sources on disk, then import them in a worker thread.

    Text-only and JSON-bodied on purpose: ``*.act.txt``, ``documents.json`` and
    adilet HTML are all text, and accepting them this way avoids adding
    python-multipart to a wheel bundle that has to be carried by hand.
    """
    names = [f.name for f in req.files]
    for name in names:
        # "." and ".." pass the character check; job.json is the job record.
        if not _SAFE_NAME.match(name) or name.startswith(".") or name == _JOB_FILE:
            raise HTTPException(400, f"unsafe filename: {name!r}")
    if len(set(names)) != len(names):
        raise HTTPException(400, "the same filename was uploaded twice")

    # Taken here and released by the worker. Checking .locked() and acquiring in
    # the thread let two requests arriving together both pass the check; the
    # second then queued behind the first instead of getting its 409 — with
    # reset=True that drops the corpus the first one just imported.
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "an import is already running")
    try:
        return _start_ingest(req)
    except BaseException:
        _ingest_lock.release()
        raise


def _start_ingest(req: IngestRequest) -> dict:

    job_id = uuid.uuid4().hex[:12]
    staged = Path(cfg().data_dir) / "uploads" / job_id
    staged.mkdir(parents=True, exist_ok=True)
    for f in req.files:
        (staged / f.name).write_text(f.content, encoding="utf-8")

    job = Job(id=job_id, path=str(staged),
              started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
              files=[f.name for f in req.files],
              options={"reset": req.reset, "verify": req.verify,
                       "build_index": req.build_index})
    job.log.append(f"staged {len(req.files)} file(s) in {staged}")
    _load_jobs()
    with _jobs_lock:
        _jobs[job_id] = job
    _save_job(job)

    threading.Thread(target=_run_ingest, args=(job, req), daemon=True,
                     name=f"ingest-{job_id}").start()
    return job.to_dict()


class _JobLogHandler(logging.Handler):
    """Tee the ingest pipeline's own log lines into the job record.

    The pipeline already logs the things an operator wants to watch — acts
    parsed, chunks written, the vector index build. Re-reporting progress from
    here would only invent a second, less accurate account of the same run.
    """

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.INFO)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.append(f"{record.levelname.lower()}: {record.getMessage()}")
        except Exception:  # noqa: BLE001 — logging must never break the import
            pass
        del self.sink[:-400]


@contextmanager
def _capture_logs(sink: list[str]):
    handler = _JobLogHandler(sink)
    root = logging.getLogger("lawrag")
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)


def _run_ingest(job: Job, req: IngestRequest) -> None:
    """Runs with ``_ingest_lock`` already held by the request; releases it."""
    try:
        started = time.perf_counter()
        try:
            from ..ingest.bundle import BundleVerificationError
            from ..ingest.pipeline import ingest_directory

            job.log.append("parsing and indexing…")
            with _capture_logs(job.log):
                totals = ingest_directory(
                    cfg(), Path(job.path), verify=req.verify,
                    reset=req.reset, build_index=req.build_index,
                )
            job.totals = totals
            job.status = "done"
            job.log.append(f"done in {int(time.perf_counter() - started)}s: {totals}")
        except BundleVerificationError as exc:
            job.status = "failed"
            job.error = f"bundle rejected — nothing was imported\n{exc}"
            job.log.append(job.error)
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append(job.error)
            log.exception("ingest job %s failed", job.id)
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _save_job(job)
    finally:
        _ingest_lock.release()


@router.get("/ingest/jobs")
def ingest_jobs() -> list[dict]:
    _load_jobs()
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)
    return [{k: v for k, v in j.to_dict().items() if k != "log"} for j in jobs]


@router.get("/ingest/jobs/{job_id}")
def ingest_job(job_id: str) -> dict:
    _load_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job.to_dict()
