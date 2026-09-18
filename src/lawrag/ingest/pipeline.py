"""End-to-end ingest: bundle -> parse -> chunk -> extract -> index."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..chunking.chunker import Chunker
from ..config import Config
from ..index.db import Database
from ..index.embed import build_embedder
from ..index.indexer import Indexer
from ..logging_setup import get_logger
from ..models import CorpusRelease, Lang, Node, NodeKind
from . import refs as R
from .bundle import build_manifest, verify_bundle
from .parser import parse_act_source, parse_adilet

log = get_logger(__name__)


def _release_from_dir(path: Path, verify: bool) -> CorpusRelease:
    if verify and (path / "manifest.json").exists():
        manifest = verify_bundle(path)
    else:
        if verify:
            log.warning("no manifest in %s — computing one locally. A real snapshot "
                        "must arrive with a manifest produced on the connected side.", path)
        manifest = build_manifest(path, source=str(path), note="manifest computed at import")
    return CorpusRelease(
        id=manifest.release_id,
        source=manifest.source,
        fetched_at=datetime.now(timezone.utc),
        manifest_sha256=manifest.digest,
        note=manifest.note,
    )


def _collect_refs_and_terms(root: Node, lang: Lang, act_code: str):
    """References grouped by the validity of the text they were read from.

    A reference is only as current as the sentence it sits in: once articles
    carry their own start dates, stamping every edge with the snapshot date
    would make the citation graph invisible to any as-of query before the day
    the snapshot was taken.
    """
    refs: dict[tuple, list] = {}
    terms = []
    for node in root.walk():
        text = node.texts.get(lang)
        if text is None or not text.body:
            continue
        if node.kind in (NodeKind.ARTICLE, NodeKind.POINT, NodeKind.NOTE):
            found = R.extract_refs(node.id, act_code, text.body, lang)
            if found:
                refs.setdefault((text.valid_from, text.valid_to), []).extend(found)
        if node.kind is NodeKind.ARTICLE and R.is_definition_article(text.heading):
            for term, definition in R.extract_definitions(node.id, act_code, text.body):
                terms.append((term, definition, act_code, node.id,
                              text.valid_from, text.valid_to))
    return refs, terms


def _refuse_mixed_editions(sources: list[Path], html_docs: list[tuple[str, dict]]) -> None:
    """Refuse a directory holding two different documents under one act code.

    Redactions of one act are expected to share an official id. Two sources for
    the same act code and language with *different* ids are two different
    documents — the synthetic test fixture sitting next to a real adilet
    snapshot, say — and ingesting both interleaves their articles by number:
    the fixture's "статья 278" closes the real one and answers as current law.
    That happened to this repository's own data/sample directory.
    """
    import yaml

    seen: dict[tuple[str, str], dict[str, str]] = {}
    for source in sources:
        raw = source.read_text(encoding="utf-8").lstrip()
        if not raw.startswith("---"):
            continue
        meta = yaml.safe_load(raw.split("---", 2)[1]) or {}
        key = (str(meta.get("code")), str(meta.get("lang") or ""))
        seen.setdefault(key, {})[str(meta.get("official_id", ""))] = source.name
    for _, meta in html_docs:
        key = (str(meta.get("code")), str(meta.get("lang") or ""))
        seen.setdefault(key, {})[str(meta.get("official_id", ""))] = meta["file"]
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    if clashes:
        detail = "; ".join(
            f"{code} [{lang or '?'}]: " + ", ".join(f"{oid or '<no id>'} ({name})"
                                                  for oid, name in ids.items())
            for (code, lang), ids in sorted(clashes.items())
        )
        raise ValueError(
            "refusing to ingest: these act codes come from more than one document "
            f"(different official ids): {detail}. Put each corpus in its own directory."
        )


def ingest_directory(cfg: Config, path: Path, *, verify: bool = True,
                     reset: bool = False, build_index: bool = True) -> dict:
    """Ingest every act source in ``path`` into the database."""
    db = Database(cfg)
    if reset:
        db.reset()
    db.apply_schema()

    embedder = build_embedder(cfg.embedding)
    indexer = Indexer(db, embedder)
    release = _release_from_dir(path, verify)
    indexer.register_release(release)
    log.info("corpus release %s", release.id)

    chunker = Chunker(cfg.chunking)
    totals = {"acts": 0, "chunks": 0, "refs": 0, "terms": 0, "release": release.id}

    sources = sorted(path.glob("*.act.txt"))
    html_meta = path / "documents.json"
    html_docs: list[tuple[str, dict]] = []
    if html_meta.exists():
        import json
        for entry in json.loads(html_meta.read_text(encoding="utf-8")):
            html_path = path / entry["file"]
            if html_path.exists():
                html_docs.append((html_path.read_text(encoding="utf-8"), entry))

    if not sources and not html_docs:
        raise FileNotFoundError(f"no *.act.txt or adilet documents.json found in {path}")
    _refuse_mixed_editions(sources, html_docs)

    for source in sources:
        act, root, doc = parse_act_source(source.read_text(encoding="utf-8"))
        totals = _index_one(indexer, chunker, act, root, doc, release.id, totals)

    for html, meta in html_docs:
        act, root, doc = parse_adilet(html, meta)
        totals = _index_one(indexer, chunker, act, root, doc, release.id, totals)

    indexer.link_aligned_chunks()
    if build_index:
        indexer.build_vector_index()

    log.info("ingest complete: %s", totals)
    log.info("chunker: %s", chunker.stats)
    return totals


def _index_one(indexer: Indexer, chunker: Chunker, act, root, doc, release_id: str,
               totals: dict) -> dict:
    indexer.upsert_act(act)
    indexer.upsert_tree(root, doc.lang, release_id)

    chunks = chunker.chunk_act(root, doc.lang, release_id)
    indexer.index_chunks(chunks, release_id)

    grouped_refs, terms = _collect_refs_and_terms(root, doc.lang, doc.act_code)
    n_refs = 0
    for (valid_from, valid_to), refs in grouped_refs.items():
        n_refs += indexer.index_xrefs(refs, doc.lang, release_id, valid_from, valid_to)
    indexer.index_terms(terms, doc.lang, release_id)

    # Anything this act still had open before the snapshot's effective date, and
    # that the snapshot no longer contains, was repealed. Nothing in the
    # per-node path can see it, so it is closed here or never. Nodes the
    # snapshot does contain are left alone: their older redactions were already
    # closed by the per-node path, and their current text may legitimately
    # start before the snapshot date.
    present = {node.id for node in root.walk() if doc.lang in node.texts}
    indexer.close_dropped(doc.act_code, doc.lang, doc.valid_from, keep=present)

    totals["acts"] += 1
    totals["chunks"] += len(chunks)
    totals["refs"] += n_refs
    totals["terms"] += len(terms)
    log.info("indexed %s [%s]: %d chunks, %d refs, %d terms",
             doc.act_code, doc.lang.value, len(chunks), n_refs, len(terms))
    return totals
