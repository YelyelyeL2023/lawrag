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
    refs = []
    terms = []
    for node in root.walk():
        text = node.texts.get(lang)
        if text is None or not text.body:
            continue
        if node.kind in (NodeKind.ARTICLE, NodeKind.POINT, NodeKind.NOTE):
            refs.extend(R.extract_refs(node.id, act_code, text.body, lang))
        if node.kind is NodeKind.ARTICLE and R.is_definition_article(text.heading):
            for term, definition in R.extract_definitions(node.id, act_code, text.body):
                terms.append((term, definition, act_code, node.id,
                              text.valid_from, text.valid_to))
    return refs, terms


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

    refs, terms = _collect_refs_and_terms(root, doc.lang, doc.act_code)
    indexer.index_xrefs(refs, doc.lang, release_id, doc.valid_from, doc.valid_to)
    indexer.index_terms(terms, doc.lang, release_id)

    # Anything this act still had open before the snapshot's effective date, and
    # that the snapshot no longer contains, was repealed. Nothing in the
    # per-node path can see it, so it is closed here or never.
    indexer.close_dropped(doc.act_code, doc.lang, doc.valid_from)

    totals["acts"] += 1
    totals["chunks"] += len(chunks)
    totals["refs"] += len(refs)
    totals["terms"] += len(terms)
    log.info("indexed %s [%s]: %d chunks, %d refs, %d terms",
             doc.act_code, doc.lang.value, len(chunks), len(refs), len(terms))
    return totals
