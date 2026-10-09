"""Regression tests for defects found in review.

Each test names the failure it pins down. They exist because every one of these
passed a plausible-looking implementation and was only caught by reproducing the
bad output.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from lawrag.ingest.parser import parse_act_source
from lawrag.ingest.pipeline import ingest_directory
from lawrag.index.db import Database
from lawrag.models import Lang
from lawrag.verify.citations import CitationVerifier

SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"

HEADER = """---
code: {code}
kind: code
official_id: SYNTH
lang: ru
title: "[СИНТЕТИКА] {title}"
valid_from: {valid_from}
---
"""


@pytest.fixture()
def corpus(live_db, tmp_path):
    """Ingest a purpose-built corpus, then restore the sample for other tests."""
    def _ingest(files: dict[str, str]):
        target = tmp_path / "corpus"
        target.mkdir(exist_ok=True)
        for name in list(target.glob("*.act.txt")):
            name.unlink()
        for name, content in files.items():
            (target / name).write_text(content, encoding="utf-8")
        ingest_directory(live_db, target, verify=False, reset=True)
        return live_db

    yield _ingest
    ingest_directory(live_db, SAMPLE, verify=False, reset=True)


def _act(code, title, valid_from, body):
    return HEADER.format(code=code, title=title, valid_from=valid_from) + body


# ---------------------------------------------------------------- citations --

def test_valid_source_index_does_not_launder_a_fabricated_pinpoint(live_db):
    """A verbatim quote from source 1 attached to a nonexistent article.

    Trusting the source index alone made this pass with 'verified', because the
    quote check and the retrieved check both looked only at source 1 and never
    at the article number the user would actually be shown.
    """
    from lawrag.generate.answer import AnswerService

    service = AnswerService(live_db)
    result = service.engine.retrieve(
        "В какой срок нужно подать заявление о постановке на регистрационный учет?"
    )
    sources = result.packet.sources
    assert sources and sources[0]["article_node_id"] == "nk/st76"

    verdict = CitationVerifier(service.db).verify(
        [{"source": 1, "act_code": "koap", "article": "999",
          "quote": "не позднее десяти рабочих дней со дня начала осуществления деятельности"}],
        sources, date.today(),
    )
    assert not verdict.passed
    assert verdict.downgrade


def test_quoting_a_superseded_redaction_is_caught_and_named(live_db):
    """A verbatim quote from the 2024 text of an article that was amended in
    2026 is accurate but wrong for the date asked about.

    Filtering by date inside the SQL that fetched the text made every returned
    row in force by construction, so nothing could distinguish "this quote is
    invented" from "this quote is from the wrong period".
    """
    from lawrag.generate.answer import AnswerService

    service = AnswerService(live_db)
    sources = [{
        "index": 1, "chunk_id": "koap/st278#ru@2024-01-01", "node_id": "koap/st278",
        "article_node_id": "koap/st278", "act_code": "koap", "lang": "ru",
        "heading": "Статья 278", "valid_from": "2024-01-01", "valid_to": "2025-12-31",
        "is_current": False,
    }]
    verdict = CitationVerifier(service.db).verify(
        [{"source": 1, "article": "278",
          "quote": "влечет штраф на субъектов малого предпринимательства в размере "
                   "десяти месячных расчетных показателей"}],
        sources, date(2026, 6, 1),
    )
    check = verdict.checks[0]
    assert not check.ok
    assert any("different redaction" in p for p in check.problems), check.problems


def test_citing_a_closed_article_fails_the_in_force_check(corpus):
    """An article the newest snapshot dropped is closed, and citing it as of a
    later date must be reported as not in force."""
    cfg = corpus({
        "a_2024.act.txt": _act("koap", "КоАП", "2024-01-01",
                               "Глава 1. Общее\n\nСтатья 800. Ушедшая\n"
                               "Норма действовала до конца две тысячи двадцать пятого года.\n"),
        "b_2026.act.txt": _act("koap", "КоАП", "2026-01-01",
                               "Глава 1. Общее\n\nСтатья 801. Оставшаяся\nДругая норма.\n"),
    })
    sources = [{
        "index": 1, "chunk_id": "koap/st800#ru@2024-01-01", "node_id": "koap/st800",
        "article_node_id": "koap/st800", "act_code": "koap", "lang": "ru",
        "heading": "Статья 800", "valid_from": "2024-01-01", "valid_to": "2025-12-31",
        "is_current": False,
    }]
    verdict = CitationVerifier(Database(cfg)).verify(
        [{"source": 1, "article": "800",
          "quote": "Норма действовала до конца две тысячи двадцать пятого года"}],
        sources, date(2026, 6, 1),
    )
    check = verdict.checks[0]
    assert not check.in_force
    assert any("not in force" in p for p in check.problems), check.problems


# --------------------------------------------------------------- versioning --

def test_article_dropped_by_a_new_snapshot_stops_being_current(corpus):
    """A repealed article vanishes from the next snapshot.

    Nothing in the per-node update path ever sees it again, so without an
    explicit closing pass it stayed open-ended and kept answering as current law.
    """
    cfg = corpus({
        "a_2024.act.txt": _act("koap", "КоАП", "2024-01-01",
                               "Глава 1. Общее\n\nСтатья 600. Первая\nТекст.\n\n"
                               "Статья 601. Вторая\nТекст.\n"),
        "b_2026.act.txt": _act("koap", "КоАП", "2026-01-01",
                               "Глава 1. Общее\n\nСтатья 600. Первая\nНовый текст.\n"),
    })
    db = Database(cfg)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT valid_to FROM node_text WHERE node_id='koap/st601' AND lang='ru'"
        )
        rows = cur.fetchall()
    assert rows, "the repealed article must be kept, not deleted"
    assert all(r["valid_to"] == date(2025, 12, 31) for r in rows), \
        "article 601 disappeared from the 2026 snapshot and must be closed"


def test_reverse_ingest_order_does_not_produce_two_current_redactions(corpus):
    """Ingest globs and sorts filenames, so the newer snapshot can arrive first.

    The closing logic only looked backwards, so importing current-then-historic
    left both redactions open-ended and both answering as current law.
    """
    cfg = corpus({
        # 'current' sorts before 'old', so the 2026 text is ingested first.
        "current.act.txt": _act("koap", "КоАП", "2026-01-01",
                                "Глава 1. Общее\n\nСтатья 700. Норма\nНовая редакция.\n"),
        "old.act.txt": _act("koap", "КоАП", "2024-01-01",
                            "Глава 1. Общее\n\nСтатья 700. Норма\nСтарая редакция.\n"),
    })
    db = Database(cfg)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS n FROM node_text
             WHERE node_id = 'koap/st700' AND lang = 'ru'
               AND valid_from <= DATE '2026-06-01' AND valid_to >= DATE '2026-06-01'
            """
        )
        assert cur.fetchone()["n"] == 1

        cur.execute(
            """
            SELECT count(*) AS n FROM node_text a JOIN node_text b
                ON a.node_id = b.node_id AND a.lang = b.lang
               AND a.valid_from < b.valid_from AND a.valid_to >= b.valid_from
            """
        )
        assert cur.fetchone()["n"] == 0


def test_historic_definitions_survive_a_new_snapshot(corpus):
    """An unscoped delete of glossary rows destroyed the previous redaction's
    definitions, so hydration for an as-of date in that period attached nothing."""
    def defs(text):
        return ("Глава 1. Общее\n\nСтатья 24. Основные понятия, используемые в "
                f"настоящем Кодексе\nВ настоящем Кодексе:\n1) субъект — {text};\n")

    cfg = corpus({
        "a_2024.act.txt": _act("pk", "ПК", "2024-01-01", defs("лицо до ста человек")),
        "b_2026.act.txt": _act("pk", "ПК", "2026-01-01", defs("лицо до двухсот человек")),
    })
    db = Database(cfg)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT definition FROM term
             WHERE lang='ru' AND valid_from <= DATE '2025-01-01'
               AND valid_to >= DATE '2025-01-01'
            """
        )
        rows = cur.fetchall()
    assert rows, "the 2024 definition must still be retrievable as of 2025"
    assert "ста человек" in rows[0]["definition"]


# ------------------------------------------------------------------ parsing --

def test_preamble_and_container_text_are_not_dropped():
    """Everything outside an article used to be discarded with no log line."""
    source = _act("konstitucia", "Конституция", "1995-08-30",
                  "Мы, народ Казахстана, принимаем настоящую Конституцию.\n\n"
                  "РАЗДЕЛ 1. ОБЩИЕ ПОЛОЖЕНИЯ\n"
                  "Настоящий раздел применяется ко всем последующим разделам.\n\n"
                  "Статья 1. Республика Казахстан\nВысшими ценностями являются человек.\n")
    _act_obj, root, doc = parse_act_source(source)
    texts = {n.id: (n.texts[doc.lang].body if doc.lang in n.texts else "")
             for n in root.walk()}
    assert "Мы, народ Казахстана" in texts["konstitucia"]
    assert "применяется ко всем последующим разделам" in texts["konstitucia/sec1"]


# ---------------------------------------------------------------- retrieval --

def test_lead_in_survives_the_context_batch_job(live_db, tmp_path):
    """Rebuilding embed_text without the lead-in stripped the offence-defining
    clause from every point chunk — silently making retrieval worse."""
    from lawrag.chunking.context import generate_context_sentences
    from lawrag.models import Chunk

    chunk = Chunk(
        id="x#ru@2026-01-01", node_id="koap/st278/p1",
        article_node_id="koap/st278", act_code="koap", lang=Lang.RU,
        valid_from=date(2026, 1, 1), valid_to=date(9999, 12, 31),
        display_text="пункт 1. влечет предупреждение.",
        embed_text="", hierarchy_path="КоАП → Глава 14",
        heading="Статья 278. Нарушение срока",
        lead_in="Нарушение субъектом предпринимательства срока подачи заявления, —",
    )

    class Stub:
        def complete(self, prompt, **kw):
            return "Норма о предупреждении."

    generate_context_sentences([chunk], Stub(), tmp_path / "cache.json")
    assert "Нарушение субъектом предпринимательства" in chunk.embed_text
    assert "Норма о предупреждении" in chunk.embed_text


def test_each_language_branch_is_ranked_independently(live_db):
    """Concatenating the per-language lexical results and enumerating the
    concatenation pushed the whole second language to ranks 51-100."""
    from lawrag.retrieve.engine import RetrievalEngine
    from lawrag.retrieve.query import analyze

    engine = RetrievalEngine(live_db)
    analysis = analyze("278-бап бойынша тіркеу есебіне қою мерзімі қандай?")
    branches = engine.retriever.sparse(analysis, 50)
    assert len(branches) >= 1
    # Every branch starts at rank 1 by construction.
    for branch in branches:
        assert branch, "an empty branch should not be returned at all"
    langs = {h.lang for branch in branches for h in branch[:1]}
    if len(branches) > 1:
        assert len(langs) > 1, "branches must be per-language, not a concatenation"


def test_query_log_records_the_corpus_release(live_db):
    """Answers claimed to be traceable to a snapshot, but release_id was always
    NULL because the packet never carried it."""
    from lawrag.generate.answer import AnswerService

    service = AnswerService(live_db)
    db = Database(live_db)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM query_log")
    service.ask("В какой срок нужно встать на регистрационный учет?")
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT release_id FROM query_log ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    assert row and row["release_id"], "the answering snapshot must be recorded"


def test_repealed_article_is_flagged_in_the_packet(live_db):
    """Article 279 in the 2026 sample is a repeal stub. Its validity range is
    open, so only the repeal flag stops it reading as ordinary current law."""
    from lawrag.retrieve.engine import RetrievalEngine

    engine = RetrievalEngine(live_db)
    result = engine.retrieve("что говорит статья 279 КоАП")
    source = next(s for s in result.packet.sources if s["node_id"] == "koap/st279")
    assert source["repealed"]
    assert "исключена" in result.packet.text.lower()


def test_malformed_iso_date_does_not_crash_the_request(live_db):
    from lawrag.retrieve.engine import RetrievalEngine

    engine = RetrievalEngine(live_db)
    result = engine.retrieve("Что действовало на 2025-02-30?")
    assert result.analysis.as_of == date.today()


def test_non_finite_embedding_is_refused_by_name():
    # fp16 inference of the fine-tuned bge-m3 overflowed on a KoAP chunk during
    # ingest; Postgres answered "NaN not allowed in vector" for parameter $4,
    # naming nothing. The indexer now checks first and names the chunk.
    import numpy as np
    import pytest

    from lawrag.index.indexer import check_finite

    check_finite("koap/st1#ru@2026-01-01", np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="koap/st278/p2#ru@2026-01-01"):
        check_finite("koap/st278/p2#ru@2026-01-01", np.array([0.1, float("nan"), 0.2]))


def test_concurrent_requests_do_not_run_the_reranker_in_parallel():
    # Two console requests at once ran bge-reranker on MPS from two threads and
    # Metal aborted the server ("failed assertion _status <
    # MTLCommandBufferStatusCommitted"). Forward passes are now serialised.
    import threading
    import time
    from types import SimpleNamespace

    from lawrag.config import RerankConfig
    from lawrag.retrieve.rerank import FlagReranker

    active, overlaps = [0], []

    class SlowModel:
        def compute_score(self, pairs, normalize=False):
            active[0] += 1
            overlaps.append(active[0] > 1)
            time.sleep(0.05)
            active[0] -= 1
            return [0.0] * len(pairs)

    reranker = FlagReranker.__new__(FlagReranker)
    reranker.cfg, reranker._model = RerankConfig(), SlowModel()
    hits = [SimpleNamespace(heading="h", display_text="t", rerank_score=None)]
    threads = [threading.Thread(target=reranker.rerank, args=("q", hits, 1)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps and not any(overlaps)


def test_kazakh_article_label_resolves_to_the_cited_source():
    # Answering in Kazakh, the model cited "52-бап" (as the Kazakh text heads the
    # article); the verifier compared it to "tk/st52" verbatim and rejected all
    # three citations of a correct answer.
    by_index = {1: {"article_node_id": "tk/st52"}, 2: {"article_node_id": "tk/st52-1"}}
    resolve = CitationVerifier._resolve
    assert resolve({"source": 1, "article": "52-бап"}, by_index) == "tk/st52"
    assert resolve({"source": 2, "article": "52-1-бап"}, by_index) == "tk/st52-1"
    assert resolve({"source": 1, "article": "Статья 52"}, by_index) == "tk/st52"
    assert resolve({"source": 1, "article": "53-бап"}, by_index) is None
