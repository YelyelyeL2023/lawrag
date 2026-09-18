"""Chunking: structural boundaries, and context that must not be lost."""

from __future__ import annotations

from pathlib import Path

import pytest

from lawrag.chunking.chunker import Chunker, article_lead_in
from lawrag.config import ChunkConfig
from lawrag.ingest.parser import parse_act_source
from lawrag.models import Lang

SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"


def _chunks(filename, **cfg_kwargs):
    act, root, doc = parse_act_source((SAMPLE / filename).read_text(encoding="utf-8"))
    chunker = Chunker(ChunkConfig(**cfg_kwargs))
    return chunker.chunk_act(root, doc.lang), root, doc


def test_short_article_stays_whole():
    chunks, _, _ = _chunks("nk.ru.act.txt")
    ids = {c.node_id for c in chunks}
    assert "nk/st76" in ids
    assert not any("/p" in i for i in ids)


def test_long_article_splits_on_points_not_tokens():
    chunks, _, _ = _chunks("koap.ru.act.txt", max_article_tokens=180)
    ids = [c.node_id for c in chunks]
    assert "koap/st278/p1" in ids
    assert "koap/st278/p2" in ids
    # Every boundary is a structural one.
    assert all("~" not in i for i in ids)


def test_point_chunks_carry_the_article_lead_in():
    """A point chunk without the article's opening disposition describes a
    penalty for unspecified conduct."""
    chunks, _, _ = _chunks("koap.ru.act.txt", max_article_tokens=180)
    p1 = next(c for c in chunks if c.node_id == "koap/st278/p1")
    assert "Нарушение субъектом предпринимательства срока" in p1.embed_text
    # ...but the quotable text stays exactly the point.
    assert "Нарушение субъектом предпринимательства срока" not in p1.display_text


def test_embed_text_carries_the_hierarchy_and_display_text_does_not():
    chunks, _, _ = _chunks("koap.ru.act.txt", max_article_tokens=180)
    chunk = chunks[0]
    assert "Глава 14" in chunk.embed_text
    assert "Глава 14" not in chunk.display_text


def test_russian_and_kazakh_chunk_to_identical_node_ids():
    """Alignment is a primary-key lookup only if the trees really match."""
    ru, _, _ = _chunks("koap.ru.act.txt", max_article_tokens=180)
    kk, _, _ = _chunks("koap.kk.act.txt", max_article_tokens=180)
    assert {c.node_id for c in ru} == {c.node_id for c in kk}


def test_table_survives_intact():
    chunks, _, _ = _chunks("koap.ru.act.txt", max_article_tokens=180)
    p2 = next(c for c in chunks if c.node_id == "koap/st278/p2")
    for row in ("Субъект малого предпринимательства", "60"):
        assert row in p2.display_text


def test_lead_in_is_empty_when_article_has_no_points():
    _, root, doc = _chunks("nk.ru.act.txt")
    article = next(n for n in root.articles() if n.id == "nk/st77")
    # st77 has points, so it does have structure; st76 lead-in should be empty
    # because its text starts directly with point 1.
    st76 = next(n for n in root.articles() if n.id == "nk/st76")
    assert article_lead_in(st76, Lang.RU) == ""
