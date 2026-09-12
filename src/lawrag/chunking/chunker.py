"""Structural chunking.

Three separate questions, answered separately:

1. *What is the atomic unit?*  A structural node — never a token window. Every
   boundary this module emits coincides with a boundary the legislator drew.
2. *What text gets embedded?*  Hierarchy header + generated context line + body,
   so a fragment about "нарушение срока" still carries the code and chapter it
   lives in.
3. *What text reaches the model?*  Assembled at query time by the retrieval
   layer, not here. This module only stores the clean text to quote.

Collapsing those three into one string is the usual cause of "the retriever
found the right article but the answer was still wrong".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import ChunkConfig
from ..ids import chunk_id, node_article_id
from ..logging_setup import get_logger
from ..models import Chunk, Lang, Node, NodeKind, NodeText
from ..ingest import structure as S

log = get_logger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[А-ЯЁӘҒҚҢӨҰҮҺІA-Z\d])")

_LABEL = {
    Lang.RU: {
        NodeKind.SECTION: "Раздел", NodeKind.CHAPTER: "Глава",
        NodeKind.PARAGRAPH: "Параграф", NodeKind.ARTICLE: "Статья",
        NodeKind.POINT: "пункт", NodeKind.SUBPOINT: "подпункт", NodeKind.NOTE: "Примечание",
    },
    Lang.KK: {
        NodeKind.SECTION: "бөлім", NodeKind.CHAPTER: "тарау",
        NodeKind.PARAGRAPH: "параграф", NodeKind.ARTICLE: "бап",
        NodeKind.POINT: "тармақ", NodeKind.SUBPOINT: "тармақша", NodeKind.NOTE: "Ескерту",
    },
}


def label(kind: NodeKind, number: str | None, lang: Lang) -> str:
    word = _LABEL[lang][kind]
    if number is None:
        return word
    # Russian puts the label first, Kazakh attaches the number as a prefix.
    return f"{number}-{word}" if lang is Lang.KK else f"{word} {number}"


@dataclass(slots=True)
class ChunkerStats:
    articles: int = 0
    chunks: int = 0
    split_articles: int = 0
    oversize_kept_whole: int = 0
    sentence_split: int = 0


def hierarchy_path(node: Node, index: dict[str, Node], lang: Lang) -> str:
    """Human-readable breadcrumb, e.g. 'КоАП → Раздел 2 → Глава 14 → Статья 278'."""
    parts: list[str] = []
    cur: Node | None = node
    while cur is not None:
        text = cur.texts.get(lang)
        if cur.kind is NodeKind.ACT:
            parts.append((text.heading if text else cur.act_code) or cur.act_code)
        elif cur.kind in _LABEL[lang]:
            head = (text.heading if text else "") or ""
            piece = label(cur.kind, cur.number, lang)
            parts.append(f"{piece}. {head}".strip().rstrip(".") if head else piece)
        cur = index.get(cur.parent_id) if cur.parent_id else None
    return " → ".join(reversed(parts))


def build_embed_text(hierarchy: str, heading: str, context_sentence: str, body: str,
                     lead_in: str = "") -> str:
    """What the dense and sparse indexes actually see.

    The header is not decoration. Without it a point reading "в случаях,
    указанных в пункте 2, штраф удваивается" has no lexical overlap with any
    realistic user query, and no dense-vector neighbourhood worth having.

    ``lead_in`` carries the article's opening disposition — the clause before
    point 1 that states *which conduct* the numbered points then attach
    sanctions to. In KZ codes that clause is where the offence is actually
    defined, so a point chunk indexed without it describes a penalty for
    nothing.
    """
    blocks = [hierarchy]
    if heading:
        blocks.append(heading)
    if context_sentence:
        blocks.append(context_sentence)
    if lead_in:
        blocks.append(lead_in)
    blocks.append("---")
    blocks.append(body)
    return "\n".join(b for b in blocks if b)


def article_lead_in(article: Node, lang: Lang) -> str:
    """Text of an article that precedes its first numbered point."""
    text = article.texts.get(lang)
    if text is None:
        return ""
    body, _notes = S.split_notes(text.body)
    parts = S.split_points(body)
    if len(parts) <= 1:
        return ""
    first_number, first_body = parts[0]
    return first_body.strip() if first_number is None else ""


def _sentence_split(text: str, max_tokens: int) -> list[str]:
    """Last-resort split for a single oversized point.

    Only reached when one point exceeds the hard cap and contains no table. Cuts
    on sentence boundaries so no fragment starts mid-clause.
    """
    sentences = _SENTENCE_END.split(text)
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        stoken = S.estimate_tokens(sentence)
        if current and size + stoken > max_tokens:
            parts.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += stoken
    if current:
        parts.append(" ".join(current))
    return parts or [text]


class Chunker:
    def __init__(self, cfg: ChunkConfig) -> None:
        self.cfg = cfg
        self.stats = ChunkerStats()

    def chunk_act(self, root: Node, lang: Lang, release_id: str = "") -> list[Chunk]:
        index = {n.id: n for n in root.walk()}
        chunks: list[Chunk] = []
        for article in root.articles():
            chunks.extend(self._chunk_article(article, index, lang, release_id))
        return chunks

    # -- internals ---------------------------------------------------------

    def _chunk_article(self, article: Node, index: dict[str, Node], lang: Lang,
                       release_id: str) -> list[Chunk]:
        text = article.texts.get(lang)
        if text is None or not (text.body or text.heading):
            return []
        self.stats.articles += 1

        hierarchy = hierarchy_path(article, index, lang)
        heading = f"{label(NodeKind.ARTICLE, article.number, lang)}. {text.heading}".strip(". ")
        full = f"{heading}\n{text.body}".strip()
        tokens = S.estimate_tokens(full)

        # A short article is one chunk. Splitting it would only fragment a rule
        # that already fits.
        if tokens <= self.cfg.max_article_tokens:
            self.stats.chunks += 1
            chunk = self._make(article, article, lang, hierarchy, heading, text,
                               full, release_id)
            return [chunk]

        points = sorted(
            (c for c in article.children
             if c.kind in (NodeKind.POINT, NodeKind.NOTE) and not c.id.endswith("/amend")),
            key=lambda n: n.ordinal,
        )
        if not points:
            # Long article with no internal structure to cut on.
            self.stats.oversize_kept_whole += 1
            self.stats.chunks += 1
            return [self._make(article, article, lang, hierarchy, heading, text, full, release_id)]

        self.stats.split_articles += 1
        lead_in = article_lead_in(article, lang)
        out: list[Chunk] = []
        for point in points:
            ptext = point.texts.get(lang)
            if ptext is None or not ptext.body.strip():
                continue
            marker = label(point.kind, point.number, lang)
            body = f"{marker}. {ptext.body}".strip(". ") if point.kind is NodeKind.POINT else ptext.body

            # A rate table or an enumerated list of grounds is one legal
            # proposition. An oversized whole beats two mutilated halves.
            if (S.estimate_tokens(body) > self.cfg.hard_max_tokens
                    and not S.contains_table(body)):
                self.stats.sentence_split += 1
                pieces = _sentence_split(body, self.cfg.hard_max_tokens)
                for i, piece in enumerate(pieces, 1):
                    out.append(self._make(article, point, lang, hierarchy, heading, ptext,
                                          piece, release_id, part=i,
                                          part_count=len(pieces), lead_in=lead_in))
            else:
                out.append(self._make(article, point, lang, hierarchy, heading, ptext,
                                      body, release_id, lead_in=lead_in))
        self.stats.chunks += len(out)
        return out or [self._make(article, article, lang, hierarchy, heading, text, full, release_id)]

    def _make(self, article: Node, node: Node, lang: Lang, hierarchy: str, heading: str,
              text: NodeText, body: str, release_id: str, part: int = 0,
              part_count: int = 0, lead_in: str = "") -> Chunk:
        node_id = node.id if part == 0 else f"{node.id}~{part}"
        cid = chunk_id(node_id, lang.value, text.valid_from)
        display = body
        if part:
            display = f"{body}\n[фрагмент {part}/{part_count}]"
        return Chunk(
            id=cid,
            node_id=node_id,
            article_node_id=node_article_id(article.id),
            act_code=article.act_code,
            lang=lang,
            valid_from=text.valid_from,
            valid_to=text.valid_to,
            display_text=display,
            embed_text=build_embed_text(hierarchy, heading, "", display, lead_in),
            hierarchy_path=hierarchy,
            heading=heading,
            token_estimate=S.estimate_tokens(display),
            lead_in=lead_in,
            # A repeal stub ("Исключена Законом РК от…") stays indexed, because
            # users do ask about repealed articles — but the packet must say so
            # rather than presenting an empty article as if it still applied.
            repealed=S.is_repealed(body),
            release_id=release_id,
        )
