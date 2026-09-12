"""Context packet assembly — what the model actually reads.

Everything the model needs to answer correctly is stated explicitly rather than
left to be inferred: which act, where in it, in force on which date, the full
article rather than the matched fragment, the referenced norms, the statutory
definitions. A model asked to reason about a bare paragraph of legal Russian
with no provenance will invent the provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..ingest.structure import estimate_tokens
from ..models import Lang, OPEN_ENDED
from .hydrate import HydratedHit

_L = {
    Lang.RU: {
        "source": "ИСТОЧНИК", "in_force": "Действует с", "until": "по",
        "current": "действующая редакция", "superseded": "УТРАТИЛА СИЛУ / прежняя редакция",
        "location": "Расположение", "article": "ТЕКСТ СТАТЬИ (полностью)",
        "refs": "УПОМИНАЕМЫЕ НОРМЫ", "defs": "ОПРЕДЕЛЕНИЯ",
        "matched": "НАЙДЕННЫЙ ФРАГМЕНТ",
        "repealed": "ВНИМАНИЕ: эта статья исключена / утратила силу",
    },
    Lang.KK: {
        "source": "ДЕРЕККӨЗ", "in_force": "Қолданылады", "until": "дейін",
        "current": "қолданыстағы редакция", "superseded": "КҮШІ ЖОЙЫЛҒАН / бұрынғы редакция",
        "location": "Орналасуы", "article": "БАПТЫҢ МӘТІНІ (толық)",
        "refs": "СІЛТЕМЕ ЖАСАЛҒАН НОРМАЛАР", "defs": "АНЫҚТАМАЛАР",
        "matched": "ТАБЫЛҒАН ҮЗІНДІ",
        "repealed": "НАЗАР АУДАРЫҢЫЗ: бұл бап алып тасталды / күші жойылды",
    },
}


@dataclass(slots=True)
class Packet:
    text: str
    sources: list[dict] = field(default_factory=list)
    token_estimate: int = 0
    truncated: bool = False
    dropped_sources: int = 0


def _validity_line(hh: HydratedHit, lang: Lang, as_of: date) -> str:
    labels = _L[lang]
    vf = hh.hit.valid_from.isoformat()
    if hh.hit.valid_to >= OPEN_ENDED:
        return f"{labels['in_force']} {vf} ({labels['current']})"
    return (f"{labels['in_force']} {vf} {labels['until']} "
            f"{hh.hit.valid_to.isoformat()} — {labels['superseded']}")


def render_source(index: int, hh: HydratedHit, lang: Lang, as_of: date) -> str:
    labels = _L[lang]
    head = f"[{labels['source']} {index}] {hh.hit.act_title} — {hh.hit.heading}"
    parts = [head, _validity_line(hh, lang, as_of)]
    if hh.hit.repealed:
        # An article whose entire body is "Исключена Законом РК от…" must not be
        # presented as an ordinary norm just because its validity range is open.
        parts.append(labels["repealed"])
    parts.append(f"{labels['location']}: {hh.hit.hierarchy_path}")
    if hh.hit.node_id != hh.hit.article_node_id:
        parts.append(f"{labels['matched']}: {hh.hit.node_id}")
    body = hh.article_text or hh.hit.display_text
    parts.append(f"{labels['article']}:\n{body}")

    refs = [s for s in hh.supporting if s.kind == "reference"]
    if refs:
        parts.append(labels["refs"] + ":")
        parts.extend(f"  • {s.label}:\n    {s.text}" for s in refs)

    defs = [s for s in hh.supporting if s.kind == "definition"]
    if defs:
        parts.append(labels["defs"] + ":")
        parts.extend(f"  • {s.label}: {s.text}" for s in defs)

    return "\n".join(parts)


def build_packet(hydrated: list[HydratedHit], lang: Lang, as_of: date,
                 max_tokens: int = 12_000) -> Packet:
    blocks: list[str] = []
    sources: list[dict] = []
    total = 0
    dropped = 0

    for i, hh in enumerate(hydrated, 1):
        block = render_source(i, hh, lang, as_of)
        cost = estimate_tokens(block)
        if total + cost > max_tokens and blocks:
            # Budget exhausted. Say so rather than silently truncating: a packet
            # that quietly dropped source 6 looks exactly like one that never
            # had it.
            dropped = len(hydrated) - i + 1
            break
        blocks.append(block)
        total += cost
        sources.append({
            "index": i,
            "chunk_id": hh.hit.chunk_id,
            "node_id": hh.hit.node_id,
            "article_node_id": hh.hit.article_node_id,
            "act_code": hh.hit.act_code,
            "lang": hh.hit.lang,
            "heading": hh.hit.heading,
            "valid_from": hh.hit.valid_from.isoformat(),
            "valid_to": hh.hit.valid_to.isoformat(),
            "is_current": hh.hit.valid_to >= OPEN_ENDED,
            "repealed": hh.hit.repealed,
            "release_id": hh.hit.release_id,
            "fused_score": round(hh.hit.fused_score, 6),
            "rerank_score": hh.hit.rerank_score,
            "provenance": hh.hit.provenance,
            "supporting": [
                {"kind": s.kind, "node_id": s.node_id, "label": s.label}
                for s in hh.supporting
            ],
        })

    return Packet(text="\n\n".join(blocks), sources=sources,
                  token_estimate=total, truncated=dropped > 0,
                  dropped_sources=dropped)
