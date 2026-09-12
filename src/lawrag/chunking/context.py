"""Per-chunk context sentences (offline batch job).

Contextual retrieval: before embedding a chunk, have a small local model write
one or two sentences situating it inside its act, and prepend that to the
embedded text. It is the single largest retrieval gain per unit of engineering
effort, and it costs nothing at query time because it runs once at ingest.

On an 8 GB card this is a *batch* job, not an interactive one. Use the small
model (``llm.context_model``), run it overnight, and cache by content hash so a
re-ingest of an unchanged article does not pay for it twice.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ids import content_sha256
from ..logging_setup import get_logger
from ..models import Chunk, Lang
from .chunker import build_embed_text

log = get_logger(__name__)

_PROMPT = {
    Lang.RU: (
        "Ниже — фрагмент нормативного правового акта Республики Казахстан и его "
        "положение в структуре акта.\n\n"
        "{hierarchy}\n\n<фрагмент>\n{body}\n</фрагмент>\n\n"
        "Напиши ОДНО предложение (максимум два), которое объясняет, что регулирует "
        "этот фрагмент и к каким ситуациям он применяется. Используй терминологию "
        "фрагмента. Не цитируй его дословно, не добавляй пояснений и не пиши ничего, "
        "кроме этого предложения."
    ),
    Lang.KK: (
        "Төменде Қазақстан Республикасының нормативтік құқықтық актісінің үзіндісі "
        "және оның акт құрылымындағы орны берілген.\n\n"
        "{hierarchy}\n\n<үзінді>\n{body}\n</үзінді>\n\n"
        "Осы үзінді нені реттейтінін және қандай жағдайларға қолданылатынын "
        "түсіндіретін БІР сөйлем (ең көбі екі) жаз. Үзіндінің терминологиясын "
        "пайдалан. Оны сөзбе-сөз көшірме, басқа ештеңе жазба."
    ),
}


class ContextCache:
    """Content-addressed cache so re-ingests skip unchanged articles."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        self._data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")


def generate_context_sentences(chunks: list[Chunk], llm, cache_path: Path,
                               model: str | None = None) -> list[Chunk]:
    """Fill ``context_sentence`` and rebuild ``embed_text`` in place.

    A failure on one chunk is logged and skipped rather than aborting the batch:
    a corpus that is 98% enriched is strictly better than one that is 0%
    enriched because article 4,112 tripped a timeout at 3am.
    """
    cache = ContextCache(cache_path)
    done = 0
    for i, chunk in enumerate(chunks, 1):
        key = content_sha256(chunk.hierarchy_path, chunk.display_text, chunk.lang.value)
        cached = cache.get(key)
        if cached is None:
            prompt = _PROMPT[chunk.lang].format(
                hierarchy=chunk.hierarchy_path, body=chunk.display_text[:4000]
            )
            try:
                cached = llm.complete(prompt, max_tokens=160, model=model).strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("context sentence failed for %s: %s", chunk.id, exc)
                continue
            cache.put(key, cached)
            done += 1
            if done % 50 == 0:
                cache.save()
                log.info("context sentences: %d generated (%d/%d chunks seen)",
                         done, i, len(chunks))
        chunk.context_sentence = cached
        # ``lead_in`` must be passed through. Rebuilding embed_text without it
        # strips the article's opening disposition from every point chunk, so
        # the job advertised as the largest retrieval gain would quietly make
        # point retrieval worse.
        chunk.embed_text = build_embed_text(
            chunk.hierarchy_path, chunk.heading, cached, chunk.display_text,
            chunk.lead_in,
        )
    cache.save()
    log.info("context sentences complete: %d generated, %d from cache",
             done, len(chunks) - done)
    return chunks
