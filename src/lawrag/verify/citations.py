"""Mechanical citation verification.

Everything here is deterministic code, not a second model call. The point is to
catch the failure that a non-lawyer user cannot catch: a citation that looks
right and is not. Four checks, in order of how badly they bite:

``resolvable``  the cited article exists in the corpus at all
``retrieved``   it was actually in the packet — a citation to a real article the
                model was never shown is recall from training data, not grounding
``in_force``    it was in force on the as-of date
``quoted``      the verbatim quote really appears in the cited text

A failure downgrades the answer to "here are the relevant provisions" rather
than letting a wrong conclusion through with a plausible footnote attached.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

from ..index.db import Database
from ..logging_setup import get_logger
from ..models import OPEN_ENDED

log = get_logger(__name__)

# A model reproducing a quote will normalise whitespace, quote marks and dashes.
# None of that is a hallucination, so comparison happens on a normalised form.
_WS = re.compile(r"\s+")
_PUNCT = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"',
                        "—": "-", "–": "-", " ": " "})

MIN_QUOTE_CHARS = 12


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT).lower()
    return _WS.sub(" ", text).strip()


@dataclass(slots=True)
class CitationCheck:
    raw: dict
    node_id: str | None = None
    resolvable: bool = False
    retrieved: bool = False
    in_force: bool = False
    quoted: bool = False
    superseded: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.resolvable and self.retrieved and self.in_force and self.quoted


@dataclass(slots=True)
class Verdict:
    checks: list[CitationCheck] = field(default_factory=list)
    passed: bool = True
    downgrade: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.checks:
            return "no_citations"
        good = sum(1 for c in self.checks if c.ok)
        return f"{good}/{len(self.checks)} citations verified"


class CitationVerifier:
    def __init__(self, db: Database) -> None:
        self.db = db

    def verify(self, citations: list[dict], sources: list[dict],
               as_of: date) -> Verdict:
        verdict = Verdict()
        by_index = {s["index"]: s for s in sources}
        retrieved_articles = {s["article_node_id"] for s in sources}
        # The full text the model was shown, per article, for quote checking.
        texts = self._article_texts(retrieved_articles, as_of)

        for raw in citations:
            check = CitationCheck(raw=raw)
            node_id = self._resolve(raw, by_index)
            check.node_id = node_id

            if node_id is None:
                check.problems.append(
                    "citation is inconsistent or does not name a resolvable article "
                    "(the act/article given does not match the source cited)")
                verdict.checks.append(check)
                continue
            check.resolvable = node_id in texts or self._exists(node_id)

            check.retrieved = node_id in retrieved_articles
            if not check.retrieved:
                check.problems.append(
                    f"{node_id} was not among the retrieved sources — "
                    "the model produced it from memory")

            row = texts.get(node_id)
            if row is not None:
                check.in_force = bool(row["in_force"])
                check.superseded = row["valid_to"] < OPEN_ENDED
                if not check.in_force:
                    check.problems.append(
                        f"{node_id} was not in force on {as_of.isoformat()}")
                check.quoted = self._quote_matches(raw.get("quote", ""), row["text"])
                if not check.quoted:
                    if self._quote_matches(raw.get("quote", ""), row["all_text"]):
                        check.problems.append(
                            "quote is from a different redaction than the one in "
                            f"force on {as_of.isoformat()}")
                    else:
                        check.problems.append(
                            "quote does not appear verbatim in the cited text")
            else:
                check.problems.append(f"no text on file for {node_id}")

            verdict.checks.append(check)

        failures = [c for c in verdict.checks if not c.ok]
        verdict.passed = not failures
        if failures:
            verdict.downgrade = True
            verdict.reasons = [p for c in failures for p in c.problems]
            log.warning("citation verification failed: %s", "; ".join(verdict.reasons))
        return verdict

    # ---------------------------------------------------------------------
    @staticmethod
    def _resolve(raw: dict, by_index: dict[int, dict]) -> str | None:
        """Map a model-produced citation onto a node id.

        The source index is resolved first, but it is NOT trusted on its own.
        Taking the index and discarding the model's own act/article fields lets
        a citation carry a real, verbatim quote from source 1 while pointing the
        user at an article that does not exist — the quote check passes, the
        retrieval check passes, and a fabricated pinpoint is printed as verified.
        The model's own claim must agree with the source it says it is citing.
        """
        index = raw.get("source")
        if isinstance(index, int) and index in by_index:
            node_id = by_index[index]["article_node_id"]
            article = str(raw.get("article") or "").strip()
            if article and not node_id.endswith(f"/st{article}"):
                return None
            act = (raw.get("act_code") or "").strip()
            if act and not node_id.startswith(f"{act}/"):
                return None
            return node_id
        act = (raw.get("act_code") or "").strip()
        article = str(raw.get("article") or "").strip()
        if act and article:
            return f"{act}/st{article}"
        # Last resort: a bare article number that matches exactly one source.
        if article:
            matches = {s["article_node_id"] for s in by_index.values()
                       if s["article_node_id"].endswith(f"/st{article}")}
            if len(matches) == 1:
                return matches.pop()
        return None

    def _exists(self, node_id: str) -> bool:
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM node WHERE id = %s", (node_id,))
            return cur.fetchone() is not None

    def _article_texts(self, node_ids: set[str], as_of: date) -> dict[str, dict]:
        """All redactions of each article, grouped by validity range.

        Deliberately NOT filtered by date in SQL. Filtering there would make the
        in-force check tautological — every row returned would be in force by
        construction, and a citation to a repealed article would be reported as
        verified rather than caught.
        """
        if not node_ids:
            return {}
        with self.db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT node_id, lang, heading, body, valid_from, valid_to
                  FROM node_text
                 WHERE node_id = ANY(%s)
                 ORDER BY valid_from
                """,
                (sorted(node_ids),),
            )
            rows = cur.fetchall()

        grouped: dict[str, dict[tuple, dict]] = {}
        for row in rows:
            span = (row["valid_from"], row["valid_to"])
            # Both language faces count as the cited text: a Kazakh quote of a
            # Russian-retrieved article is still a faithful quote.
            entry = grouped.setdefault(row["node_id"], {}).setdefault(
                span, {"text": "", "valid_from": span[0], "valid_to": span[1]}
            )
            entry["text"] += f"\n{row['heading']}\n{row['body']}"

        out: dict[str, dict] = {}
        for node_id, spans in grouped.items():
            covering = [e for e in spans.values()
                        if e["valid_from"] <= as_of <= e["valid_to"]]
            chosen = covering[0] if covering else max(
                spans.values(), key=lambda e: e["valid_from"]
            )
            # The quote is checked against every redaction: quoting a norm
            # accurately but from the wrong period is an in-force failure, not a
            # quotation failure, and the two must be reported separately.
            chosen = dict(chosen)
            chosen["all_text"] = "\n".join(e["text"] for e in spans.values())
            chosen["in_force"] = bool(covering)
            out[node_id] = chosen
        return out

    @staticmethod
    def _quote_matches(quote: str, source_text: str) -> bool:
        q = normalize(quote)
        if len(q) < MIN_QUOTE_CHARS:
            # Too short to verify meaningfully; treat as unsupported rather than
            # waving it through on a coincidental substring match.
            return False
        return q in normalize(source_text)
