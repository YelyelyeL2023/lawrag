"""Stable identifier scheme.

Every identifier is derived from the legal structure, never from ingest order or
a database sequence, so re-ingesting a corpus snapshot reproduces exactly the
same IDs. That is what makes it possible to diff two snapshots, to cite an
article in an audit log, and to align the Russian and Kazakh texts of the same
norm.

    act code            koap
    article node        koap/st278
    point node          koap/st278/p2
    subpoint node       koap/st278/p2/sp1
    chunk               koap/st278/p2#ru@2026-01-01
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Article numbers in KZ codes are not plain integers: 278, 278-1, 278-1-2.
ARTICLE_NUMBER_RE = re.compile(r"^\d+(?:-\d+)*$")


def slugify(value: str) -> str:
    norm = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _SLUG_RE.sub("-", norm.lower()).strip("-")


def act_id(act_code: str) -> str:
    return act_code


def article_id(act_code: str, number: str) -> str:
    return f"{act_code}/st{number}"


def point_id(article: str, number: str) -> str:
    return f"{article}/p{number}"


def subpoint_id(point: str, number: str) -> str:
    return f"{point}/sp{number}"


def chunk_id(node_id: str, lang: str, valid_from: date) -> str:
    return f"{node_id}#{lang}@{valid_from.isoformat()}"


def parse_chunk_id(cid: str) -> tuple[str, str, date]:
    node, rest = cid.split("#", 1)
    lang, iso = rest.split("@", 1)
    return node, lang, date.fromisoformat(iso)


def node_article_id(node_id: str) -> str:
    """Collapse a point/subpoint node id back to its owning article."""
    parts = node_id.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else node_id


def content_sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()
