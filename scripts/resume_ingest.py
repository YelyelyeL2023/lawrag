"""Finish an ingest that was interrupted part-way.

`lawrag ingest` commits per act and per embedding batch, so a Ctrl-C leaves a
usable-but-incomplete corpus: every chunk that was written is still there, but
the chunks in the batch that was in flight have no embedding, and neither
`link_aligned_chunks()` nor `build_vector_index()` ever ran.

    python scripts/resume_ingest.py             # report only
    python scripts/resume_ingest.py --embed     # embed the chunks that lack one
    python scripts/resume_ingest.py --index     # build the HNSW index
    python scripts/resume_ingest.py --embed --index --link
"""

from __future__ import annotations

import argparse

from lawrag.config import load_config
from lawrag.index.db import Database
from lawrag.index.embed import build_embedder
from lawrag.index.indexer import BATCH, Indexer, _vec_literal


def report(db: Database) -> int:
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.act_code, c.lang,
                   count(*)                        AS chunks,
                   count(e.chunk_id)               AS embedded,
                   count(*) - count(e.chunk_id)    AS missing
              FROM chunk c
              LEFT JOIN chunk_embedding e ON e.chunk_id = c.id
             GROUP BY c.act_code, c.lang
             ORDER BY c.act_code, c.lang
            """
        )
        rows = cur.fetchall()
        cur.execute(
            "SELECT to_regclass('chunk_embedding_hnsw') IS NOT NULL AS present"
        )
        has_index = cur.fetchone()["present"]

    if not rows:
        print("no chunks in the database at all — nothing was committed.")
        return 0

    print(f"{'act':<16}{'lang':<6}{'chunks':>9}{'embedded':>10}{'missing':>9}")
    total_missing = 0
    for r in rows:
        total_missing += r["missing"]
        print(f"{r['act_code']:<16}{r['lang']:<6}{r['chunks']:>9,}"
              f"{r['embedded']:>10,}{r['missing']:>9,}")
    print(f"\nHNSW index: {'present' if has_index else 'MISSING'}")
    print(f"chunks without an embedding: {total_missing:,}")
    return total_missing


def embed_missing(cfg, db: Database) -> None:
    embedder = build_embedder(cfg.embedding)
    while True:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.embed_text
                  FROM chunk c
                  LEFT JOIN chunk_embedding e ON e.chunk_id = c.id
                 WHERE e.chunk_id IS NULL
                 LIMIT %s
                """,
                (BATCH,),
            )
            rows = cur.fetchall()
        if not rows:
            break
        vectors = embedder.encode([r["embed_text"] for r in rows])
        with db.connect() as conn, conn.cursor() as cur:
            for row, vec in zip(rows, vectors):
                cur.execute(
                    """
                    INSERT INTO chunk_embedding (chunk_id, model, dim, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        model = EXCLUDED.model, dim = EXCLUDED.dim,
                        embedding = EXCLUDED.embedding
                    """,
                    (row["id"], embedder.name, len(vec), _vec_literal(vec)),
                )
        print(f"embedded {len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--embed", action="store_true",
                    help="Embed chunks that have no embedding yet.")
    ap.add_argument("--link", action="store_true",
                    help="Re-run link_aligned_chunks() (ru/kk pairing).")
    ap.add_argument("--index", action="store_true",
                    help="Build the HNSW index over chunk_embedding.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db = Database(cfg)

    missing = report(db)

    if args.embed and missing:
        embed_missing(cfg, db)
    if args.link or args.index:
        indexer = Indexer(db, build_embedder(cfg.embedding) if args.embed else None)
        if args.link:
            print(f"linked {indexer.link_aligned_chunks():,} aligned chunks")
        if args.index:
            indexer.build_vector_index()
            print("HNSW index built")

    if args.embed or args.link or args.index:
        print()
        report(db)


if __name__ == "__main__":
    main()
