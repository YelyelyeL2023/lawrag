"""Bitemporal storage: superseded law is closed, never deleted."""

from __future__ import annotations

from datetime import date


def test_both_redactions_are_kept(live_db):
    from lawrag.index.db import Database

    db = Database(live_db)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT valid_from, valid_to FROM node_text
             WHERE node_id = 'koap/st278' AND lang = 'ru'
             ORDER BY valid_from
            """
        )
        rows = cur.fetchall()

    assert len(rows) == 2
    assert rows[0]["valid_from"] == date(2024, 1, 1)
    assert rows[0]["valid_to"] == date(2025, 12, 31)
    assert rows[1]["valid_from"] == date(2026, 1, 1)
    assert rows[1]["valid_to"] == date(9999, 12, 31)


def test_ranges_do_not_overlap(live_db):
    from lawrag.index.db import Database

    db = Database(live_db)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.node_id, a.lang FROM node_text a
              JOIN node_text b
                ON a.node_id = b.node_id AND a.lang = b.lang
               AND a.valid_from < b.valid_from
               AND a.valid_to >= b.valid_from
            """
        )
        overlaps = cur.fetchall()
    assert overlaps == [], f"overlapping redactions: {overlaps}"


def test_article_lookup_is_point_in_time(live_db):
    from lawrag.index.db import Database

    db = Database(live_db)
    with db.connect() as conn, conn.cursor() as cur:
        for when, expected in [(date(2025, 3, 15), date(2024, 1, 1)),
                               (date(2026, 3, 15), date(2026, 1, 1))]:
            cur.execute(
                """
                SELECT valid_from FROM node_text
                 WHERE node_id = 'koap/st278' AND lang = 'ru'
                   AND valid_from <= %s AND valid_to >= %s
                """,
                (when, when),
            )
            rows = cur.fetchall()
            assert len(rows) == 1
            assert rows[0]["valid_from"] == expected
