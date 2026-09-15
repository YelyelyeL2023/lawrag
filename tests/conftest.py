from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
TEST_DSN_ENV = "LAWRAG_TEST_DSN"


def _db_available(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3):
            return True
    except Exception:  # noqa: BLE001
        return False


def _create_database(dsn: str) -> None:
    """Create the database named in *dsn* if the server is up but it is not.

    The suite ingests with reset=True, so it must never share a database with a
    corpus anyone cares about — LAWRAG_TEST_DSN points somewhere else on the
    same server. That database has to exist before the schema can be applied,
    and it will not on a Postgres volume that was initialised before the test
    database was a thing. Creating it here keeps the guarantee in the place that
    depends on it, rather than in an initdb script that runs once per volume.

    Silent on failure: the caller re-checks connectivity and skips.
    """
    from urllib.parse import urlsplit, urlunsplit

    try:
        import psycopg

        parts = urlsplit(dsn)
        name = parts.path.lstrip("/")
        if not name:
            return
        admin = urlunsplit(parts._replace(path="/postgres"))
        with psycopg.connect(admin, connect_timeout=3, autocommit=True) as conn:
            row = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
            ).fetchone()
            if not row:
                # An identifier, not a value, so it cannot be parameterised —
                # quote it and double any embedded quote.
                conn.execute('CREATE DATABASE "{}"'.format(name.replace('"', '""')))
    except Exception:  # noqa: BLE001
        return


@pytest.fixture(scope="session")
def config():
    from lawrag.config import load_config

    return load_config(ROOT / "config" / "default.yaml")


@pytest.fixture(scope="session")
def live_db(config):
    """Ingest the sample corpus once into a real Postgres, or skip.

    Only LAWRAG_TEST_DSN is honoured, never db.dsn: the suite drops and
    recreates the schema, and the configured DSN is a shared corpus database.
    """
    dsn = os.environ.get(TEST_DSN_ENV)
    if not dsn:
        pytest.skip(f"{TEST_DSN_ENV} not set; database-backed tests skipped")
    if not _db_available(dsn):
        _create_database(dsn)
    if not _db_available(dsn):
        pytest.skip(f"no Postgres at {dsn}; database-backed tests skipped")
    config.db.dsn = dsn

    from lawrag.ingest.pipeline import ingest_directory

    ingest_directory(config, SAMPLE, verify=False, reset=True)
    return config
