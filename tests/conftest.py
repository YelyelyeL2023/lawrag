from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def _db_available(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3):
            return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def config():
    from lawrag.config import load_config

    return load_config(ROOT / "config" / "default.yaml")


@pytest.fixture(scope="session")
def live_db(config):
    """Ingest the sample corpus once into a real Postgres, or skip."""
    dsn = os.environ.get("LAWRAG_TEST_DSN", config.db.dsn)
    if not _db_available(dsn):
        pytest.skip(f"no Postgres at {dsn}; database-backed tests skipped")
    config.db.dsn = dsn

    from lawrag.ingest.pipeline import ingest_directory

    ingest_directory(config, SAMPLE, verify=False, reset=True)
    return config
