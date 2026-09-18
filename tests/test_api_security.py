"""Who may call the API, and what survives a restart."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LAWRAG_CONFIG", str(ROOT / "config" / "local.yaml"))
    from lawrag.api import console, main
    from lawrag.ingest import pipeline

    monkeypatch.setattr(main.cfg, "data_dir", str(tmp_path))
    monkeypatch.setattr(console, "_jobs", {})
    monkeypatch.setattr(console, "_jobs_loaded", False)

    # These tests are about who may start an import and what survives a
    # restart, not about the pipeline. The real ingest_directory opens the
    # database before it parses anything, and on a machine with no Postgres
    # that connection hangs for the TCP timeout while the next test waits on
    # the import lock and gets a 409. Stand in a pipeline that fails the way a
    # bad upload does, immediately.
    def _refuse(cfg, path, **kwargs):
        raise ValueError(f"not a source: {sorted(p.name for p in Path(path).iterdir())}")

    monkeypatch.setattr(pipeline, "ingest_directory", _refuse)

    class _NoDatabase:
        def __init__(self, cfg):
            pass

        def stats(self):
            raise RuntimeError("no database in this test")

    monkeypatch.setattr(main, "Database", _NoDatabase)     # /health answers 503 at once
    yield TestClient(main.app), tmp_path
    # Let the worker thread finish before monkeypatch undoes the stub.
    _wait_for_idle()


def _wait_for_idle():
    import time

    from lawrag.api import console

    for _ in range(200):
        if not console._ingest_lock.locked():
            return
        time.sleep(0.05)


def _upload():
    return {"files": [{"name": "x.act.txt", "content": "not a source"}],
            "verify": False, "reset": False, "build_index": False}


def test_token_is_required_everywhere_when_set(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("LAWRAG_API_TOKEN", "s3cret")
    assert c.get("/health").status_code != 401          # liveness stays open
    assert c.get("/api/ingest/jobs").status_code == 401
    assert c.get("/api/ingest/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/ingest/jobs", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_without_token_imports_are_refused_from_other_machines(client, monkeypatch):
    from lawrag.api import security

    c, _ = client
    monkeypatch.delenv("LAWRAG_API_TOKEN", raising=False)
    monkeypatch.delenv("LAWRAG_ALLOW_REMOTE_WRITES", raising=False)
    monkeypatch.setattr(security, "_is_loopback", lambda host: False)
    r = c.post("/api/ingest", json=_upload())
    assert r.status_code == 403
    # Reading is still allowed.
    assert c.get("/api/ingest/jobs").status_code == 200

    monkeypatch.setenv("LAWRAG_ALLOW_REMOTE_WRITES", "1")
    _wait_for_idle()
    assert c.post("/api/ingest", json=_upload()).status_code == 200


def test_ingest_jobs_survive_a_restart(client, monkeypatch):
    from lawrag.api import console

    c, data_dir = client
    monkeypatch.delenv("LAWRAG_API_TOKEN", raising=False)
    _wait_for_idle()
    job = c.post("/api/ingest", json=_upload()).json()

    record = data_dir / "uploads" / job["id"] / "job.json"
    for _ in range(200):                     # the worker thread writes on finish
        if record.exists() and json.loads(record.read_text())["status"] != "running":
            break
        import time; time.sleep(0.05)
    assert json.loads(record.read_text())["status"] == "failed"   # not a real source

    # A new process: nothing in memory.
    monkeypatch.setattr(console, "_jobs", {})
    monkeypatch.setattr(console, "_jobs_loaded", False)
    ids = [j["id"] for j in c.get("/api/ingest/jobs").json()]
    assert job["id"] in ids


def test_a_job_left_running_is_reported_as_interrupted(client, monkeypatch):
    from lawrag.api import console

    c, data_dir = client
    monkeypatch.delenv("LAWRAG_API_TOKEN", raising=False)
    stale = data_dir / "uploads" / "deadbeef0000"
    stale.mkdir(parents=True)
    (stale / "job.json").write_text(json.dumps({
        "id": "deadbeef0000", "path": str(stale), "started_at": "2026-09-01T00:00:00+00:00",
        "files": ["a.act.txt"], "options": {}, "status": "running", "log": []}))
    job = c.get("/api/ingest/jobs/deadbeef0000").json()
    assert job["status"] == "interrupted"
